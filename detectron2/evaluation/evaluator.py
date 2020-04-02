# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch
import cv2
import numpy as np
import os.path as osp

from detectron2.utils.comm import is_main_process
from detectron2.structures import Instances


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator, vis=False, vis_dir=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            if vis:
                vis_gt_pred_heatmap(inputs, outputs, vis_dir)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def vis_gt_pred_heatmap(inputs, outputs, vis_root, top_n=3):
    # gt name and bbox
    name = inputs[0]['file_name'].split('/')[-1].split('.')[0]
    img, gt_instances = get_gt_img_instances(inputs[0])  # img in rgb order
    gt_boxes = gt_instances.gt_boxes.tensor.int().tolist()

    # pred bbox and socre
    top_n = len(gt_boxes)
    pred_instances = outputs[0]['instances'].to('cpu')
    pred_boxes = pred_instances.pred_boxes.tensor.int().tolist()[:top_n]
    pred_scores = pred_instances.scores.tolist()[:top_n]

    # draw gt
    gt_color = (255, 0, 0)
    gt_img = img.copy()
    for x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(gt_img, (x1, y1), (x2, y2), gt_color, 3)
    cv2.imwrite(osp.join(vis_root, '{}_gt.jpg'.format(name)), gt_img[:, :, ::-1])

    # draw pred
    pred_color = (0, 255, 0)
    pred_img = img.copy()
    for (x1, y1, x2, y2), score in zip(pred_boxes, pred_scores):
        if score < 0.3:
            continue
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), pred_color, 3)
        cv2.putText(pred_img, '{:.2f}'.format(score), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
    cv2.imwrite(osp.join(vis_root, '{}_pred.jpg'.format(name)), pred_img[:, :, ::-1])

    # draw gt and pred on same img
    gt_pred_img = img.copy()
    for x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(gt_pred_img, (x1, y1), (x2, y2), gt_color, 3)
    for (x1, y1, x2, y2), score in zip(pred_boxes, pred_scores):
        if score < 0.3:
            continue
        cv2.rectangle(gt_pred_img, (x1, y1), (x2, y2), pred_color, 3)
        cv2.putText(gt_pred_img, '{:.2f}'.format(score), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
    cv2.imwrite(osp.join(vis_root, '{}_gt_pred.jpg'.format(name)), gt_pred_img[:, :, ::-1])

    # cam_heatmap
    if 'heatmaps' in outputs[0]:
        heatmaps = outputs[0]['heatmaps']
        h, w = img.shape[:2]
        for i, x in enumerate(heatmaps):
            heatmap = get_heatmap(x.cpu().numpy()[0], (w, h))
            heatmap_overlay = cv2.addWeighted(heatmap, 0.3, img, 0.5, 0)
            cv2.imwrite(osp.join(vis_root, '{}_heatmap_{}_cam.jpg'.format(name, i+3)), heatmap_overlay[:, :, ::-1])

    # raw_heatmap
    if 'raw_heatmaps' in outputs[0]:
        heatmaps = outputs[0]['raw_heatmaps']
        h, w = img.shape[:2]
        for i, x in enumerate(heatmaps):
            heatmap = get_heatmap(x.cpu().numpy()[0], (w, h))
            heatmap_overlay = cv2.addWeighted(heatmap, 0.3, img, 0.5, 0)
            cv2.imwrite(osp.join(vis_root, '{}_heatmap_{}_raw.jpg'.format(name, i+3)), heatmap_overlay[:, :, ::-1])


def get_gt_img_instances(input_dict):
    """
    image and instances in input_dict is resized by mapper, restore
    """
    instance = input_dict['instances']
    target_h, target_w = input_dict['height'], input_dict['width']
    h, w = instance.image_size

    img = input_dict['image'].permute(1, 2, 0).byte().numpy()[:, :, ::-1]  # h, w, c, has been resized by mapper
    target_img = cv2.resize(img, dsize=(target_w, target_h))  # resize to ori size

    scale_x, scale_y = (target_w / w, target_h / h)

    target_instances = Instances((target_h, target_w), **instance.get_fields())
    if target_instances.has('gt_boxes'):
        output_boxes = target_instances.gt_boxes
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(target_instances.image_size)
    target_instances = target_instances[output_boxes.nonempty()]

    return target_img, target_instances


def get_heatmap(x, size):
    x = x - np.min(x)
    heatmap = x / np.max(x)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap,  dsize=size)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap
