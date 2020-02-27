import numpy as np
import cv2
import torch

from detectron2.utils.visualizer import Visualizer, _create_text_labels, GenericMask, ColorMode
from detectron2.engine.hooks import HookBase
from detectron2.structures import Instances
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.evaluation.evaluator import inference_context
from detectron2.utils.visualizer import VisImage


class MyVisualizer(Visualizer):
    """
    modify: Draw instance-level gt annotations on an image.
    """
    def __init__(self, img_rgb, metadata, scale=1.0):
        super().__init__(img_rgb, metadata, scale)

    def draw_instance_gts(self, gts):
        """
        Args:
            gts (Instances): the input of the model.
                Following fields will be used to draw:
                "gt_boxes", "gt_classes", "gt_masks" (or "gt_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = gts.gt_boxes if gts.has("gt_boxes") else None
        scores = gts.scores if gts.has("scores") else None
        classes = gts.gt_classes if gts.has("gt_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = gts.gt_keypoints if gts.has("gt_keypoints") else None

        if gts.has("gt_masks"):
            masks = np.asarray(gts.gt_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            assert gts.has("gt_masks"), "ColorMode.IMAGE_BW requires segmentations"
            self.output.img = self._create_grayscale_image(
                (gts.gt_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


class VisulizeHook(HookBase):
    """
    Visualize prediction result periodically.
    """

    def __init__(self, train_dataloader, test_dataloader, train_metadata, test_metadata, period):
        self._period = period
        self.train_iter = iter(train_dataloader)
        self.test_iter = iter(test_dataloader)
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata

    def make_vis_img(self, data_iter, metadata):
        """
        note the img size here:
        image will first be resized to pre-configed sized by mapper in dataloader,
        then the detector get the resized img and do inference on it, post-process in detector will resize the
        prediction in ori size,
        so the size of the prediction in 'output' is the same as ori size, while the size of the image in input['image']
        is not, which is modified by the mapper.
        """
        inputs = next(data_iter)
        hsil_inputs = [input for input in inputs if input["image_label"] == 0]  # hsil inputs
        input = hsil_inputs[0:1]
        output = self.trainer.model(input)  # has benn resized to ori size by detector

        #  detection image
        img, instances = get_gt_img_instances(input[0])

        visualizer = MyVisualizer(img, metadata, scale=1.2)
        gt_vis = visualizer.draw_instance_gts(instances)  # draw gt
        gt_img = torch.from_numpy(gt_vis.get_image()).permute(2, 0, 1)  # c, h, w

        pred_vis = visualizer.draw_instance_predictions(output[0]['instances'].to('cpu'))  # draw pred
        pred_img = torch.from_numpy(pred_vis.get_image()).permute(2, 0, 1)  # c, h, w
        vis_imgs = [gt_img, pred_img]

        cam_imgs = None
        if 'heatmaps' in output[0]:
            # cam image
            target_h, target_w = input[0]['height'], input[0]['width']
            heatmaps = output[0]["heatmaps"]
            img = VisImage(img).get_image()
            cam_imgs = []
            for h in heatmaps:
                cam = get_cam(tensor2numpy(h), size=(target_w, target_h))
                # cam_img = cam * 0.3 + img * 0.5
                cam_img = cv2.addWeighted(cam, 0.3, img, 0.5, 0)
                cam_img = torch.from_numpy(cam_img).permute(2, 0, 1)  # c, h, w
                cam_imgs.append(cam_img)
        return vis_imgs, cam_imgs

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0:
            with inference_context(self.trainer.model), torch.no_grad():
                # train_vis_img = self.make_vis_img(self.train_iter, self.train_metadata)
                # test_vis_img = self.make_vis_img(self.test_iter, self.test_metadata)
                train_vis_img, train_cam_img = self.make_vis_img(self.train_iter, self.train_metadata)
                test_vis_img, test_cam_img = self.make_vis_img(self.test_iter, self.test_metadata)

                storage = get_event_storage()
                vis_imgs = {
                    'train_gt': train_vis_img[0],
                    'train_pred': train_vis_img[1],
                    'test_gt': test_vis_img[0],
                    'tset_pred': test_vis_img[1],
                }
                if train_cam_img is not None:
                    for idx, (x, y) in enumerate(zip(train_cam_img, test_cam_img)):
                        key_x = 'train_cam_p' + str(idx+3)
                        key_y = 'test_cam_p' + str(idx+3)
                        vis_imgs[key_x] = x
                        vis_imgs[key_y] = y

                storage.put_imgs(vis_imgs)


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


def get_cam(x, size):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img,  dsize=size)
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
    return cam_img


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


class EventStorageWithVis(EventStorage):
    """
    add storage for images: train, valid or test image with gt and prediction
    """

    def __init__(self, start_iter):
        """
        vis imgs are put in imgs with name
        """
        super().__init__(start_iter)
        self.imgs = {}
        self.is_new = False

    def put_imgs(self, imgs):
        self.imgs = imgs
        self.is_new = True

    def get_imgs(self):
        self.is_new = False
        return self.imgs
