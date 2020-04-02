# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import time
from torch.nn.parallel import DistributedDataParallel
import logging
import os
from collections import OrderedDict
import torch

from fvcore.nn.precise_bn import get_bn_modules

from detectron2.data import MetadataCatalog
from detectron2.engine.train_loop import SimpleTrainer
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import EventWriter
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    build_detection_cam_aux_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    verify_results,
)

from lib.dataset_mapper import MyDatasetMapper
from lib.evaluator import MyEvaluator
from lib.tb_writer import TensorboardXWriter
from lib.visualization import VisulizeHook, EventStorageWithVis
from lib.dataset import regist_cervix_dataset


class CervixTrainer(SimpleTrainer):
    """
    SimpleTrainer:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.


    On the top of SimpleTrainer, add:
    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists.
    3. Register a few common hooks.

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):

    Examples:

    .. code-block:: python

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        super().__init__(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if cfg.MODEL.CAM_CLASSIFIER:
            return build_detection_cam_aux_train_loader(cfg, MyDatasetMapper(cfg, is_train=True))
        return build_detection_train_loader(cfg, MyDatasetMapper(cfg, is_train=True))
        # return build_detection_cam_aux_train_loader(cfg, MyDatasetMapper(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, eval_only=False):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        #  cfg.DATASETS.TEST[0] = 'cervix_valid'
        res = []
        if eval_only:
            datasets = cfg.DATASETS.EVAL
        else:
            datasets = cfg.DATASETS.TEST
        for dataset in datasets:
            res.append(build_detection_test_loader(cfg, dataset, MyDatasetMapper(cfg, is_train=False)))
        return res

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        output_folder = cfg.OUTPUT_DIR if output_folder is None else output_folder
        # return COCOEvaluator(dataset_name, cfg, True, output_dir=output_folder)
        return MyEvaluator(dataset_name, cfg, output_folder)

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:

        .. code-block:: python

            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]  # cfg.TEST.PRECISE_BN.ENABLED = False

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))  # 5000

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))  # cfg.TEST.EVAL_PERIOD = 5000

        # do visualization on train and valid data
        ret.append(VisulizeHook(self.build_train_loader(cfg),
                                self.build_test_loader(cfg)[0],
                                MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                                MetadataCatalog.get(cfg.DATASETS.TEST[0]),
                                cfg.SOLVER.VIS_PERIOD))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers()))
        return ret

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.

        Otherwise, load a model specified by the config.

        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (
                self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
                    "iteration", -1
                )
                + 1
        )

    def train(self):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        self.iter = self.start_iter

        with EventStorageWithVis(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            finally:
                self.after_train()

    @classmethod
    def test(cls, cfg, model, evaluators=None, eval_only=False, vis=False, vis_root=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger('detectron2.evaluation.testing')
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        data_loaders = cls.build_test_loader(cfg, eval_only)
        if eval_only:
            dataset_names = cfg.DATASETS.EVAL
        else:
            dataset_names = cfg.DATASETS.TEST
        for idx, (dataset_name, data_loader) in enumerate(zip(dataset_names, data_loaders)):
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator, vis, vis_root)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for [{}] in csv format:".format(dataset_name))
                print_csv_format(results_i)

        res = {}
        if len(results) == 1:
            res = list(results.values())[0]
        elif len(results) > 1:
            for dataset_name, result_i in results.items():
                if 'valid' in dataset_name:
                    res.update(result_i)
                else:
                    for k, v in result_i.items():
                        new_k = '{} {}'.format(dataset_name, k)
                        res[new_k] = v
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # torch.multiprocessing.set_sharing_strategy('file_system')
    cfg = setup(args)
    regist_cervix_dataset(cfg)

    if args.eval_only:
        model = CervixTrainer.build_model(cfg)
        # cfg.MODEL.WEIGHTS: 1999 means load from checkpoint model_0001999.pth in the OUTPUT_DIR
        if cfg.MODEL.RESUME_ITER > 0:
            cfg.defrost()
            resume_iter = cfg.MODEL.RESUME_ITER
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                             "model_{:07d}.pth".format(resume_iter))
            if not os.path.exists(cfg.MODEL.WEIGHTS):
                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                                 "model_{:07d}.pth".format(resume_iter - 1))
            cfg.freeze()
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if args.vis:
            vis_root = os.path.join(cfg.OUTPUT_DIR, 'vis')
            vis = True
            if not os.path.exists(vis_root):
                os.makedirs(vis_root)
        else:
            vis_root = None
            vis = False
        res = CervixTrainer.test(cfg, model, eval_only=True, vis=vis, vis_root=vis_root)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = CervixTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:  # False
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
