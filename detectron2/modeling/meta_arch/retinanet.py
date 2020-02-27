# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.logger import log_first_n

from ..anchor_generator import build_anchor_generator
from ..backbone import build_backbone
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["RetinaNet"]


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


@META_ARCH_REGISTRY.register()
class RetinaNet(nn.Module):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES  # 1
        self.in_features              = cfg.MODEL.RETINANET.IN_FEATURES  # ["p3", "p4", "p5", "p6", "p7"]
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA  # 0.25
        self.focal_loss_gamma         = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA  # 2.0
        self.smooth_l1_loss_beta      = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA  # 0.1
        self.cam_loss_weight          = cfg.MODEL.CAM_LOSS_WEIGHT   # 0.2

        # Inference parameters:
        self.score_threshold          = cfg.MODEL.RETINANET.SCORE_THRESH_TEST  # 0.05
        self.topk_candidates          = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST  # 1000
        self.nms_threshold            = cfg.MODEL.RETINANET.NMS_THRESH_TEST  # 0.5
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE  # 100
        # CAM auxiliary classification
        self.cam_aux                  = cfg.MODEL.CAM_CLASSIFIER  # True or False
        self.cam_norm_forward = cfg.MODEL.CAM_NORM_FORWARD
        # fmt: on

        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()  # {'p3': 256, h, w, 8}
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        if self.cam_aux:
            self.cam_cls = CAMClassifier(cfg, feature_shapes)
            self.ce_loss = nn.CrossEntropyLoss()

        self.head = RetinaNetHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)  # ImageList(tensor, image_sizes)
        image_labels = torch.as_tensor([x["image_label"] for x in batched_inputs], dtype=torch.long).to(self.device)

        if "instances" in batched_inputs[0]:
            if self.cam_aux and not self.cam_norm_forward:
                # 0: hsil, 1: norm or lsil
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs if x['image_label'] == 0]
            else:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]  # 取出各阶段的feature, 从p3到p7

        if self.cam_aux:
            cam_logits, features, heatmaps = self.cam_cls(features, image_labels)  # weighted hsil features
            if not self.cam_norm_forward:
                features = [feature[image_labels == 0] for feature in features]

        # print(features[0].size())
        # print(features[1].size())
        # print(features[2].size())
        # print(features[3].size())
        # print(features[4].size())
        # assert 1 == 0
        box_cls, box_delta = self.head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_instances)
            if self.cam_aux:
                return self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta, cam_logits, image_labels)
            return self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta)
        else:
            # when inference, all images are hsil, image are fed one by one
            results = self.inference(box_cls, box_delta, anchors, images)
            processed_results = []
            for idx, (results_per_image, input_per_image, image_size) in enumerate(zip(
                results, batched_inputs, images.image_sizes
            )):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                output = {
                    "instances": r
                }
                if self.cam_aux:
                    heatmaps_this_img = [heatmap[idx] for heatmap in heatmaps]
                    output['heatmaps'] = heatmaps_this_img
                processed_results.append(output)
            return processed_results

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas,
               cam_cls_logits=None, image_labels=None):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, num_foreground)

        # cam loss
        if self.cam_aux:
            cam_loss = 0
            # image_labels = torch.cat([image_labels, image_labels], 0)
            for cam_cls_logit in cam_cls_logits:
                cam_loss += self.ce_loss(cam_cls_logit, image_labels)
            cam_loss = cam_loss * self.cam_loss_weight
            return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg, "cam_loss": cam_loss}

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    @torch.no_grad()
    def get_ground_truth(self, anchors, targets):
        """
        Args:
            anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
        """
        gt_classes = []
        gt_anchors_deltas = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        # list[Tensor(R, 4)], one for each image

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = len(targets_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors_per_image.tensor, matched_gt_boxes.tensor
                )

                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors_per_image.tensor)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)
        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    def inference(self, box_cls, box_delta, anchors, images):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, anchors_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k (1000) top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold   # 0.05
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class RetinaNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels      = input_shape[0].channels  # 256
        num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES  # 1
        num_convs        = cfg.MODEL.RETINANET.NUM_CONVS  # 4
        prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB  # 0.01
        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg


class CAMClassifier(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        in_channels = input_shape[0].channels
        self.cam_num_classes = cfg.MODEL.RETINANET.CAM_NUM_CLASSES  # 2

        self.cam_cls = nn.Linear(in_channels, self.cam_num_classes, bias=False)
        # Initialization
        torch.nn.init.normal_(self.cam_cls.weight, mean=0, std=0.01)
        # # for multi fcs use
        # self.fc_p7 = nn.Linear(in_channels, self.cam_num_classes, bias=False)
        # self.fc_p6 = nn.Linear(in_channels, self.cam_num_classes, bias=False)
        # self.fc_p5 = nn.Linear(in_channels, self.cam_num_classes, bias=False)
        # self.fc_p4 = nn.Linear(in_channels, self.cam_num_classes, bias=False)
        # self.fc_p3 = nn.Linear(in_channels, self.cam_num_classes, bias=False)
        # self.multi_fcs = [self.fc_p3, self.fc_p4, self.fc_p5, self.fc_p6, self.fc_p7]
        # # Initialization
        # for fc in self.multi_fcs:
        #     torch.nn.init.normal_(fc.weight, mean=0, std=0.01)

        # # for gap and gmp use
        # self.gap_cls = nn.Linear(in_channels, self.cam_num_classes, bias=False)
        # self.gmp_cls = nn.Linear(in_channels, self.cam_num_classes, bias=False)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, bias=True)
        # self.relu = nn.ReLU(True)
        # # Initialization
        # torch.nn.init.normal_(self.gap_cls.weight, mean=0, std=0.01)
        # torch.nn.init.normal_(self.gmp_cls.weight, mean=0, std=0.01)
        # torch.nn.init.normal_(self.conv1x1.weight, mean=0, std=0.01)
        # torch.nn.init.constant_(self.conv1x1.bias, 0)

    def forward(self, features, labels):
        """

        :param features:
        :param labels:
        :return: heatmaps:
        """
        logits = []

        # for gap or gmp use
        for feature in features:
            feature = nn.functional.adaptive_avg_pool2d(feature, 1)
            # feature = nn.functional.adaptive_max_pool2d(feature, 1)
            logit = self.cam_cls(feature.view(feature.shape[0], -1))
            logits.append(logit)

        # self.cam_cls.parameters() -> [1(no bias, only weight) * num_cls * channels]
        hsil_weight = list(self.cam_cls.parameters())[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 1, 256, 1, 1
        res_features = []
        for feature in features:
            # TODO *hsil_weight or *(1+hsil_weight)
            tmp_feature = torch.zeros_like(feature)
            tmp_feature[labels == 0] = feature[labels == 0] * hsil_weight
            tmp_feature[labels == 1] = feature[labels == 1]
            res_features.append(tmp_feature)
        heatmaps = [torch.sum(feature, dim=1, keepdim=True) for feature in res_features]

        # # for multi fcs use
        # w_features = []
        # for feature, fc in zip(features, self.multi_fcs):   # feature in per level
        #     x = nn.functional.adaptive_avg_pool2d(feature, 1)
        #     logit = fc(x.view(x.shape[0], -1))
        #     logits.append(logit)
        #
        #     weight = list(fc.parameters())[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #     w_features.append(feature * weight)
        #
        # w_hsil_features = [w_feature[labels == 0] for w_feature in w_features]  # hsil label: 0
        # heatmaps = [torch.sum(w_hsil_feature, dim=1, keepdim=True) for w_hsil_feature in w_hsil_features]

        # #  for gap and gmp use
        # for feature in features:
        #     fea_gap = nn.functional.adaptive_avg_pool2d(feature, 1)
        #     fea_gmp = nn.functional.adaptive_max_pool2d(feature, 1)
        #     logit_gap = self.gap_cls(fea_gap.view(fea_gap.shape[0], -1))
        #     logit_gmp = self.gmp_cls(fea_gmp.view(fea_gmp.shape[0], -1))
        #     logit = torch.cat([logit_gap, logit_gmp], 0)
        #     logits.append(logit)
        #
        # w_gap = list(self.gap_cls.parameters())[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # w_gmp = list(self.gmp_cls.parameters())[0][0].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #
        # hsil_features = [feature[labels == 0] for feature in features]
        # w_gap_features = [feature * w_gap for feature in hsil_features]
        # w_gmp_features = [feature * w_gmp for feature in hsil_features]
        #
        # w_hsil_features = []
        # for w_gap_feature, w_gmp_feature in zip(w_gap_features, w_gmp_features):
        #     w_hsil_feature = torch.cat([w_gap_feature, w_gmp_feature], 1)
        #     w_hsil_features.append(self.relu(self.conv1x1(w_hsil_feature)))
        # heatmaps = [torch.sum(w_hsil_feature, dim=1, keepdim=True) for w_hsil_feature in w_hsil_features]

        return logits, res_features, heatmaps
