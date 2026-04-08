# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
import torch.nn.functional as F


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """BBox coder for NMS-free detectors.

    Args:
        pc_range (list[float]): Range of point cloud.
        voxel_size (list[float], optional): Size of each voxel. Default: None.
        post_center_range (list[float], optional): Center range used to filter
            decoded boxes. Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes based on
            score. Default: None.
        num_classes (int): Number of foreground classes. Default: 10.
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def _select_topk(self, cls_scores, bbox_preds):
        """Select top-k predictions from sigmoid classification logits.

        Args:
            cls_scores (Tensor): Classification logits with shape
                [num_query, num_classes].
            bbox_preds (Tensor): Normalized box predictions with shape
                [num_query, code_size].

        Returns:
            tuple[Tensor, Tensor, Tensor]: Top-k scores, labels, and aligned
            bbox predictions.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()       # (num_query, n_cls)
        query_scores, labels = cls_scores.max(dim=-1)
        scores, bbox_index = query_scores.topk(max_num)
        labels = labels[bbox_index]
        bbox_preds = bbox_preds[bbox_index]     # (max_num, code_size)  code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        return scores, labels, bbox_preds

    def _get_post_center_range(self, device):
        if self.post_center_range is None:
            return None
        if isinstance(self.post_center_range, torch.Tensor):
            return self.post_center_range.to(device=device)
        return torch.tensor(self.post_center_range, device=device)

    def _build_predictions(self, scores, labels, bbox_preds):
        """Build final prediction dict from filtered top-k candidates.

        Args:
            scores (Tensor): Top-k scores with shape [num_pred].
            labels (Tensor): Predicted labels with shape [num_pred].
            bbox_preds (Tensor): Normalized bbox predictions with shape
                [num_pred, code_size].

        Returns:
            dict: Prediction dict containing `bboxes`, `scores`, and `labels`.
        """
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)    # (max_num, 7/9)  code_size: (cx, cy, cz, dx, dy, dz, ry, vx, vy)
        final_scores = scores       # (max_num, )
        final_preds = labels        # (max_num, )

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        post_center_range = self._get_post_center_range(scores.device)
        if post_center_range is not None:
            
            mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)

            if self.score_threshold is not None:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,  # (N_pred, 7/9)
                'scores': scores,   # (N_pred, )
                'labels': labels    # (N_pred, )
            }

        else:
            predictions_dict = {
                'bboxes': final_box_preds,  # (max_num, 7/9)
                'scores': final_scores,     # (max_num, )
                'labels': final_preds       # (max_num, )
            }
        return predictions_dict

    def decode_single(self, cls_scores, bbox_preds):
        """Decode predictions for a single sample.

        Args:
            cls_scores (Tensor): Classification logits with shape
                [num_query, cls_out_channels].
            bbox_preds (Tensor): Normalized bbox predictions with shape
                [num_query, code_size]. The default box layout is
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).

        Returns:
            dict: Decoded prediction dict containing `bboxes`, `scores`, and
            `labels`.
        """
        scores, labels, bbox_preds = self._select_topk(cls_scores, bbox_preds)
        return self._build_predictions(scores, labels, bbox_preds)

    def decode(self, preds_dicts):
        """Decode predictions for a batch using the last decoder layer.

        Args:
            preds_dicts (dict): Model outputs containing:
                - `all_cls_scores` (Tensor): Classification logits of shape
                  [num_decoder_layers, bs, num_query, cls_out_channels].
                - `all_bbox_preds` (Tensor): Normalized bbox predictions of
                  shape [num_decoder_layers, bs, num_query, code_size].

        Returns:
            list[dict]: Decoded boxes.
        """
        # 选择最后一层的decode layer的输出
        all_cls_scores = preds_dicts['all_cls_scores'][-1]      # (B, N_query, n_cls)
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]      # (B, N_query, code_size)

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list


@BBOX_CODERS.register_module()
class NMSFreeClsCoder(NMSFreeCoder):
    """Variant of :class:`NMSFreeCoder` for softmax-based classification.

    This coder assumes the last classification channel is background and
    selects the best foreground class for each query before applying top-k.
    """

    def _select_topk(self, cls_scores, bbox_preds):
        """Select top-k predictions from softmax classification logits.

        Args:
            cls_scores (Tensor): Classification logits with shape
                [num_query, num_classes + 1], where the last channel is
                background.
            bbox_preds (Tensor): Normalized box predictions with shape
                [num_query, code_size].

        Returns:
            tuple[Tensor, Tensor, Tensor]: Top-k scores, labels, and aligned
            bbox predictions.
        """
        cls_scores, labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
        scores, indexs = cls_scores.view(-1).topk(self.max_num)
        labels = labels[indexs]
        bbox_preds = bbox_preds[indexs]
        return scores, labels, bbox_preds
