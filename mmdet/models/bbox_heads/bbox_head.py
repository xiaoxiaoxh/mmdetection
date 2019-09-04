import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import time
import math
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from ..utils import bias_init_with_prob


@HEADS.register_module
class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 samples_per_cls_file=None,
                 init_cls_prob=None,
                 **kwargs):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False
        self.use_cos_cls_fc = False  # not use cosine classifier by default
        self.init_cls_prob = init_cls_prob  # init_cls_prob
        if samples_per_cls_file and osp.exists(samples_per_cls_file):  # add samples_per_cls_file
            with open(samples_per_cls_file, 'r') as f:
                self.samples_per_cls = torch.Tensor([int(line.strip()) for line in f.readlines()])
                self.prob_per_cls = self.samples_per_cls / torch.sum(self.samples_per_cls)
        else:
            self.samples_per_cls = None
            self.prob_per_cls = None

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

        self.iter = None  # add current iter
        self.max_iters = None
        self.epoch = None
        self.max_epochs = None

    def init_weights(self):
        if self.with_cls:
            if hasattr(self, 'cos_cls_fc_gamma') and self.use_cos_cls_fc:
                stdv = 1. / math.sqrt(self.fc_cls.weight.size(1))
                self.fc_cls.weight.data.uniform_(-stdv, stdv)
                nn.init.constant_(self.fc_cls.gamma, self.cos_cls_fc_gamma)
            else:
                nn.init.normal_(self.fc_cls.weight, 0, 0.01)
                if self.init_cls_prob is not None:
                    bias_cls = bias_init_with_prob(self.init_cls_prob)
                    nn.init.constant_(self.fc_cls.bias, bias_cls)
                else:
                    nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             img_meta=None,
             rcnn_train_cfg=None,
             img_ids=None,
             **kwargs):
        losses = dict()
        if cls_score is not None:
            if 'use_anno_info' in rcnn_train_cfg and \
                    rcnn_train_cfg.use_anno_info:

                device = cls_score.get_device()
                num_classes = cls_score.shape[1]
                if not isinstance(img_meta, list):
                    img_meta = [img_meta]
                assert img_ids is not None and \
                    'neg_category_ids' in img_meta[0] and \
                    'not_exhaustive_category_ids' in img_meta[0], \
                    'img_ids can not be none'
                pos_idxs = labels > 0
                neg_idxs = labels == 0
                pos_cls_loss = self.loss_cls(
                        cls_score[pos_idxs, :],
                        labels[pos_idxs],
                        label_weights[pos_idxs],
                        gamma=self.fc_cls.gamma if hasattr(self.fc_cls, 'gamma') else None,
                        avg_factor=None,
                        reduction_override='sum')
                neg_cls_loss = cls_score.new_zeros(1)
                if 'ignore_cls_weight' in rcnn_train_cfg:
                    ignore_cls_weight = rcnn_train_cfg.ignore_cls_weight
                else:
                    ignore_cls_weight = 0
                img_num = len(img_meta)
                for i in range(img_num):
                    neg_category_ids = torch.Tensor(img_meta[i]['neg_category_ids']).cuda().long()
                    not_exhaustive_cat_ids = torch.Tensor(
                        img_meta[i]['not_exhaustive_category_ids']).cuda().long()
                    neg_cls_weights = \
                        torch.ones(num_classes, device=device, dtype=torch.float32) * ignore_cls_weight
                    neg_cls_weights[neg_category_ids] = 1
                    neg_cls_weights[not_exhaustive_cat_ids] = 0
                    neg_cls_weights[0] = 1
                    img_neg_idxs = (img_ids == i) & neg_idxs
                    neg_cls_loss += self.loss_cls(
                        cls_score[img_neg_idxs, :],
                        labels[img_neg_idxs],
                        label_weights[img_neg_idxs],
                        gamma=self.fc_cls.gamma if hasattr(self.fc_cls, 'gamma') else None,
                        avg_factor=None,
                        cls_weight=neg_cls_weights,
                        reduction_override='sum')
                    del neg_cls_weights
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                losses['loss_cls'] = (pos_cls_loss + neg_cls_loss) / avg_factor
            else:
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    gamma=self.fc_cls.gamma if hasattr(self.fc_cls, 'gamma') else None,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
            if hasattr(self.fc_cls, 'gamma'):
                losses['cls_gamma'] = self.fc_cls.gamma
            # print('loss_cls: {}'.format(losses['loss_cls'].item()))
            losses['acc'] = accuracy(cls_score[label_weights > 0, :], labels[label_weights > 0])
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
