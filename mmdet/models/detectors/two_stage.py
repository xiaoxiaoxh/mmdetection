import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
            self.use_bbox = True
        else:
            self.use_bbox = False

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)
            self.use_mask = True
        else:
            self.use_mask = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_valid_idxs=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        if 'stage_gt_groups' in self.train_cfg.rcnn and \
                gt_valid_idxs is not None and \
                (self.use_bbox or self.use_mask):
            gt_groups = self.train_cfg.rcnn.stage_gt_groups
            assert isinstance(gt_groups, list) and len(gt_groups) > self.current_stage, \
                'stage_gt_groups must be a list of strings'
            cat_fake_idxs = img_meta[0]['cat_fake_idxs']  # {'f': [-1,-1,1,2...], 'cf': [-1,1,-1,2...]}
            freq_groups = img_meta[0]['freq_groups']  # [[0,2...], [2,4...], [3,7...]]
            freq_group_dict = img_meta[0]['freq_group_dict']  # {'rcf':(0,1,2), 'cf':(1,2), 'f':(2,)}
            gt_group = gt_groups[self.current_stage]  # 'rcf' or 'cf or 'f'
            assert gt_group in freq_group_dict and \
                gt_group in cat_fake_idxs and \
                gt_group in gt_valid_idxs[0], \
                'gt_group must be in {}'.format(list(freq_group_dict.keys()))
            gt_valid_idxs = [img_valid_idx[gt_group] for img_valid_idx in gt_valid_idxs]
            device = gt_bboxes[0].get_device()

            if gt_group != 'rcf':
                valid_img_idx = []
                for i in range(img.size(0)):
                    if len(gt_valid_idxs[i]) > 0:
                        valid_img_idx.append(i)
                if len(valid_img_idx) == 0:
                    return losses

                gt_labels = [gt_labels[i][gt_valid_idxs[i]] for i in valid_img_idx]
                if self.with_bbox:
                    gt_bboxes = [gt_bboxes[i][gt_valid_idxs[i], :] for i in valid_img_idx]
                if self.with_mask:
                    gt_masks = [gt_masks[i][gt_valid_idxs[i], :, :] for i in valid_img_idx]

                cat_fake_idxs = torch.Tensor(cat_fake_idxs[gt_group]).long().to(device)
                cat_fake_idxs = torch.cat([cat_fake_idxs.new_zeros(1), cat_fake_idxs], dim=0)

                freq_groups_all = []
                for group_idx in freq_group_dict[gt_group]:
                    freq_groups_all += freq_groups[group_idx]
                valid_cls = torch.Tensor(sorted(freq_groups_all)).long().to(device)
                valid_cls = torch.cat([valid_cls.new_zeros(1), valid_cls], dim=0)
                assert valid_cls.shape[0] == torch.max(cat_fake_idxs).item() + 1, \
                    'valid classes num must match max label value'

                img = img[valid_img_idx, :, :, :]
                x = [lvl[valid_img_idx, :, :, :] for lvl in x]
                img_meta = [img_meta[i] for i in valid_img_idx]
                if gt_bboxes_ignore:
                    gt_bboxes_ignore = [gt_bboxes_ignore[i] for i in valid_img_idx]
                proposal_list = [proposal_list[i] for i in valid_img_idx]
            else:
                cat_fake_idxs = None
                valid_cls = None
        else:
            cat_fake_idxs = None
            valid_cls = None

        # assign gts and sample proposals
        if (self.with_bbox or self.with_mask) and (self.use_bbox or self.use_mask):
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            sampling_img_id_list = []  # add sampling_img_id_list
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # add sampling_img_id_list
                sampling_img_id_list.append(torch.ones_like(sampling_result.bboxes[:, 0]).long() * i)
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox and self.use_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes,
                                                     gt_labels,
                                                     self.train_cfg.rcnn)
            sampling_img_ids = torch.cat(sampling_img_id_list, dim=0)  # add sampling_img_ids
            if cat_fake_idxs is not None:
                fake_labels = cat_fake_idxs[bbox_targets[0]]  # add fake labels
            else:
                fake_labels = None
            # add img_meta and train_cfg and sampling_img_ids
            loss_bbox = self.bbox_head.loss(cls_score if valid_cls is None else cls_score[:, valid_cls],
                                            bbox_pred,
                                            *bbox_targets,
                                            img_meta=img_meta,
                                            rcnn_train_cfg=self.train_cfg.rcnn,
                                            img_ids=sampling_img_ids,
                                            fake_labels=fake_labels)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask and self.use_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False, out_proposal=False, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        # TODO: support multiple images per GPU
        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            results = (bbox_results, )
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            results = (bbox_results, segm_results)

        if out_proposal:
            if rescale:
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                proposal_list[0][:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                proposal_list[0][:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)
                proposal_list[0][:, :4] /= scale_factor
            proposal_list = proposal_list[0].cpu().numpy()
            results = results + (proposal_list, )
        return results

    def aug_test(self, imgs, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
