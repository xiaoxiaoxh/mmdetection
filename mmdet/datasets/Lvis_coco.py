from pycocotools.coco import COCO
from .registry import DATASETS
from .Lvis import LvisDataSet
import mmcv
import warnings
import numpy as np
import os.path as osp
import mmcv_custom
from mmcv.parallel import DataContainer as DC
from .utils import random_scale, to_tensor
from imagecorruptions import corrupt


@DATASETS.register_module
class LvisCocoDataSet(LvisDataSet):
    def load_annotations(self, ann_file):
        assert isinstance(ann_file, dict) and \
               'lvis' in ann_file and 'coco' in ann_file, \
               'ann_file must be a dict with coco and lvis'
        img_infos = super(LvisCocoDataSet, self).load_annotations(ann_file['lvis'])
        self.coco = COCO(ann_file['coco'])
        self.cat_ids_coco = self.coco.getCatIds()
        self.cat2label_coco = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids_coco)
        }
        return img_infos

    def get_ann_info_coco(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info_coco(ann_info, self.with_mask)

    def _parse_ann_info_coco(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label_coco[ann['category_id']])  # add for coco
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        if 'COCO' in img_info['filename']:
            img = mmcv_custom.imread(osp.join(self.img_prefix,
                                        img_info['filename'][
                                        img_info['filename'].find('COCO_val2014_') + len('COCO_val2014_'):]))
        else:
            img = mmcv_custom.imread(osp.join(self.img_prefix, img_info['filename']))
        # corruption
        if self.corruption is not None:
            img = corrupt(
                img,
                severity=self.corruption_severity,
                corruption_name=self.corruption)
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx, freq_groups=('rcf', 'cf', 'f'))
        ann_coco = self.get_ann_info_coco(idx)  # add for coco
        gt_bboxes = ann['bboxes']
        gt_bboxes_coco = ann_coco['bboxes']  # add for coco
        gt_labels = ann['labels']
        gt_labels_coco = ann_coco['labels']  # add for coco
        gt_valid_idxs = ann['gt_valid_idxs']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if (len(gt_bboxes) == 0 or len(gt_bboxes_coco) == 0) and self.skip_img_without_anno:
            warnings.warn('Skip the image "%s" that has no valid gt bbox' %
                          osp.join(self.img_prefix, img_info['filename']))
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)
            # add for coco
            _, gt_bboxes_coco, gt_labels_coco = self.extra_aug(img, gt_bboxes_coco, gt_labels_coco)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv_custom.imread(
                osp.join(self.seg_prefix,
                         img_info['filename'].replace('jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack([proposals, scores
                                   ]) if scores is not None else proposals

        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor, flip)
        gt_bboxes_coco = self.bbox_transform(gt_bboxes_coco, img_shape, scale_factor, flip)  # add for coco

        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape, scale_factor, flip)
            gt_masks_coco = self.mask_transform(ann_coco['masks'], pad_shape, scale_factor, flip)  # add for coco

        ori_shape = (img_info['height'], img_info['width'], 3)
        not_exhaustive_category_ids = img_info['not_exhaustive_category_ids']
        neg_category_ids = img_info['neg_category_ids']
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            not_exhaustive_category_ids=not_exhaustive_category_ids,
            neg_category_ids=neg_category_ids,
            # cat_group_idxs=self.cat_group_idxs,
            cat_instance_count=self.cat_instance_count,
            freq_groups=self.freq_groups,
            cat_fake_idxs=self.cat_fake_idxs,
            freq_group_dict=self.freq_group_dict,
            valid_cls_extra=self.cat_ids_coco
            )

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)),
            gt_bboxes_extra=DC(to_tensor(gt_bboxes_coco)),  # add for coco
            gt_valid_idxs=DC(gt_valid_idxs, cpu_only=True),
        )
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
            data['gt_labels_extra'] = DC(to_tensor(gt_labels_coco))  # add for coco
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
            data['gt_masks_extra'] = DC(gt_masks_coco, cpu_only=True)  # add for coco
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)

        return data

