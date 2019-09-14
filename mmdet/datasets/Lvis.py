from .custom import CustomDataset
from lvis import LVIS
from .registry import DATASETS
import mmcv
import warnings
import numpy as np
import os.path as osp
import mmcv_custom
from mmcv.parallel import DataContainer as DC
from .utils import random_scale, to_tensor
from imagecorruptions import corrupt


@DATASETS.register_module
class LvisDataSet(CustomDataset):
    def __init__(self, samples_per_cls_file=None, **kwargs):
        self.samples_per_cls_file = samples_per_cls_file
        super(LvisDataSet, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        self.lvis = LVIS(ann_file)
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.CLASSES = [_ for _ in self.cat_ids]
        self.cat_instance_count = [_ for _ in self.cat_ids]
        self.cat_image_count = [_ for _ in self.cat_ids]
        img_count_lbl = ["r", "c", "f"]
        self.freq_groups = [[] for _ in img_count_lbl]
        self.cat_group_idxs = [_ for _ in self.cat_ids]
        freq_group_count = {'f': 0,
                            'cf': 0}
        self.cat_fake_idxs = {'f': [-1 for _ in self.cat_ids],
                              'cf': [-1 for _ in self.cat_ids]}
        for value in self.lvis.cats.values():
            idx = value['id'] - 1
            self.CLASSES[idx] = value['name']
            self.cat_instance_count[idx] = value['instance_count']
            self.cat_image_count[idx] = value['image_count']
            group_idx = img_count_lbl.index(value["frequency"])
            self.freq_groups[group_idx].append(idx + 1)
            self.cat_group_idxs[idx] = group_idx
            if group_idx == 1:  # common
                freq_group_count['cf'] += 1
                self.cat_fake_idxs['cf'][idx] = freq_group_count['cf']
            elif group_idx == 2:  # freq
                freq_group_count['cf'] += 1
                freq_group_count['f'] += 1
                self.cat_fake_idxs['cf'][idx] = freq_group_count['cf']
                self.cat_fake_idxs['f'][idx] = freq_group_count['f']

        if self.samples_per_cls_file is not None:
            with open(self.samples_per_cls_file, 'w') as file:
                file.writelines(str(x)+'\n' for x in self.cat_instance_count)

        self.img_ids = self.lvis.get_img_ids()
        img_infos = []
        for i in self.img_ids:
            info = self.lvis.load_imgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx, freq_group_idxs=(0, 1, 2)):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = self.lvis.load_anns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask, freq_group_idxs=freq_group_idxs)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.lvis.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=True, freq_group_idxs=(0, 1, 2)):
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
            if self.cat_group_idxs[ann['category_id'] - 1] not in freq_group_idxs:
                continue
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.lvis.ann_to_mask(ann))
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

        ann_all = self.get_ann_info(idx)
        ann_f = self.get_ann_info(idx, freq_group_idxs=(2, ))
        ann_cf = self.get_ann_info(idx, freq_group_idxs=(1, 2))

        gt_bboxes_all = ann_all['bboxes']
        gt_bboxes_f = ann_f['bboxes'] if len(ann_f['bboxes']) > 0 else np.zeros((0, ))
        gt_bboxes_cf = ann_cf['bboxes'] if len(ann_cf['bboxes']) > 0 else np.zeros((0, ))

        gt_labels_all = ann_all['labels']
        gt_labels_f = ann_f['labels'] if len(ann_f['labels']) > 0 else np.zeros((0, ))
        gt_labels_cf = ann_cf['labels'] if len(ann_cf['labels']) > 0 else np.zeros((0, ))
        if self.with_crowd:
            gt_bboxes_ignore = ann_all['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes_all) == 0 and self.skip_img_without_anno:
            warnings.warn('Skip the image "%s" that has no valid gt bbox' %
                          osp.join(self.img_prefix, img_info['filename']))
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes_all, gt_labels_all = self.extra_aug(img, gt_bboxes_all, gt_labels_all)
            if len(gt_labels_f) > 0:
                _, gt_bboxes_f, gt_labels_f = self.extra_aug(img, gt_bboxes_f, gt_labels_f)
            if len(gt_labels_cf) > 0:
                _, gt_bboxes_cf, gt_labels_cf = self.extra_aug(img, gt_bboxes_cf, gt_labels_cf)

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

        gt_bboxes_all = self.bbox_transform(gt_bboxes_all, img_shape, scale_factor, flip)
        if len(gt_bboxes_f) > 0:
            gt_bboxes_f = self.bbox_transform(gt_bboxes_f, img_shape, scale_factor, flip)
        if len(gt_bboxes_cf) > 0:
            gt_bboxes_cf = self.bbox_transform(gt_bboxes_cf, img_shape, scale_factor, flip)

        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks_all = self.mask_transform(ann_all['masks'], pad_shape, scale_factor, flip)
            gt_masks_f = self.mask_transform(ann_f['masks'], pad_shape, scale_factor, flip) \
                if len(gt_labels_f) > 0 else np.zeros((0, ))
            gt_masks_cf = self.mask_transform(ann_cf['masks'], pad_shape, scale_factor, flip) \
                if len(gt_labels_cf) > 0 else np.zeros((0, ))

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
            )

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes_all)),
            gt_bboxes_f=DC(to_tensor(gt_bboxes_f)),
            gt_bboxes_cf=DC(to_tensor(gt_bboxes_cf)),
        )
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels_all))
            data['gt_labels_f'] = DC(to_tensor(gt_labels_f))
            data['gt_labels_cf'] = DC(to_tensor(gt_labels_cf))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks_all, cpu_only=True)
            data['gt_masks_f'] = DC(gt_masks_f, cpu_only=True)
            data['gt_masks_cf'] = DC(gt_masks_cf, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
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
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack([_proposal, score
                                       ]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data
