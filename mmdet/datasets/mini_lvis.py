from lvis import LVIS
from .registry import DATASETS
from .Lvis import LvisDataSet
from .coco import CocoDataset
import os.path as osp
import json

@DATASETS.register_module
class MiniLvisDataSet(LvisDataSet):
    """Only contains the same 80 classes as COCO dataset"""
    # TODO: fix the lvis-coco classes
    LVIS_TO_COCO = {'computer_mouse': 'mouse',
                    'cellphone': 'cell_phone',
                    'hair_dryer': 'hair_drier',
                    'sausage': 'hot_dog',  # not sure!!!
                    'laptop_computer': 'laptop',
                    'microwave_oven': 'microwave',
                    'toaster_oven': 'oven',
                    'flowerpot': 'potted_plant',  # not sure!!
                    'remote_control': 'remote',
                    'ski': 'skis',
                    'baseball': 'sports_ball',  # too many types of balls
                    'wineglass': 'wine_glass',
                    }
    def load_annotations(self, ann_file):
        self.lvis = LVIS(ann_file)
        COCO_CLASSES = sorted(list(CocoDataset.CLASSES))
        self.synonyms_classes = [value['synonyms'] for value in self.lvis.cats.values()]
        self.cat_ids = []
        self.CLASSES = []
        self.not_in_classes = COCO_CLASSES
        for id in self.lvis.get_cat_ids():
            for name in self.lvis.cats[id]['synonyms']:
                if name in self.LVIS_TO_COCO:
                    self.cat_ids.append(id)
                    self.CLASSES.append(name)
                    self.not_in_classes.remove(self.LVIS_TO_COCO[name])
                    break
                elif '(' not in name and name in COCO_CLASSES:
                    self.cat_ids.append(id)
                    self.CLASSES.append(name)
                    self.not_in_classes.remove(name)
                    break
                elif '_' in name:
                    new_name = name.split('_(')[0]
                    if new_name in COCO_CLASSES:
                        self.cat_ids.append(id)
                        self.CLASSES.append(name)
                        self.not_in_classes.remove(new_name)
                        break
        data_dir = osp.dirname(ann_file)
        with open(osp.join(data_dir, 'synonyms_classes.json'), 'w') as f:
            f.write(json.dumps(self.synonyms_classes, indent=2))
        with open(osp.join(data_dir, 'not_in_classes.json'), 'w') as f:
            f.write(json.dumps(self.not_in_classes, indent=2))
        with open(osp.join(data_dir, 'coco_classes.json'), 'w') as f:
            f.write(json.dumps(COCO_CLASSES, indent=2))
        with open(osp.join(data_dir, 'lvis_coco_classes.json'), 'w') as f:
            f.write(json.dumps(self.CLASSES, indent=2))
        self.CLASSES = tuple(self.CLASSES)
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.CLASSES = CocoDataset.CLASSES
        self.img_ids = self.lvis.get_img_ids()
        img_infos = []
        for i in self.img_ids:
            info = self.lvis.load_imgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = self.lvis.load_anns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

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
