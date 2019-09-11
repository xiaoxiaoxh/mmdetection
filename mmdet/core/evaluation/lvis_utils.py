import mmcv
import numpy as np
import logging
from lvis import LVIS, LVISEval, LVISResults
from lvis.eval import Params
from .recall import eval_recalls
from collections import defaultdict, OrderedDict
from terminaltables import AsciiTable

class ParamsCustom(Params):
    def __init__(self, iou_type):
        """Params for LVIS evaluation API."""
        super(ParamsCustom, self).__init__(iou_type)
        self.use_proposal = False


class LVISEvalCustom(LVISEval):
    def __init__(self, lvis_gt, lvis_dt, iou_type="segm"):
        """Constructor for LVISEval.
        Args:
            lvis_gt (LVIS class instance, or str containing path of annotation file)
            lvis_dt (LVISResult class instance, or str containing path of result file,
            or list of dict)
            iou_type (str): segm or bbox evaluation
        """
        self.logger = logging.getLogger(__name__)

        if iou_type not in ["bbox", "segm"]:
            raise ValueError("iou_type: {} is not supported.".format(iou_type))

        if isinstance(lvis_gt, LVIS):
            self.lvis_gt = lvis_gt
        elif isinstance(lvis_gt, str):
            self.lvis_gt = LVIS(lvis_gt)
        else:
            raise TypeError("Unsupported type {} of lvis_gt.".format(lvis_gt))

        if isinstance(lvis_dt, LVISResults):
            self.lvis_dt = lvis_dt
        elif isinstance(lvis_dt, (str, list)):
            # set max_dets=-1 to avoid ignoring
            self.lvis_dt = LVISResults(self.lvis_gt, lvis_dt, max_dets=-1)
        else:
            raise TypeError("Unsupported type {} of lvis_dt.".format(lvis_dt))

        # per-image per-category evaluation results
        self.eval_imgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = ParamsCustom(iou_type=iou_type)  # parameters
        self.results = OrderedDict()
        self.ious = {}  # ious between all gts and dts

        self.params.img_ids = sorted(self.lvis_gt.get_img_ids())
        self.params.cat_ids = sorted(self.lvis_gt.get_cat_ids())

    def _prepare(self):
        """Prepare self._gts and self._dts for evaluation based on params."""
        cat_ids = self.params.cat_ids if self.params.cat_ids else None

        gts = self.lvis_gt.load_anns(
            self.lvis_gt.get_ann_ids(img_ids=self.params.img_ids, cat_ids=cat_ids)
        )
        dts = self.lvis_dt.load_anns(
            self.lvis_dt.get_ann_ids(img_ids=self.params.img_ids,
                                     cat_ids=None if self.params.use_proposal else cat_ids)
        )
        # convert ground truth to mask if iou_type == 'segm'
        if self.params.iou_type == "segm":
            self._to_mask(gts, self.lvis_gt)
            self._to_mask(dts, self.lvis_dt)

        # set ignore flag
        for gt in gts:
            if "ignore" not in gt:
                gt["ignore"] = 0

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        # For federated dataset evaluation we will filter out all dt for an
        # image which belong to categories not present in gt and not present in
        # the negative list for an image. In other words detector is not penalized
        # for categories about which we don't have gt information about their
        # presence or absence in an image.
        img_data = self.lvis_gt.load_imgs(ids=self.params.img_ids)
        # per image map of categories not present in image
        img_nl = {d["id"]: d["not_exhaustive_category_ids"] for d in img_data}
        # per image list of categories present in image
        img_pl = defaultdict(set)
        for ann in gts:
            img_pl[ann["image_id"]].add(ann["category_id"])
        # per image map of categoires which have missing gt. For these
        # categories we don't penalize the detector for flase positives.
        self.img_nel = {d["id"]: d["not_exhaustive_category_ids"] for d in img_data}

        for dt in dts:
            img_id, cat_id = dt["image_id"], dt["category_id"]
            if self.params.use_proposal:  # for proposal eval
                for cat_id in img_pl[img_id]:
                    dt['category_id'] = cat_id
                    self._dts[img_id, cat_id].append(dt)
                continue
            elif cat_id not in img_nl[img_id] and cat_id not in img_pl[img_id]:
                continue
            self._dts[img_id, cat_id].append(dt)

        self.freq_groups = self._prepare_freq_group()

    def _summarize(
        self, summary_type, iou_thr=None, area_rng="all", freq_group_idx=None
    ):
        aidx = [
            idx
            for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]

        if summary_type == 'ap':
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                s = s[:, :, self.freq_groups[freq_group_idx], aidx]
            else:
                s = s[:, :, :, aidx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:  # add freq_group for recall
                s = s[:, self.freq_groups[freq_group_idx], aidx]
            else:
                s = s[:, :, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Compute and display summary metrics for evaluation results."""
        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        max_dets = self.params.max_dets

        self.results["AP"]   = self._summarize('ap')
        self.results["AP50"] = self._summarize('ap', iou_thr=0.50)
        self.results["AP75"] = self._summarize('ap', iou_thr=0.75)
        self.results["APs"]  = self._summarize('ap', area_rng="small")
        self.results["APm"]  = self._summarize('ap', area_rng="medium")
        self.results["APl"]  = self._summarize('ap', area_rng="large")
        self.results["APr"]  = self._summarize('ap', freq_group_idx=0)
        self.results["APc"]  = self._summarize('ap', freq_group_idx=1)
        self.results["APf"]  = self._summarize('ap', freq_group_idx=2)

        key = "AR@{}".format(max_dets)
        self.results[key] = self._summarize('ar')

        for area_rng in ["small", "medium", "large"]:
            key = "AR{}@{}".format(area_rng[0], max_dets)
            self.results[key] = self._summarize('ar', area_rng=area_rng)
        # add freq_group for recall
        for idx, freq_group in enumerate(self.params.img_count_lbl):
            key = "AR{}@{}".format(freq_group[0], max_dets)
            self.results[key] = self._summarize('ar', freq_group_idx=idx)


def lvis_eval(result_files, result_types, lvis, max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in [
            'proposal_fast', 'proposal', 'bbox', 'segm'
        ]

    if mmcv.is_str(lvis):
        lvis = LVIS(lvis)
    assert isinstance(lvis, LVIS)

    img_ids = lvis.get_img_ids()
    for res_type in result_types:
        result_file = result_files['proposal' if res_type == 'proposal_fast' else res_type]
        if isinstance(result_file, str):
            assert result_file.endswith('.json')

        iou_type = 'bbox' if res_type in ['proposal', 'proposal_fast'] else res_type
        lvisEval = LVISEvalCustom(lvis, result_file, iou_type)
        lvisEval.params.img_ids = img_ids
        if res_type == 'proposal_fast':
            lvis_fast_eval_recall(result_file, lvisEval, np.array(max_dets))
            continue
        elif res_type == 'proposal':
            lvisEval.params.use_proposal = True
            for max_det in max_dets:
                lvisEval.params.max_dets = max_det
                lvisEval.run()
                for area_rng in ["small", "medium", "large"]:
                    key = "AR{}@{}".format(area_rng[0], max_det)
                    print('{}={:.3f}'.format(key, lvisEval.get_results()[key]))
                freq_group = lvisEval.params.img_count_lbl
                for idx in range(len(freq_group)):
                    key = "AR{}@{}".format(freq_group[idx][0], max_det)
                    print('{}={:.3f}'.format(key, lvisEval.get_results()[key]))
                key = "AR@{}".format(max_det)
                print('{}={:.3f}'.format(key, lvisEval.get_results()[key]))
            continue
        lvisEval.run()
        print('-'*8+'{} results'.format(res_type)+'-'*8)
        lvisEval.print_results()


def lvis_fast_eval_recall(results,
                          lvisEval,
                          max_dets,
                          iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
        if isinstance(results[0], tuple):
            proposals = []
            for idx in range(len(results)):
                proposals.append(results[idx][-1])
            results = proposals
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
            format(type(results)))
    assert isinstance(results, list), \
        'results must be a list of numpy arrays, not {}'.format(type(results))

    img_ids = lvisEval.lvis_gt.get_img_ids()
    area_rngs = lvisEval.params.area_rng

    freq_groups = [[] for _ in lvisEval.params.img_count_lbl]
    cat_data = lvisEval.lvis_gt.load_cats(lvisEval.params.cat_ids)
    for idx, _cat_data in enumerate(cat_data):
        frequency = _cat_data["frequency"]
        freq_groups[lvisEval.params.img_count_lbl.index(frequency)].append(idx+1)

    gt_bboxes_all = defaultdict(list)
    dt_bboxes_all = defaultdict(list)
    for area_idx in range(len(area_rngs)):
        for group_idx in range(len(freq_groups)):
            gt_bboxes_all[group_idx, area_idx] = []
        dt_bboxes_all[area_idx] = []

    gt_bboxes = defaultdict(list)
    dt_bboxes = defaultdict(list)
    group_num = np.zeros((len(freq_groups), 1))
    for i in range(len(img_ids)):
        for area_idx in range(len(area_rngs)):
            for group_idx in range(len(freq_groups)):
                gt_bboxes[group_idx, area_idx] = []
            dt_bboxes[area_idx] = []

        ann_ids = lvisEval.lvis_gt.get_ann_ids(img_ids=[img_ids[i]])
        ann_info = lvisEval.lvis_gt.load_anns(ann_ids)
        det_ids = lvisEval.lvis_dt.get_ann_ids(img_ids=[img_ids[i]])
        det_info = lvisEval.lvis_dt.load_anns(det_ids)

        img_pl = set()
        for ann in ann_info:
            img_pl.add(ann["category_id"])
        img_pl_group = dict()
        for cat in img_pl:
            for group_idx, freq_group in enumerate(freq_groups):
                if cat in freq_group:
                    img_pl_group[cat] = group_idx
                    break

        for ann in ann_info:
            for area_idx, area_rng in enumerate(area_rngs):
                if area_rng[0] <= ann['area'] <= area_rng[1]:
                    group_idx = img_pl_group[ann['category_id']]
                    group_num[group_idx] += 1
                    x1, y1, w, h = ann['bbox']
                    gt_bboxes[group_idx, area_idx].append(
                        [x1, y1, x1 + w - 1, y1 + h - 1])

        for det in det_info:
            for area_idx, area_rng in enumerate(area_rngs):
                if area_rng[0] <= det['area'] <= area_rng[1]:
                    x1, y1, w, h = det['bbox']
                    dt_bboxes[area_idx].append(
                        [x1, y1, x1 + w - 1, y1 + h - 1, det['score']])

        for area_idx in range(len(area_rngs)):
            for group_idx in range(len(freq_groups)):
                if len(gt_bboxes[group_idx, area_idx]) == 0:
                    gt_bboxes_all[group_idx, area_idx].append(np.zeros((0, 4)))
                else:
                    gt_bboxes_all[group_idx, area_idx].append(
                        np.array(gt_bboxes[group_idx, area_idx], dtype=np.float32))
            if len(dt_bboxes[area_idx]) == 0:
                dt_bboxes_all[area_idx].append(np.zeros((0, 5)))
            else:
                dt_bboxes_all[area_idx].append(
                    np.array(dt_bboxes[area_idx], dtype=np.float32))

    ar_all = np.zeros((len(freq_groups), len(area_rngs), len(max_dets)))
    for group_idx in range(len(freq_groups)):
        for area_idx in range(len(area_rngs)):
            recalls = eval_recalls(
                gt_bboxes_all[group_idx, area_idx],
                dt_bboxes_all[area_idx],
                max_dets, iou_thrs, print_summary=False)
            ar_all[group_idx, area_idx, :] = recalls.mean(axis=1)
    ar_all = np.transpose(ar_all, (2, 0, 1))

    # print(group_num.tolist())
    recalls = np.zeros((max_dets.size, len(freq_groups) + len(area_rngs)))
    for det_num_idx, max_det in enumerate(max_dets):
        ar = ar_all[det_num_idx, :, :]
        ar_cat_all = np.sum(ar * group_num, axis=0) / np.sum(group_num)
        for area_idx in range(len(area_rngs)):
            recalls[det_num_idx, area_idx] = ar_cat_all[area_idx]  # all small meduim large
        for group_idx in range(len(freq_groups)):
            recalls[det_num_idx, group_idx + len(area_rngs)] = ar[group_idx, 0]  # rare common freq

    area_rng_name = ["all", "small", "medium", "large"]
    freq_group_name = ["rare", "common", "freq"]
    row_header = [''] + area_rng_name + freq_group_name
    table_data = [row_header]
    for i, num in enumerate(max_dets):
        row = [
            '{:.3f}'.format(val)
            for val in recalls[i, :].tolist()
        ]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print(table.table)

