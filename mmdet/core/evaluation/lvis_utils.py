import mmcv
import numpy as np
from lvis import LVIS, LVISEval, LVISResults
from .recall import eval_recalls


class LVISEvalCustom(LVISEval):
    def __init__(self, lvis_gt, lvis_dt, iou_type="segm"):
        """Constructor for LVISEval.
        Args:
            lvis_gt (LVIS class instance, or str containing path of annotation file)
            lvis_dt (LVISResult class instance, or str containing path of result file,
            or list of dict)
            iou_type (str): segm or bbox evaluation
        """
        if isinstance(lvis_gt, LVIS):
            self.lvis_gt = lvis_gt
        elif isinstance(lvis_gt, str):
            self.lvis_gt = LVIS(lvis_gt)
        else:
            raise TypeError("Unsupported type {} of lvis_gt.".format(lvis_gt))

        if isinstance(lvis_dt, LVISResults):
            self.lvis_dt = lvis_dt
        elif isinstance(lvis_dt, (str, list)):
            self.lvis_dt = LVISResults(self.lvis_gt, lvis_dt, max_dets=0)
        else:
            raise TypeError("Unsupported type {} of lvis_dt.".format(lvis_dt))
        super(LVISEvalCustom, self).__init__(lvis_gt, lvis_dt, iou_type)


def lvis_eval(result_files, result_types, lvis, max_dets=(100, 300, 1000, 2000)):
    for res_type in result_types:
        assert res_type in [
            'proposal_fast', 'proposal', 'bbox', 'segm'
        ]

    if mmcv.is_str(lvis):
        lvis = LVIS(lvis)
    assert isinstance(lvis, LVIS)

    if result_types == ['proposal_fast']:
        ar = lvis_fast_eval_recall(result_files, lvis, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    img_ids = lvis.get_img_ids()
    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        iou_type = 'bbox' if res_type == 'proposal' else res_type
        lvisEval = LVISEvalCustom(lvis, result_file, iou_type)
        lvisEval.params.img_ids = img_ids
        if res_type == 'proposal':
            for max_det in max_dets:
                lvisEval.params.max_dets = max_det
                lvisEval.evaluate()
                lvisEval.accumulate()
                for area_rng in ["small", "medium", "large"]:
                    key = "AR_{}@{}".format(area_rng, max_det)
                    print('{}={:.3f}'.format(key, lvisEval._summarize('ar', area_rng=area_rng)))
                for idx in [0, 1, 2]:
                    freq_group = ['rare', 'common', 'freq']
                    key = "AR_{}@{}".format(freq_group[idx], max_det)
                    print('{}={:.3f}'.format(key, lvisEval._summarize('ar', freq_group_idx=idx)))
                key = "AR_{}@{}".format('all', max_det)
                print('{}={:.3f}'.format(key, lvisEval._summarize('ar')))
            continue
        lvisEval.run()
        print('-'*8+'{} results'.format(res_type)+'-'*8)
        lvisEval.print_results()


def lvis_fast_eval_recall(results,
                          lvis,
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

    gt_bboxes = []
    img_ids = lvis.get_img_ids()
    for i in range(len(img_ids)):
        ann_ids = lvis.get_ann_ids(img_ids=[img_ids[i]])
        ann_info = lvis.load_anns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=True)
    ar = recalls.mean(axis=1)
    return ar
