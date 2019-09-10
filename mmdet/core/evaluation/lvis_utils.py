import mmcv
import numpy as np
from lvis import LVIS, LVISEval
from .recall import eval_recalls


def lvis_eval(result_files, result_types, lvis, max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in [
            'proposal_fast', 'bbox', 'segm'
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

        iou_type = res_type
        lvisEval = LVISEval(lvis, result_file, iou_type)
        lvisEval.params.img_ids = img_ids
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
            results = proposal2numpy(results)
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


def proposal2numpy(results):
    # TODO: support multiple imgs per GPU
    proposals = []
    for idx in range(len(results)):
        proposals.append(results[idx][-1][0].cpu().numpy())
    return proposals

