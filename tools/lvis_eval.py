from argparse import ArgumentParser
import os
import sys

import mmcv
from mmdet.apis import init_dist
from mmdet.core import lvis_eval, results2json
from mmdet.datasets import build_dataset


def main():
    parser = ArgumentParser(description='LVIS Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('--cfg', help='config file path')
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        choices=['proposal_fast', 'proposal', 'bbox', 'segm', 'keypoint'],
        default=['bbox'],
        help='result types')
    parser.add_argument(
        '--max-dets',
        type=int,
        nargs='+',
        default=[100, 300, 1000],
        help='proposal numbers, only used for recall evaluation')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.cfg)
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result)
    eval_types = args.types
    if eval_types:
        print('Starting evaluate {}'.format(' and '.join(eval_types)))
        if eval_types == ['proposal_fast']:
            result_file = args.result
            lvis_eval(result_file, eval_types, dataset.lvis, args.max_dets)
        else:
            if not isinstance(results[0], dict):
                result_files = results2json(dataset, results, args.result)
                lvis_eval(result_files, eval_types, dataset.lvis, args.max_dets)
            else:
                for name in results[0]:
                    print('\nEvaluating {}'.format(name))
                    outputs_ = [out[name] for out in results]
                    result_file = args.result + '.{}'.format(name)
                    result_files = results2json(dataset, outputs_, result_file)
                    lvis_eval(result_files, eval_types, dataset.lvis, args.max_dets)


if __name__ == '__main__':
    path = os.getcwd()
    sys.path.insert(0, path)
    main()
