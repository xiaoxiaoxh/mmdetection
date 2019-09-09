import os
import sys
path = os.getcwd()
sys.path.insert(0, path)
from argparse import ArgumentParser

from mmdet.core import lvis_eval


def main():
    parser = ArgumentParser(description='LVIS Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('--ann', help='annotation file path')
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
    lvis_eval(args.result, args.types, args.ann, args.max_dets)


if __name__ == '__main__':
    main()
