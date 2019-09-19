from __future__ import division
from collections import OrderedDict

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, obj_from_dict
from mmcv_custom import Runner

from mmdet import datasets
from mmdet.core import (CocoDistEvalmAPHook, CocoDistEvalRecallHook, LvisDistEvalmAPHook,
                        DistEvalmAPHook, DistOptimizerHook, Fp16OptimizerHook)
from mmdet.datasets import DATASETS, build_dataloader
from mmdet.models import RPN
from .env import get_root_logger


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    try:
        losses = model(**data)
        loss, log_vars = parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
    except RuntimeError as e:
        raise e

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)


def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=True)
        for ds in dataset
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level,
                    resume_from=cfg.resume_from, load_from=cfg.load_from, auto_resume=cfg.auto_resume)

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.test  # change to test dataset
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(
                CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg))
        else:
            dataset_type = DATASETS.get(val_dataset_cfg.type)
            if issubclass(dataset_type, datasets.CocoDataset):
                runner.register_hook(
                    CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg))
            elif issubclass(dataset_type, datasets.LvisDataSet):
                runner.register_hook(
                    LvisDistEvalmAPHook(val_dataset_cfg, **eval_cfg))
            else:
                runner.register_hook(
                    DistEvalmAPHook(val_dataset_cfg, **eval_cfg))

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for ds in dataset
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level,
                    resume_from=cfg.resume_from, load_from=cfg.load_from, auto_resume=cfg.auto_resume)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
