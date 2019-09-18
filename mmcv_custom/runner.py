import os
import os.path as osp
import mmcv
from mmcv.runner.utils import obj_from_dict, get_host_info
import mmcv_custom.lr_updater as lr_updater
from mmcv_custom.lr_updater import LrUpdaterHook
import torch
import re
import logging
import time
from .checkpoint import save_checkpoint, load_checkpoint


class Runner(mmcv.runner.Runner):
    """A training helper for PyTorch.

        Custom version of mmcv runner, overwrite init_optimizer and train and run method
    """
    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 resume_from=None,
                 load_from=None,
                 auto_resume=False,
                 **kwargs
                 ):
        super(Runner, self).__init__(model,
                                     batch_processor,
                                     optimizer=optimizer,
                                     work_dir=work_dir,
                                     log_level=log_level,
                                     logger=logger)
        self.resume_from = resume_from
        self.load_from = load_from
        self.auto_resume_bool = auto_resume
        if isinstance(optimizer, dict):
            self.optimizer_cfg = optimizer
        else:
            self.optimizer_cfg = None
        self._stage_epoch = 0
        self._stage_max_epochs = 0
        self._stage_iter = 0
        self._stage = 0

    @property
    def stage_epoch(self):
        """int: Current epoch(relative) in one stage."""
        return self._stage_epoch

    @property
    def stage_iter(self):
        """int: Current iteration(relative) in one stage."""
        return self._stage_iter

    @property
    def current_stage(self):
        """int: Current stage num."""
        return self._stage

    @property
    def stage_max_epochs(self):
        """int: Current stage max epochs."""
        return self._stage_max_epochs

    @staticmethod
    def build_optimizer(model, optimizer_cfg, filter_no_grad=False):
        """Build optimizer from configs.

        Args:
            model (:obj:`nn.Module`): The model with parameters to be optimized.
            optimizer_cfg (dict): The config dict of the optimizer.
                Positional fields are:
                    - type: class name of the optimizer.
                    - lr: base learning rate.
                Optional fields are:
                    - any arguments of the corresponding optimizer type, e.g.,
                      weight_decay, momentum, etc.
                    - paramwise_options: a dict with 3 accepted fileds
                      (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                      `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                      the lr and weight decay respectively for all bias parameters
                      (except for the normalization layers), and
                      `norm_decay_mult` will be multiplied to the weight decay
                      for all weight and bias parameters of normalization layers.
            filter_no_grad (bool): Whether to filter params whose require_grad=False
        Returns:
            torch.optim.Optimizer: The initialized optimizer.
        """
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = optimizer_cfg.copy()
        paramwise_options = optimizer_cfg.pop('paramwise_options', None)
        # if no paramwise option is specified, just use the global setting
        if paramwise_options is None:
            return obj_from_dict(optimizer_cfg, torch.optim,
                                 dict(params=filter(lambda p: p.requires_grad, model.parameters())
                                 if filter_no_grad else model.parameters()))
        else:
            assert isinstance(paramwise_options, dict)
            # get base lr and weight decay
            base_lr = optimizer_cfg['lr']
            base_wd = optimizer_cfg.get('weight_decay', None)
            # weight_decay must be explicitly specified if mult is specified
            if ('bias_decay_mult' in paramwise_options
                    or 'norm_decay_mult' in paramwise_options):
                assert base_wd is not None
            # get param-wise options
            bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
            bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
            norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
            # set param-wise lr and weight decay
            params = []
            for name, param in model.named_parameters():
                param_group = {'params': [param]}
                if not param.requires_grad and not filter_no_grad:  # add filter_no_grad option
                    # FP16 training needs to copy gradient/weight between master
                    # weight copy and model weight, it is convenient to keep all
                    # parameters here to align with model.parameters()
                    params.append(param_group)
                    continue

                # for norm layers, overwrite the weight decay of weight and bias
                # TODO: obtain the norm layer prefixes dynamically
                if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                    if base_wd is not None:
                        param_group['weight_decay'] = base_wd * norm_decay_mult
                # for other layers, overwrite both lr and weight decay of bias
                elif name.endswith('.bias'):
                    param_group['lr'] = base_lr * bias_lr_mult
                    if base_wd is not None:
                        param_group['weight_decay'] = base_wd * bias_decay_mult
                # otherwise use the global settings

                params.append(param_group)

            optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
            return optimizer_cls(params, **optimizer_cfg)

    def init_optimizer(self, optimizer, filter_no_grad=False):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.
            filter_no_grad (bool): wheter to filter params whose requires_grad=False
        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = self.build_optimizer(self.model, optimizer, filter_no_grad=filter_no_grad)
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.max_iters = self.max_iters
        model.max_epochs = self.max_epochs
        model.current_stage = self.current_stage
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            model.iter = self.iter
            model.epoch = self.epoch
            model.stage_iter = self.stage_iter
            model.stage_epoch = self.stage_epoch
            self.call_hook('before_train_iter')
            try:
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=True, **kwargs)
                if not isinstance(outputs, dict):
                    raise TypeError('batch_processor() must return a dict')
                outputs['log_vars']['lr'] = self.current_lr()[0]  # add lr in log variables
                outputs['log_vars']['epoch'] = self.epoch  # add epoch in log variables
                if 'log_vars' in outputs:
                    self.log_buffer.update(outputs['log_vars'],
                                           outputs['num_samples'])
                self.outputs = outputs
                self.call_hook('after_train_iter')
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('WARNING: CUDA ran out of memory!')
                    os.system('ps -ef | grep python | grep -v grep | awk \'{print "kill -9 "$2}\' | sh')
                else:
                    raise e
            self._iter += 1
            self._stage_iter += 1  # add _stage_iter

        self.call_hook('after_train_epoch')
        self._epoch += 1
        self._stage_epoch += 1  # add _stage_epoch

    @staticmethod
    def train_all_stage(runner, data_loader, stage_epoch=0, resume_optimizer=False, **kwargs):
        runner.train(data_loader, **kwargs)

    @staticmethod
    def train_rpn_stage(runner, data_loader, stage_epoch=0, resume_optimizer=False, **kwargs):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model.use_bbox = False
        model.use_mask = False

        runner.train(data_loader, **kwargs)

    @staticmethod
    def train_head_stage(runner, data_loader, stage_epoch=0, resume_optimizer=False, **kwargs):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model.use_bbox = True
        model.use_mask = True
        if stage_epoch == 0 or resume_optimizer:
            for name, module in model.named_children():
                if name in ['backbone', 'neck', 'rpn_head']:
                    for param in module.parameters():
                        param.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = True
            if stage_epoch == 0:
                if model.with_bbox:
                    model.bbox_roi_extractor.init_weights()
                    model.bbox_head.init_weights()
                if model.with_mask:
                    model.mask_head.init_weights()
                    if not model.share_roi_extractor:
                        model.mask_roi_extractor.init_weights()

        runner.train(data_loader, **kwargs)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        try:
            if self.resume_from:
                self.resume(self.resume_from)
            elif self.load_from:
                self.load_checkpoint(self.load_from)
            elif self.auto_resume_bool:
                self.auto_resume()
            resume_optimizer = True
        except ValueError as e:  # can not load optimizer state_dict beacuse of param_group dismatching
            self.logger.warn(str(e))
            raise e

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                if i < self.current_stage:
                    continue
                mode, epochs = flow
                if isinstance(mode, str):  # self.train() or custom functions
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))

                self._stage_max_epochs = epochs
                for epoch in range(epochs):
                    if 'train' in mode and (self.epoch >= max_epochs or self.stage_epoch >= epochs):
                        break
                    if 'stage' in mode:
                        # TODO: fix multiple data_loaders
                        epoch_runner(self, data_loader=data_loaders[0],
                                     stage_epoch=epoch,
                                     resume_optimizer=resume_optimizer, **kwargs)
                    else:
                        epoch_runner(data_loaders[i], **kwargs)
                resume_optimizer = False
                self._stage += 1
                self._stage_epoch = 0
                self._stage_iter = 0

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter,
                        stage_epoch=self.stage_epoch + 1,
                        stage_iter=self.stage_iter,
                        stage=self.current_stage,
                        )
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter,
                        stage_epoch=self.stage_epoch + 1,
                        stage_iter=self.stage_iter,
                        stage=self.current_stage)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        # linkpath = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # use relative symlink
        # mmcv.symlink(filename, linkpath)

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'stage_epoch' in checkpoint['meta']:
            self._stage_epoch = checkpoint['meta']['stage_epoch']
        else:
            self._stage_epoch = self.epoch
        if 'stage_iter' in checkpoint['meta']:
            self._stage_iter = checkpoint['meta']['stage_iter']
        else:
            self._stage_iter = self.iter
        if 'stage' in checkpoint['meta']:
            self._stage = checkpoint['meta']['stage']
        else:
            self._stage = 0
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def auto_resume(self, resume_optimizer=True):
        latest_epoch = -1
        latest_name = ''
        for root, dirs, files in os.walk(self.work_dir, topdown=True):
            for name in files:
                if 'epoch_' in name:
                    epoch_num = int(name[name.find('_')+1:name.find('.pth')])
                    latest_name = name if epoch_num > latest_epoch else latest_name
                    latest_epoch = epoch_num if epoch_num > latest_epoch else latest_epoch

        filename = osp.join(self.work_dir, latest_name)
        if latest_name != '' and latest_epoch >= 0 and osp.exists(filename):
            self.logger.info('latest checkpoint {} found'.format(latest_name))
            self.resume(filename, resume_optimizer=resume_optimizer)
        # linkname = osp.join(self.work_dir, 'latest.pth')
        # if osp.exists(linkname):
        #     self.logger.info('latest checkpoint found')
        #     self.resume(linkname)
