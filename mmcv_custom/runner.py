import os
import os.path as osp
import mmcv
from mmcv.runner.utils import obj_from_dict
import torch
import time
from .parameters import parameters
from .checkpoint import save_checkpoint, load_checkpoint


class Runner(mmcv.runner.Runner):
    """A training helper for PyTorch.

        Custom version of mmcv runner, overwrite init_optimizer method
    """

    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(
                optimizer, torch.optim,
                dict(params=parameters(self.model, optimizer.lr)))
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
        self.model.module.bbox_head.max_iters = self.max_iters
        self.model.module.bbox_head.max_epochs = self.max_epochs
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.model.module.bbox_head.iter = self.iter
            self.model.module.bbox_head.epoch = self.epoch
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

        self.call_hook('after_train_epoch')
        self._epoch += 1

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
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        # linkpath = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # use relative symlink
        # mmcv.symlink(filename, linkpath)

    def auto_resume(self):
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
            self.resume(filename)
        # linkname = osp.join(self.work_dir, 'latest.pth')
        # if osp.exists(linkname):
        #     self.logger.info('latest checkpoint found')
        #     self.resume(linkname)
