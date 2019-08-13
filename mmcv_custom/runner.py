import os
import os.path as osp
import mmcv
from mmcv.runner.utils import obj_from_dict
import torch
from .parameters import parameters
from mmcv.runner.checkpoint import save_checkpoint


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
