import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp

from ..registry import LOSSES
from .cross_entropy_loss import cross_entropy
from mmdet.core import force_fp32


def ldam_loss(pred,
              target,
              weights,
              samples_per_cls=None,
              complexity=0.3,
              avg_factor=None,
              gamma=10.0,
              reduction=True,
              delta=None,
              **kwargs):
    """Label-Distrubution aware margin loss"""
    device = pred.get_device()
    target = target.to(device)

    if (delta is not None) or (samples_per_cls is None):
        delta_batch = torch.ones(target.size()).to(device) * delta  # (batch_size) Tensor
    else:
        delta_all = complexity / torch.pow(samples_per_cls, 1 / 4)  # (num_classes) Tensor
        delta_batch = torch.index_select(delta_all, 0, target)  # (batch_size) Tensor
    sparse_delta = torch.zeros(pred.size(), device=device) \
        .scatter_(1, target.unsqueeze(1), delta_batch.unsqueeze(1))  # (batch_size, num_classes)
    margin_pred = pred - sparse_delta * gamma

    loss = cross_entropy(
        margin_pred, target, weights, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module
class LDAMLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0,
                 samples_per_cls_file=None,
                 complexity=0.3,
                 delta=None,
                 **kwargs):
        super(LDAMLoss, self).__init__()
        assert (use_sigmoid is False) and (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.complexity = complexity
        self.delta = delta
        self.fp16_enabled = False

        if samples_per_cls_file and osp.exists(samples_per_cls_file):
            with open(samples_per_cls_file, 'r') as f:
                self.samples_per_cls = [int(line.strip()) for line in f.readlines()]
        else:
            self.samples_per_cls = None

        self.cls_criterion = ldam_loss

    @force_fp32(apply_to=('gamma', ))
    def forward(self,
                cls_score,
                label,
                weight=None,
                gamma=10.0,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            samples_per_cls=self.samples_per_cls,
            complexity=self.complexity,
            gamma=gamma,
            delta=self.delta,
            **kwargs)
        return loss_cls
