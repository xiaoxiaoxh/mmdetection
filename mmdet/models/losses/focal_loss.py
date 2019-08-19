import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    # pred_sigmoid = pred.sigmoid()
    # target = target.type_as(pred)
    # pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # focal_weight_v1 = (alpha * target + (1 - alpha) *
    #                 (1 - target)) * pt.pow(gamma)

    target = target.to(pred.get_device())
    one_hot_target = torch.zeros(pred.size()[0], pred.size()[1], device=pred.get_device())
    one_hot_target = one_hot_target.scatter_(1, target.unsqueeze(1), 1).float()
    focal_weight = (alpha * one_hot_target + (1 - alpha) * (1 - one_hot_target)) * \
                   torch.exp(-gamma * one_hot_target * pred -
                   gamma * torch.log1p(torch.exp(-1.0 * pred)))
    loss = F.binary_cross_entropy_with_logits(
        pred, one_hot_target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None,
                       use_py_version=False):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    if use_py_version:
        loss = py_sigmoid_focal_loss(pred, target, weight=weight, gamma=gamma, alpha=alpha,
                                     reduction=reduction, avg_factor=avg_factor)
    else:
        loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
        # TODO: find a proper way to handle the shape of weight
        if weight is not None:
            weight = weight.view(-1, 1)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 use_py_version=False):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_py_version = use_py_version

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
                use_py_version=self.use_py_version)
        else:
            raise NotImplementedError
        return loss_cls
