import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


@LOSSES.register_module
class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self,
                 bins=10,
                 momentum=0,
                 reduction='mean',
                 use_sigmoid=True,
                 loss_weight=1.0,
                 **kwargs):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.reduction = reduction
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight,
                reduction_override=None,
                **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, weight = _expand_binary_labels(
                target, weight, pred.size(-1))
        target, weight = target.float(), weight.float()
        edges = self.edges
        mmt = self.momentum
        ghm_weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    ghm_weights[inds] = tot / self.acc_sum[i]
                else:
                    ghm_weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            ghm_weights = ghm_weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, ghm_weights, reduction='none')
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor=tot)
        return loss * self.loss_weight


# TODO: code refactoring to make it consistent with other losses
@LOSSES.register_module
class GHMR(nn.Module):
    """GHM Regression Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector"
    https://arxiv.org/abs/1811.05181

    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        loss_weight (float): The weight of the total GHM-R loss.
    """

    def __init__(self,
                 mu=0.02,
                 bins=10,
                 momentum=0,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] = 1e3
        self.momentum = momentum
        self.reduction = reduction
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.loss_weight = loss_weight

    # TODO: support reduction parameter
    def forward(self,
                pred,
                target,
                weight,
                reduction_override=None,
                avg_factor=None,
                **kwargs):
        """Calculate the GHM-R loss.

        Args:
            pred (float tensor of size [batch_num, 4 (* class_num)]):
                The prediction of box regression layer. Channel number can be 4
                or 4 * class_num depending on whether it is class-agnostic.
            target (float tensor of size [batch_num, 4 (* class_num)]):
                The target regression values with the same size of pred.
            weight (float tensor of size [batch_num, 4 (* class_num)]):
                The weight of each sample, 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        ghm_weights = torch.zeros_like(g)

        valid = weight > 0
        tot = max(weight.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    ghm_weights[inds] = tot / self.acc_sum[i]
                else:
                    ghm_weights[inds] = tot / num_in_bin
        if n > 0:
            ghm_weights /= n

        loss = loss * ghm_weights
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss * self.loss_weight
