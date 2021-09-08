# ------------------------------------------------------------------------------
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# ------------------------------------------------------------------------------
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F


class LeastSquaresGenerativeAdversarialLoss(nn.Module):
    """
    Loss for `Least Squares Generative Adversarial Network (LSGAN) <https://arxiv.org/abs/1611.04076>`_

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - prediction (tensor): unnormalized discriminator predictions
        - real (bool): if the ground truth label is for real images or fake images. Default: true

    .. warning::
        Do not use sigmoid as the last layer of Discriminator.

    """
    def __init__(self, reduction='mean'):
        super(LeastSquaresGenerativeAdversarialLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, prediction, real=True):
        if real:
            label = torch.ones_like(prediction)
        else:
            label = torch.zeros_like(prediction)
        return self.mse_loss(prediction, label)


class VanillaGenerativeAdversarialLoss(nn.Module):
    """
    Loss for `Vanilla Generative Adversarial Network <https://arxiv.org/abs/1406.2661>`_

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - prediction (tensor): unnormalized discriminator predictions
        - real (bool): if the ground truth label is for real images or fake images. Default: true

    .. warning::
        Do not use sigmoid as the last layer of Discriminator.

    """
    def __init__(self, reduction='mean'):
        super(VanillaGenerativeAdversarialLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, prediction, real=True):
        if real:
            label = torch.ones_like(prediction)
        else:
            label = torch.zeros_like(prediction)
        return self.bce_loss(prediction, label)


class WassersteinGenerativeAdversarialLoss(nn.Module):
    """
    Loss for `Wasserstein Generative Adversarial Network <https://arxiv.org/abs/1701.07875>`_

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - prediction (tensor): unnormalized discriminator predictions
        - real (bool): if the ground truth label is for real images or fake images. Default: true

    .. warning::
        Do not use sigmoid as the last layer of Discriminator.

    """
    def __init__(self, reduction='mean'):
        super(WassersteinGenerativeAdversarialLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, prediction, real=True):
        if real:
            return -prediction.mean()
        else:
            return prediction.mean()

class ContourLoss(nn.Module):
    pass


class SemanticConsistency(nn.Module):
    """
    Semantic consistency loss is introduced by
    `CyCADA: Cycle-Consistent Adversarial Domain Adaptation (ICML 2018) <https://arxiv.org/abs/1711.03213>`_

    This helps to prevent label flipping during image translation.

    Args:
        ignore_index (tuple, optional): Specifies target values that are ignored
            and do not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: ().
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> loss = SemanticConsistency()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, ignore_index=(), reduction='mean'):
        super(SemanticConsistency, self).__init__()
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        for class_idx in self.ignore_index:
            target[target == class_idx] = 255
        return self.loss(input, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)):
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()