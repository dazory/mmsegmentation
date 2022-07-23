# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from ..utils import get_class_weight, weight_reduce_loss

from .jsd import *
from ..cross_entropy_loss import cross_entropy, binary_cross_entropy, mask_cross_entropy

@LOSSES.register_module()
class CrossEntropyLossPlus(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce',
                 avg_non_ignore=False,
                 additional_loss='jsd',
                 additional_loss_weight_reduce=False,
                 lambda_weight=0.0001,
                 temper=1,
                 add_act=None,
                 wandb_name=None
                 ):
        super(CrossEntropyLossPlus, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self._loss_name = loss_name

        self.additional_loss = additional_loss
        self.additional_loss_weight_reduce = additional_loss_weight_reduce
        self.lambda_weight = lambda_weight
        self.temper = temper
        self.add_act = add_act
        self.wandb_name = wandb_name

        if self.additional_loss == 'jsd':
            self.cls_additional = jsd
        elif self.additional_loss == 'jsdv1_3_1':
            self.cls_additional = jsdv1_3_1
        else:
            self.cls_additional = None

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        # Note: for BCE loss, label < 0 is invalid.

        loss_cls = self.loss_weight * self.cls_criterion(
            torch.chunk(cls_score, 3)[0],
            torch.chunk(label, 3)[0],
            torch.chunk(weight, 3)[0] if weight is not None else weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=ignore_index,
            **kwargs)

        loss_additional = 0
        if self.cls_additional is not None:
            if self.additional_loss_weight_reduce == False:
                weight = None
            loss_additional, p_distribution = self.cls_additional(
                cls_score,
                label,
                weight,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                avg_non_ignore=self.avg_non_ignore,
                ignore_index=ignore_index,
                temper=self.temper,
                add_act=self.add_act,
                lambda_weight=self.lambda_weight
            )

        loss = loss_cls + self.lambda_weight * loss_additional

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
