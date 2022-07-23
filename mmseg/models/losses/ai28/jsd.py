import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_class_weight, weight_reduce_loss

def jsd(pred, label, weight=None, class_weight=None, reduction='mean',
        avg_factor=None, avg_non_ignore=False, ignore_index=-100, **kwargs):

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    # label, _, _ = torch.chunk(label,3)

    p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                              F.softmax(pred_aug1, dim=1), \
                              F.softmax(pred_aug2, dim=1)
    p_clean, p_aug1, p_aug2 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
                              p_aug1.reshape((1,) + p_aug1.shape).contiguous(), \
                              p_aug2.reshape((1,) + p_aug2.shape).contiguous(),

    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {'p_clean': p_clean,
                      'p_aug1': p_aug1,
                      'p_aug2': p_aug2,
                      'p_mixture': p_mixture}

    return loss, p_distribution


def jsdv1_3_1(pred, label, weight=None, class_weight=None, reduction='mean',
              avg_factor=None, avg_non_ignore=False, ignore_index=-100, **kwargs):

    avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    # label, _, _ = torch.chunk(label,3)

    # if pred_orig.shape[-1] == 1:
    #     p_clean, p_aug1, p_aug2 = torch.cat((torch.sigmoid(pred_orig), 1 - torch.sigmoid(pred_orig)), dim = 1), \
    #                               torch.cat((torch.sigmoid(pred_aug1), 1 - torch.sigmoid(pred_aug1)), dim = 1), \
    #                               torch.cat((torch.sigmoid(pred_aug2), 1 - torch.sigmoid(pred_aug2)), dim = 1),
    p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                              F.softmax(pred_aug1, dim=1), \
                              F.softmax(pred_aug2, dim=1)

    # p_clean, p_aug1, p_aug2 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
    #                           p_aug1.reshape((1,) + p_aug1.shape).contiguous(), \
    #                           p_aug2.reshape((1,) + p_aug2.shape).contiguous(),
    p_clean, p_aug1, p_aug2 = p_clean.contiguous(), \
                              p_aug1.contiguous(), \
                              p_aug2.contiguous(),

    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {'p_clean': p_clean,
                      'p_aug1': p_aug1,
                      'p_aug2': p_aug2,
                      'p_mixture': p_mixture}

    return loss, p_distribution
