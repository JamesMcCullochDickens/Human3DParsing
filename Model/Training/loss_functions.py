from typing import *

import torch.nn as nn
import torch
import torch.nn.functional as F


def get_ce_loss(label_smoothing: float = 0.0, ignore_index: int = -100,
                ce_weights=None) -> nn.Module:
    if ce_weights is None:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                   ignore_index=ignore_index,
                                   weight=ce_weights)
    else:
        return weighted_ce_loss(label_smoothing=label_smoothing,
                                   ignore_index=ignore_index,
                                   ce_weights=ce_weights)


class weighted_ce_loss(nn.Module):
    def __init__(self, ce_weights: torch.Tensor, label_smoothing: float = 0.0,
                 ignore_index: int = -100):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                           ignore_index=ignore_index,
                                           weight=ce_weights)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self.ce_loss.weight is not None:
            self.ce_loss.weight = self.ce_loss.weight.to(pred.get_device())
        return self.ce_loss(pred, gt)



class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: Tensor of shape [B*N, C] (logits or probabilities)
        target: Tensor of shape [B*N] (long dtype with class indices)
        """
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)  # convert logits to probabilities

        # One-hot encode target: shape [B*N, C]
        target_one_hot = F.one_hot(target, num_classes=num_classes).float()

        # Compute intersection and union
        intersection = torch.sum(pred * target_one_hot, dim=0)  # [C]
        union = torch.sum(pred + target_one_hot, dim=0)  # [C]

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [C]
        loss = 1.0 - dice.mean()

        return loss


class CEDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, label_smoothing: float = 0.0, ignore_index: int = -100,
                ce_weights=None, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.ce_loss = get_ce_loss(label_smoothing, ignore_index, ce_weights)
        self.dice_loss = DiceLoss(smooth)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight


    def forward(self, pred, target):
        l1 = self.ce_loss(pred, target)
        l2 = self.dice_loss(pred, target)
        overall_loss = self.ce_weight*l1 + self.ce_weight*l2
        return overall_loss