"""
Loss functions for crack segmentation.

Provides DiceLoss and a combined BCEDiceLoss that handles the extreme
class imbalance typical in crack detection (cracks occupy <5% of pixels).
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Sørensen-Dice loss for binary segmentation.

    Measures the overlap between predicted and ground truth masks.
    Handles class imbalance well since it focuses on the positive class.

    Args:
        smooth: Smoothing factor to avoid division by zero.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model output [B, 1, H, W] (before sigmoid).
            targets: Ground truth binary mask [B, 1, H, W].

        Returns:
            Dice loss (1 - Dice coefficient).
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy and Dice Loss.

    BCE provides stable gradients for all pixels, while Dice focuses
    on the positive (crack) class, making this combination effective
    for imbalanced segmentation tasks.

    Args:
        bce_weight: Weight for the BCE component (default: 0.5).
        dice_weight: Weight for the Dice component (default: 0.5).
        smooth: Smoothing factor for Dice loss.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model output [B, 1, H, W] (before sigmoid).
            targets: Ground truth binary mask [B, 1, H, W].

        Returns:
            Weighted sum of BCE and Dice losses.
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme class imbalance.

    Down-weights easy examples and focuses on hard misclassified pixels,
    useful when crack pixels are very rare.

    Args:
        alpha: Weighting factor for positive class.
        gamma: Focusing parameter (higher = more focus on hard examples).
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()
