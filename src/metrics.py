"""
Evaluation metrics for binary segmentation.

All metrics operate on predictions (after sigmoid + thresholding)
and binary ground truth masks.
"""

import torch


def pixel_accuracy(preds, targets):
    """Compute pixel-level accuracy.

    Args:
        preds: Binary predictions [B, 1, H, W].
        targets: Ground truth [B, 1, H, W].

    Returns:
        Accuracy as a float.
    """
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()


def iou_score(preds, targets, smooth=1e-6):
    """Compute Intersection over Union (Jaccard Index).

    Args:
        preds: Binary predictions [B, 1, H, W].
        targets: Ground truth [B, 1, H, W].
        smooth: Smoothing to avoid division by zero.

    Returns:
        IoU score as a float.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def dice_coefficient(preds, targets, smooth=1e-6):
    """Compute Dice Coefficient (F1 for segmentation).

    Args:
        preds: Binary predictions [B, 1, H, W].
        targets: Ground truth [B, 1, H, W].
        smooth: Smoothing to avoid division by zero.

    Returns:
        Dice coefficient as a float.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )
    return dice.item()


def precision_score(preds, targets, smooth=1e-6):
    """Compute precision (positive predictive value).

    Args:
        preds: Binary predictions [B, 1, H, W].
        targets: Ground truth [B, 1, H, W].
        smooth: Smoothing to avoid division by zero.

    Returns:
        Precision as a float.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    true_positives = (preds * targets).sum()
    predicted_positives = preds.sum()
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return precision.item()


def recall_score(preds, targets, smooth=1e-6):
    """Compute recall (sensitivity / true positive rate).

    Args:
        preds: Binary predictions [B, 1, H, W].
        targets: Ground truth [B, 1, H, W].
        smooth: Smoothing to avoid division by zero.

    Returns:
        Recall as a float.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    true_positives = (preds * targets).sum()
    actual_positives = targets.sum()
    recall = (true_positives + smooth) / (actual_positives + smooth)
    return recall.item()


def f1_score(preds, targets, smooth=1e-6):
    """Compute F1 score (harmonic mean of precision and recall).

    Args:
        preds: Binary predictions [B, 1, H, W].
        targets: Ground truth [B, 1, H, W].
        smooth: Smoothing to avoid division by zero.

    Returns:
        F1 score as a float.
    """
    prec = precision_score(preds, targets, smooth)
    rec = recall_score(preds, targets, smooth)
    return 2 * (prec * rec) / (prec + rec + smooth)


class MetricTracker:
    """Tracks and averages metrics across batches during an epoch.

    Usage:
        tracker = MetricTracker()
        for batch in dataloader:
            tracker.update(preds, targets)
        results = tracker.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._pixel_acc = 0.0
        self._iou = 0.0
        self._dice = 0.0
        self._precision = 0.0
        self._recall = 0.0
        self._count = 0

    @torch.no_grad()
    def update(self, logits, targets, threshold=0.5):
        """Update metrics with a batch of predictions.

        Args:
            logits: Raw model output [B, 1, H, W].
            targets: Ground truth [B, 1, H, W].
            threshold: Threshold for binarizing sigmoid output.
        """
        preds = (torch.sigmoid(logits) > threshold).float()

        self._pixel_acc += pixel_accuracy(preds, targets)
        self._iou += iou_score(preds, targets)
        self._dice += dice_coefficient(preds, targets)
        self._precision += precision_score(preds, targets)
        self._recall += recall_score(preds, targets)
        self._count += 1

    def compute(self):
        """Compute average metrics over all batches.

        Returns:
            Dict of metric_name → average_value.
        """
        n = max(self._count, 1)
        return {
            'pixel_accuracy': self._pixel_acc / n,
            'iou': self._iou / n,
            'dice': self._dice / n,
            'precision': self._precision / n,
            'recall': self._recall / n,
        }
