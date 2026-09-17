"""
Evaluate the trained U-Net model on the test set.

Loads the best checkpoint and computes per-image and aggregate
segmentation metrics on the test dataset.
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import UNet
from src.dataset import CrackDataset
from src.transforms import get_val_transform
from src.metrics import (
    pixel_accuracy, iou_score, dice_coefficient,
    precision_score, recall_score
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate U-Net on the test set'
    )
    parser.add_argument('--image_dir', type=str,
                        default='dataset/test_img',
                        help='Path to test images')
    parser.add_argument('--mask_dir', type=str,
                        default='dataset/test_lab',
                        help='Path to test masks')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device, threshold=0.5):
    """Run evaluation on the full dataset.

    Returns:
        Dict of aggregate metrics.
    """
    model.eval()

    all_metrics = {
        'pixel_accuracy': [],
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': [],
    }

    pbar = tqdm(dataloader, desc='Evaluating')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        preds = (torch.sigmoid(outputs) > threshold).float()

        # Per-batch metrics
        all_metrics['pixel_accuracy'].append(pixel_accuracy(preds, masks))
        all_metrics['iou'].append(iou_score(preds, masks))
        all_metrics['dice'].append(dice_coefficient(preds, masks))
        all_metrics['precision'].append(precision_score(preds, masks))
        all_metrics['recall'].append(recall_score(preds, masks))

    # Average across all batches
    avg_metrics = {
        k: sum(v) / len(v) for k, v in all_metrics.items()
    }
    return avg_metrics


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load model ---
    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Best training IoU: {checkpoint.get('best_iou', 'N/A')}")

    # --- Dataset ---
    img_size = (args.img_size, args.img_size)
    test_dataset = CrackDataset(
        args.image_dir, args.mask_dir,
        transform=get_val_transform(img_size)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    print(f"Test samples: {len(test_dataset)}")

    # --- Evaluate ---
    metrics = evaluate(model, test_loader, device, args.threshold)

    # --- Print results ---
    print(f"\n{'='*50}")
    print(f"  Test Set Evaluation Results")
    print(f"{'='*50}")
    print(f"  Pixel Accuracy : {metrics['pixel_accuracy']:.4f}")
    print(f"  IoU (Jaccard)  : {metrics['iou']:.4f}")
    print(f"  Dice (F1)      : {metrics['dice']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
