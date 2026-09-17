"""
Visualization utilities for crack segmentation.

Plot training curves, sample predictions, and overlay masks
for qualitative analysis of model performance.
"""

import os
import csv
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import UNet
from src.dataset import CrackDataset
from src.transforms import get_val_transform


def plot_training_curves(log_path, output_dir='outputs/figures'):
    """Plot training loss, validation loss, IoU, and Dice curves.

    Args:
        log_path: Path to the training_log.csv file.
        output_dir: Directory to save the figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    epochs, train_losses, val_losses = [], [], []
    ious, dices, precisions, recalls, lrs = [], [], [], [], []

    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            ious.append(float(row['val_iou']))
            dices.append(float(row['val_dice']))
            precisions.append(float(row['val_precision']))
            recalls.append(float(row['val_recall']))
            lrs.append(float(row['lr']))

    # --- Figure 1: Loss curves ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    # Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # IoU
    axes[0, 1].plot(epochs, ious, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].grid(True, alpha=0.3)

    # Dice
    axes[1, 0].plot(epochs, dices, 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].set_title('Validation Dice Coefficient')
    axes[1, 0].grid(True, alpha=0.3)

    # Precision & Recall
    axes[1, 1].plot(epochs, precisions, 'c-', label='Precision', linewidth=2)
    axes[1, 1].plot(epochs, recalls, 'y-', label='Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")

    # --- Figure 2: Learning rate schedule ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lrs, 'orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    lr_path = os.path.join(output_dir, 'lr_schedule.png')
    plt.savefig(lr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"LR schedule saved to: {lr_path}")


def plot_sample_predictions(model, dataset, device, num_samples=8,
                            output_dir='outputs/figures'):
    """Plot a grid of sample predictions from the dataset.

    Args:
        model: Trained U-Net model.
        dataset: CrackDataset with val/test transforms.
        device: torch device.
        num_samples: Number of samples to plot.
        output_dir: Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')

    # Denormalization constants
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]

        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8)

        # Denormalize image for display
        img_display = image.cpu() * std + mean
        img_display = img_display.clamp(0, 1).permute(1, 2, 0).numpy()

        mask_display = mask.squeeze().cpu().numpy()

        axes[row, 0].imshow(img_display)
        axes[row, 0].set_title('Input')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask_display, cmap='gray')
        axes[row, 1].set_title('Ground Truth')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pred_binary, cmap='gray')
        axes[row, 2].set_title('Prediction')
        axes[row, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sample_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to: {save_path}")


def plot_overlay(image, mask, pred, output_path=None):
    """Overlay predicted mask on the original image.

    Green = True Positive, Red = False Positive, Blue = False Negative.

    Args:
        image: Original image as numpy array [H, W, 3].
        mask: Ground truth binary mask [H, W].
        pred: Predicted binary mask [H, W].
        output_path: Optional path to save the figure.
    """
    overlay = image.copy().astype(np.float32)

    tp = (pred == 1) & (mask == 1)  # True Positive (green)
    fp = (pred == 1) & (mask == 0)  # False Positive (red)
    fn = (pred == 0) & (mask == 1)  # False Negative (blue)

    alpha = 0.5
    overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    overlay[fp] = overlay[fp] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    overlay[fn] = overlay[fn] * (1 - alpha) + np.array([0, 0, 255]) * alpha

    overlay = overlay.astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (G=TP, R=FP, B=FN)')
    axes[3].axis('off')

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualization utilities for crack segmentation'
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['curves', 'samples', 'both'],
                        help='What to visualize')
    parser.add_argument('--log_path', type=str,
                        default='outputs/training_log.csv',
                        help='Path to training log CSV')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str,
                        default='dataset/test_img',
                        help='Image directory for sample predictions')
    parser.add_argument('--mask_dir', type=str,
                        default='dataset/test_lab',
                        help='Mask directory for sample predictions')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/figures',
                        help='Output directory for figures')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of sample predictions')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in ('curves', 'both'):
        plot_training_curves(args.log_path, args.output_dir)

    if args.mode in ('samples', 'both'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = UNet(in_channels=3, out_channels=1).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device,
                                weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        dataset = CrackDataset(
            args.image_dir, args.mask_dir,
            transform=get_val_transform((256, 256))
        )

        plot_sample_predictions(
            model, dataset, device,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()
