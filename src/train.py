"""
Training script for the U-Net crack segmentation model.

Features:
  - 80/20 train/val split
  - AdamW optimizer with OneCycleLR scheduler
  - BCEDice combined loss for class-imbalanced segmentation
  - Automatic best-model checkpointing based on validation IoU
  - CSV training log for analysis
  - Progress bars via tqdm
"""

import os
import sys
import csv
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import UNet, count_parameters
from src.dataset import CrackDataset
from src.transforms import get_train_transform, get_val_transform
from src.losses import BCEDiceLoss
from src.metrics import MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train U-Net for crack segmentation'
    )

    # Data
    parser.add_argument('--image_dir', type=str,
                        default='dataset/train_img_resized',
                        help='Path to training images')
    parser.add_argument('--mask_dir', type=str,
                        default='dataset/resized_lab_mask',
                        help='Path to training masks')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size (square)')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data for validation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (0 for Windows)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory for training logs')

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc='  Train', leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate and return average loss and metrics."""
    model.eval()
    total_loss = 0.0
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc='  Val  ', leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        tracker.update(outputs, masks)

    avg_loss = total_loss / len(dataloader)
    metrics = tracker.compute()
    return avg_loss, metrics


def main():
    args = parse_args()

    # --- Setup ---
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Dataset ---
    img_size = (args.img_size, args.img_size)
    full_dataset = CrackDataset(
        args.image_dir, args.mask_dir,
        transform=None  # Will be set per-split below
    )

    # Split into train/val
    total = len(full_dataset)
    val_size = int(total * args.val_split)
    train_size = total - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Create separate datasets with proper transforms
    train_dataset = CrackDataset(
        args.image_dir, args.mask_dir,
        transform=get_train_transform(img_size)
    )
    val_dataset = CrackDataset(
        args.image_dir, args.mask_dir,
        transform=get_val_transform(img_size)
    )

    # Use subset indices
    train_dataset = torch.utils.data.Subset(train_dataset, train_subset.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --- Model ---
    model = UNet(in_channels=3, out_channels=1).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # --- Loss, Optimizer, Scheduler ---
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
    )

    # --- Training Log ---
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'val_loss', 'val_iou',
            'val_dice', 'val_precision', 'val_recall',
            'val_pixel_acc', 'lr', 'time_sec'
        ])

    # --- Training Loop ---
    best_iou = 0.0
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print(f"Epoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Log results
        print(f"  Train Loss: {train_loss:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  |  "
              f"Val IoU: {val_metrics['iou']:.4f}  |  "
              f"Val Dice: {val_metrics['dice']:.4f}  |  "
              f"LR: {current_lr:.6f}  |  "
              f"Time: {epoch_time:.1f}s")

        # Save to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_metrics['iou']:.6f}",
                f"{val_metrics['dice']:.6f}",
                f"{val_metrics['precision']:.6f}",
                f"{val_metrics['recall']:.6f}",
                f"{val_metrics['pixel_accuracy']:.6f}",
                f"{current_lr:.8f}",
                f"{epoch_time:.2f}"
            ])

        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 'best_model.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"  ✓ Best model saved (IoU: {best_iou:.4f})")

        print()

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
    }, final_path)

    print(f"{'='*60}")
    print(f"Training complete!")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")
    print(f"Training log saved to: {log_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
