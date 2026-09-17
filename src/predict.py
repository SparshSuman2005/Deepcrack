"""
Inference / Prediction script for crack segmentation.

Run the trained U-Net on individual images or an entire directory,
producing side-by-side visualizations: Original | Ground Truth | Prediction.
"""

import os
import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import UNet


def load_model(checkpoint_path, device):
    """Load the trained U-Net model from a checkpoint."""
    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    return model


def preprocess_image(image_path, img_size=256):
    """Load and preprocess a single image for inference.

    Returns:
        input_tensor: [1, 3, H, W] normalized tensor.
        original: Original PIL image (for visualization).
    """
    original = Image.open(image_path).convert('RGB')

    image = TF.resize(original, (img_size, img_size))
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return image.unsqueeze(0), original


@torch.no_grad()
def predict_single(model, image_path, device, img_size=256, threshold=0.5):
    """Run prediction on a single image.

    Returns:
        original: Original PIL image.
        pred_mask: Binary prediction mask as numpy array [H, W].
    """
    input_tensor, original = preprocess_image(image_path, img_size)
    input_tensor = input_tensor.to(device)

    output = model(input_tensor)
    pred = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_mask = (pred > threshold).astype(np.uint8)

    return original, pred_mask


def visualize_prediction(original, pred_mask, gt_mask=None,
                         save_path=None, title=None):
    """Create a side-by-side visualization of the prediction.

    Args:
        original: Original PIL image.
        pred_mask: Binary prediction [H, W].
        gt_mask: Optional ground truth mask [H, W].
        save_path: Optional path to save the figure.
        title: Optional title for the figure.
    """
    ncols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    col = 1
    if gt_mask is not None:
        axes[col].imshow(gt_mask, cmap='gray')
        axes[col].set_title('Ground Truth')
        axes[col].axis('off')
        col += 1

    # Prediction
    axes[col].imshow(pred_mask, cmap='gray')
    axes[col].set_title('Prediction')
    axes[col].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def predict_directory(model, image_dir, output_dir, device,
                      mask_dir=None, img_size=256, threshold=0.5):
    """Run prediction on all images in a directory.

    Args:
        model: Trained U-Net model.
        image_dir: Directory containing input images.
        output_dir: Directory to save prediction visualizations.
        device: torch device.
        mask_dir: Optional directory with ground truth masks.
        img_size: Image size for inference.
        threshold: Binarization threshold.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"Predicting on {len(image_files)} images...")

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        original, pred_mask = predict_single(
            model, img_path, device, img_size, threshold
        )

        # Load ground truth if available
        gt_mask = None
        if mask_dir:
            base = os.path.splitext(img_name)[0]
            for ext in ['.png', '.jpg', '.bmp']:
                gt_path = os.path.join(mask_dir, base + ext)
                if os.path.exists(gt_path):
                    gt_img = Image.open(gt_path).convert('L')
                    gt_img = gt_img.resize(
                        (img_size, img_size), Image.Resampling.NEAREST
                    )
                    gt_mask = np.array(gt_img)
                    gt_mask = (gt_mask > 127).astype(np.uint8)
                    break

        save_path = os.path.join(output_dir, f'pred_{base}.png')
        visualize_prediction(
            original, pred_mask, gt_mask,
            save_path=save_path, title=img_name
        )

    print(f"Predictions saved to: {output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run crack segmentation inference'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to a single image or directory')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Optional path to ground truth masks')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/predictions',
                        help='Output directory for visualizations')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)

    if os.path.isfile(args.input):
        # Single image prediction
        original, pred_mask = predict_single(
            model, args.input, device, args.img_size, args.threshold
        )
        base = os.path.splitext(os.path.basename(args.input))[0]
        save_path = os.path.join(args.output_dir, f'pred_{base}.png')
        visualize_prediction(original, pred_mask, save_path=save_path)
        print(f"Saved prediction to {save_path}")

    elif os.path.isdir(args.input):
        # Directory prediction
        predict_directory(
            model, args.input, args.output_dir, device,
            mask_dir=args.mask_dir,
            img_size=args.img_size,
            threshold=args.threshold
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory.")
        sys.exit(1)


if __name__ == '__main__':
    main()
