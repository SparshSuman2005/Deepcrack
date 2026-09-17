"""
CrackDataset — PyTorch Dataset for the DeepCrack crack segmentation dataset.

Loads image-mask pairs from separate directories, matching filenames
(image: .jpg, mask: .png). Masks are binarized at threshold 127.
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    """Dataset for crack segmentation.

    Args:
        image_dir: Path to the directory containing input images (.jpg).
        mask_dir: Path to the directory containing ground-truth masks (.png).
        transform: Optional callable that takes (image, mask) as PIL Images
                   and returns (image_tensor, mask_tensor).
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Collect image filenames and verify matching masks exist
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
        ])

        # Build a map of mask basenames for fast lookup
        mask_files = set(os.listdir(mask_dir))
        self.pairs = []
        for img_name in self.images:
            base = os.path.splitext(img_name)[0]
            # Try common mask extensions
            for ext in ['.png', '.jpg', '.bmp']:
                mask_name = base + ext
                if mask_name in mask_files:
                    self.pairs.append((img_name, mask_name))
                    break

        print(f"[CrackDataset] Found {len(self.pairs)} image-mask pairs "
              f"from {image_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]

        # Load image as RGB and mask as grayscale
        image = Image.open(
            os.path.join(self.image_dir, img_name)
        ).convert('RGB')
        mask = Image.open(
            os.path.join(self.mask_dir, mask_name)
        ).convert('L')

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # Default: convert to numpy arrays
            image = np.array(image, dtype=np.float32) / 255.0
            mask = np.array(mask, dtype=np.float32) / 255.0
            # Binarize mask
            mask = (mask > 0.5).astype(np.float32)

        return image, mask
