"""
Transforms for crack segmentation training and evaluation.

Provides joint image-mask transforms that apply identical spatial
transformations to both image and mask (critical for segmentation),
while applying color augmentations only to the image.
"""

import random
import numpy as np
import torch
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF


class JointTransform:
    """Applies synchronized transforms to both image and mask.

    Spatial transforms (flip, rotate, crop) are applied identically to
    both image and mask. Color transforms are applied only to the image.

    Args:
        resize: Tuple (H, W) to resize both image and mask.
        augment: If True, apply data augmentation (for training).
    """

    def __init__(self, resize=(256, 256), augment=False):
        self.resize = resize
        self.augment = augment
        # ImageNet normalization stats
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image, mask):
        """
        Args:
            image: PIL Image (RGB).
            mask: PIL Image (Grayscale).

        Returns:
            image_tensor: Float tensor [3, H, W], normalized.
            mask_tensor: Float tensor [1, H, W], binary (0 or 1).
        """
        # --- Resize ---
        image = TF.resize(image, self.resize, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.resize, interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            # --- Random Horizontal Flip ---
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # --- Random Vertical Flip ---
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # --- Random Rotation (0, 90, 180, 270) ---
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # --- Random Affine (small rotation + translate) ---
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                translate = [random.randint(-20, 20), random.randint(-20, 20)]
                image = TF.affine(image, angle=angle, translate=translate,
                                  scale=1.0, shear=0)
                mask = TF.affine(mask, angle=angle, translate=translate,
                                 scale=1.0, shear=0,
                                 interpolation=TF.InterpolationMode.NEAREST)

            # --- Color Jitter (image only) ---
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

            # --- Gaussian Blur (image only) ---
            if random.random() > 0.3:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # --- Convert to tensors ---
        image = TF.to_tensor(image)  # [3, H, W], float [0, 1]
        mask = torch.from_numpy(np.array(mask, dtype=np.float32))  # [H, W]

        # --- Binarize mask ---
        mask = (mask > 127.0).float()  # Binary: 0 or 1
        mask = mask.unsqueeze(0)  # [1, H, W]

        # --- Normalize image ---
        image = TF.normalize(image, self.mean, self.std)

        return image, mask


def get_train_transform(resize=(256, 256)):
    """Returns the training transform with data augmentation."""
    return JointTransform(resize=resize, augment=True)


def get_val_transform(resize=(256, 256)):
    """Returns the validation/test transform (no augmentation)."""
    return JointTransform(resize=resize, augment=False)
