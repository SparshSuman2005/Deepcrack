"""
U-Net Model for binary segmentation.

Standard U-Net architecture with 4 encoder levels, a bottleneck,
and 4 decoder levels with skip connections. Uses batch normalization
for training stability.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI 2015.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive (Conv3x3 → BatchNorm → ReLU) blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Encoder block: DoubleConv → MaxPool2x2."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)       # Skip connection output
        pooled = self.pool(features)   # Downsampled for next level
        return features, pooled


class Decoder(nn.Module):
    """Decoder block: ConvTranspose2x2 → Concat skip → DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch due to odd dimensions
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
        ])

        x = torch.cat([skip, x], dim=1)  # Concatenate along channel dim
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net for binary segmentation.

    Architecture:
        Encoder: 3→64→128→256→512 (with max-pooling between levels)
        Bottleneck: 512→1024
        Decoder: 1024→512→256→128→64 (with skip connections)
        Output: 64→1 (1x1 conv)

    Args:
        in_channels: Number of input channels (default: 3 for RGB).
        out_channels: Number of output channels (default: 1 for binary mask).
        features: List of feature sizes for each encoder level.
    """

    def __init__(self, in_channels=3, out_channels=1,
                 features=None):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        # --- Encoder path ---
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(Encoder(ch, f))
            ch = f

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # --- Decoder path ---
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        ch = features[-1] * 2
        for f in reversed_features:
            self.decoders.append(Decoder(ch, f))
            ch = f

        # --- Final 1x1 convolution ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (reverse order of skip connections)
        skip_connections = skip_connections[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])

        # Final output
        x = self.final_conv(x)
        return x


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick sanity check
    model = UNet(in_channels=3, out_channels=1)
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
