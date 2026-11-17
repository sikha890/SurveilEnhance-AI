
"""
SRCNN: Super-Resolution Convolutional Neural Network.

Three convolutional layers:
1. Feature extraction: 9x9, 64 filters
2. Nonlinear mapping: 1x1, 32 filters
3. Reconstruction: 5x5, 3 output channels

Input:  RGB image tensor (N, 3, H, W)
Output: RGB image tensor (N, 3, H-12, W-12)
"""

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """
    Basic SRCNN model for RGB image super-resolution.
    """

    def __init__(self):
        super().__init__()

        # Layer 1: Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=9,
                padding=0,
            ),
            nn.ReLU(inplace=True),
        )

        # Layer 2: Nonlinear mapping
        self.nonlinear_mapping = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
        )

        # Layer 3: Image reconstruction
        self.reconstruction = nn.Conv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=5,
            padding=0,
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Bicubic-upsampled RGB tensor (N, 3, H, W).

        Returns:
            Reconstructed RGB tensor.
        """

        if x.ndim != 4:
            raise ValueError(
                "Expected a 4D tensor with shape (N, 3, H, W)"
            )

        if x.shape[1] != 3:
            raise ValueError(
                "SRCNN expects 3 input channels (RGB)"
            )

        if x.shape[2] < 13 or x.shape[3] < 13:
            raise ValueError(
                "Input height and width must be at least 13 pixels"
            )

        features = self.feature_extraction(x)

        mapped_features = self.nonlinear_mapping(features)

        output = self.reconstruction(mapped_features)

        return output
