
"""
Loss functions for SRCNN training.
"""

import torch.nn as nn


class SRCNNLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, prediction, target):

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target dimensions must match."
            )

        return self.mse(prediction, target)
