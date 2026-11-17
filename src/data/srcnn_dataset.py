
"""
PyTorch Dataset for SRCNN.

Loads paired HR and LR images, converts them into tensors,
upsamples LR images, and crops HR targets to match SRCNN output.
"""

from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SRCNNDataset(Dataset):

    def __init__(
        self,
        hr_dir,
        lr_dir,
        scale_factor=4,
        input_crop=6,
    ):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)

        self.scale_factor = scale_factor
        self.input_crop = input_crop

        if self.scale_factor <= 0:
            raise ValueError("Scale factor must be positive.")

        if self.input_crop < 0:
            raise ValueError("Input crop cannot be negative.")

        if not self.hr_dir.is_dir():
            raise FileNotFoundError(
                f"HR directory not found: {self.hr_dir}"
            )

        if not self.lr_dir.is_dir():
            raise FileNotFoundError(
                f"LR directory not found: {self.lr_dir}"
            )

        extensions = {".png", ".jpg", ".jpeg"}

        hr_files = {
            path.name
            for path in self.hr_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        }

        lr_files = {
            path.name
            for path in self.lr_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        }

        if hr_files != lr_files:
            raise ValueError(
                "HR and LR directories must contain matching filenames."
            )

        self.filenames = sorted(hr_files)

        if not self.filenames:
            raise ValueError("No paired images found.")

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def image_to_tensor(image):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        tensor = torch.from_numpy(
            image.copy()
        ).permute(2, 0, 1).float() / 255.0

        return tensor

    def __getitem__(self, index):

        filename = self.filenames[index]

        hr_path = self.hr_dir / filename
        lr_path = self.lr_dir / filename

        hr_image = cv2.imread(
            str(hr_path),
            cv2.IMREAD_COLOR
        )

        lr_image = cv2.imread(
            str(lr_path),
            cv2.IMREAD_COLOR
        )

        if hr_image is None or lr_image is None:
            raise ValueError(
                f"Could not read image pair: {filename}"
            )

        hr_height, hr_width = hr_image.shape[:2]

        expected_height = hr_height // self.scale_factor
        expected_width = hr_width // self.scale_factor

        if (
            hr_height % self.scale_factor != 0
            or hr_width % self.scale_factor != 0
            or lr_image.shape[:2] != (
                expected_height,
                expected_width
            )
        ):
            raise ValueError(
                f"Invalid HR-LR dimensions: {filename}"
            )

        hr_tensor = self.image_to_tensor(hr_image)
        lr_tensor = self.image_to_tensor(lr_image)

        # Bicubic upsampling
        lr_tensor = F.interpolate(
            lr_tensor.unsqueeze(0),
            size=(hr_height, hr_width),
            mode="bicubic",
            align_corners=False
        ).squeeze(0)

        # Clamp bicubic overshoot
        lr_tensor = torch.clamp(
            lr_tensor,
            min=0.0,
            max=1.0
        )

        # Crop HR target to match SRCNN output
        crop = self.input_crop

        if crop > 0:
            if (
                hr_height <= 2 * crop
                or hr_width <= 2 * crop
            ):
                raise ValueError(
                    f"Image too small for cropping: {filename}"
                )

            hr_tensor = hr_tensor[
                :,
                crop:-crop,
                crop:-crop
            ]

        return lr_tensor, hr_tensor
