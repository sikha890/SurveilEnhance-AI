
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.srcnn_patch_dataset import SRCNNPatchDataset


class TestSRCNNPatchDataset(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()

        self.root = Path(self.temp_dir.name)

        self.hr_dir = self.root / "HR"
        self.lr_dir = self.root / "LR"

        self.hr_dir.mkdir()
        self.lr_dir.mkdir()

        rng = np.random.default_rng(42)

        for index in range(2):

            # Create a synthetic HR image.
            hr_image = rng.integers(
                0,
                256,
                size=(256, 256, 3),
                dtype=np.uint8,
            )

            # Generate its corresponding 4x LR image.
            lr_image = cv2.resize(
                hr_image,
                (64, 64),
                interpolation=cv2.INTER_AREA,
            )

            filename = f"image_{index}.png"

            cv2.imwrite(
                str(self.hr_dir / filename),
                hr_image,
            )

            cv2.imwrite(
                str(self.lr_dir / filename),
                lr_image,
            )

    def tearDown(self):

        self.temp_dir.cleanup()

    def test_training_dataset_length(self):

        dataset = SRCNNPatchDataset(
            self.hr_dir,
            self.lr_dir,
            patch_size=128,
            patches_per_image=3,
            training=True,
        )

        self.assertEqual(len(dataset), 6)

    def test_validation_dataset_length(self):

        dataset = SRCNNPatchDataset(
            self.hr_dir,
            self.lr_dir,
            patch_size=128,
            training=False,
        )

        self.assertEqual(len(dataset), 2)

    def test_patch_tensor_dimensions(self):

        dataset = SRCNNPatchDataset(
            self.hr_dir,
            self.lr_dir,
            patch_size=128,
            training=True,
        )

        lr_tensor, hr_tensor = dataset[0]

        self.assertEqual(
            tuple(lr_tensor.shape),
            (3, 128, 128),
        )

        self.assertEqual(
            tuple(hr_tensor.shape),
            (3, 116, 116),
        )

    def test_pixel_values_are_normalized(self):

        dataset = SRCNNPatchDataset(
            self.hr_dir,
            self.lr_dir,
            patch_size=128,
            training=True,
        )

        lr_tensor, hr_tensor = dataset[0]

        self.assertTrue(torch.all(lr_tensor >= 0))
        self.assertTrue(torch.all(lr_tensor <= 1))

        self.assertTrue(torch.all(hr_tensor >= 0))
        self.assertTrue(torch.all(hr_tensor <= 1))

    def test_validation_patch_is_deterministic(self):

        dataset = SRCNNPatchDataset(
            self.hr_dir,
            self.lr_dir,
            patch_size=128,
            training=False,
        )

        lr_first, hr_first = dataset[0]
        lr_second, hr_second = dataset[0]

        self.assertTrue(
            torch.equal(lr_first, lr_second)
        )

        self.assertTrue(
            torch.equal(hr_first, hr_second)
        )

    def test_invalid_patch_size_is_rejected(self):

        with self.assertRaises(ValueError):

            SRCNNPatchDataset(
                self.hr_dir,
                self.lr_dir,
                patch_size=127,
                scale_factor=4,
            )


if __name__ == "__main__":
    unittest.main()
