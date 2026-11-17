
import unittest
import tempfile

from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data.srcnn_dataset import SRCNNDataset


class TestSRCNNDataset(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.hr_dir = self.root / "HR"
        self.lr_dir = self.root / "LR"

        self.hr_dir.mkdir()
        self.lr_dir.mkdir()

        hr = np.full(
            (256, 256, 3),
            128,
            dtype=np.uint8
        )

        lr = cv2.resize(
            hr,
            (64, 64),
            interpolation=cv2.INTER_AREA
        )

        cv2.imwrite(str(self.hr_dir / "test.png"), hr)
        cv2.imwrite(str(self.lr_dir / "test.png"), lr)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_length(self):

        dataset = SRCNNDataset(
            self.hr_dir,
            self.lr_dir
        )

        self.assertEqual(len(dataset), 1)

    def test_tensor_dimensions(self):

        dataset = SRCNNDataset(
            self.hr_dir,
            self.lr_dir
        )

        lr, hr = dataset[0]

        self.assertEqual(
            tuple(lr.shape),
            (3, 256, 256)
        )

        self.assertEqual(
            tuple(hr.shape),
            (3, 244, 244)
        )

    def test_pixel_range(self):

        dataset = SRCNNDataset(
            self.hr_dir,
            self.lr_dir
        )

        lr, hr = dataset[0]

        self.assertGreaterEqual(lr.min().item(), 0.0)
        self.assertLessEqual(lr.max().item(), 1.0)

        self.assertGreaterEqual(hr.min().item(), 0.0)
        self.assertLessEqual(hr.max().item(), 1.0)

    def test_tensor_dtype(self):

        dataset = SRCNNDataset(
            self.hr_dir,
            self.lr_dir
        )

        lr, hr = dataset[0]

        self.assertEqual(lr.dtype, torch.float32)
        self.assertEqual(hr.dtype, torch.float32)

    def test_dataloader_batch(self):

        dataset = SRCNNDataset(
            self.hr_dir,
            self.lr_dir
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        lr_batch, hr_batch = next(iter(loader))

        self.assertEqual(
            tuple(lr_batch.shape),
            (1, 3, 256, 256)
        )

        self.assertEqual(
            tuple(hr_batch.shape),
            (1, 3, 244, 244)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
