
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.generate_div2k_pairs import (
    crop_to_scale,
    discover_images,
    process_split,
)


class TestDIV2KPipeline(unittest.TestCase):

    def test_crop_to_scale(self):
        image = np.zeros((1357, 2041, 3), dtype=np.uint8)

        result = crop_to_scale(image, 4)

        self.assertEqual(result.shape, (1356, 2040, 3))

    def test_already_divisible_dimensions(self):
        image = np.zeros((256, 512, 3), dtype=np.uint8)

        result = crop_to_scale(image, 4)

        self.assertEqual(result.shape, (256, 512, 3))

    def test_reject_image_smaller_than_scale(self):
        image = np.zeros((3, 3, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            crop_to_scale(image, 4)

    def test_recursive_image_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()

            cv2.imwrite(
                str(root / "image1.png"),
                np.zeros((16, 16, 3), dtype=np.uint8),
            )

            cv2.imwrite(
                str(nested / "image2.png"),
                np.zeros((16, 16, 3), dtype=np.uint8),
            )

            (root / "notes.txt").write_text("not an image")

            images = discover_images(
                root,
                (".png", ".jpg", ".jpeg"),
            )

            self.assertEqual(len(images), 2)

    def test_missing_source_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"

            with self.assertRaises(FileNotFoundError):
                discover_images(missing, (".png",))

    def test_generate_one_hr_lr_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            source = root / "source"
            hr_output = root / "output" / "HR"
            lr_output = root / "output" / "LR"

            source.mkdir()

            image = np.random.default_rng(42).integers(
                0,
                256,
                size=(64, 64, 3),
                dtype=np.uint8,
            )

            cv2.imwrite(str(source / "sample.png"), image)

            config = {
                "dataset": {
                    "supported_extensions": [".png"],
                    "scale_factor": 4,
                },
                "degradation": {
                    "blur_kernel": 5,
                    "blur_sigma": 1.0,
                    "noise_std": 10.0,
                    "jpeg_quality": 40,
                },
                "reproducibility": {
                    "random_seed": 42,
                },
            }

            processed, failed = process_split(
                split_name="test",
                source_dir=source,
                hr_output_dir=hr_output,
                lr_output_dir=lr_output,
                config=config,
            )

            self.assertEqual(processed, 1)
            self.assertEqual(failed, 0)

            hr = cv2.imread(
                str(hr_output / "sample.png")
            )

            lr = cv2.imread(
                str(lr_output / "sample.png")
            )

            self.assertIsNotNone(hr)
            self.assertIsNotNone(lr)

            self.assertEqual(hr.shape, (64, 64, 3))
            self.assertEqual(lr.shape, (16, 16, 3))


if __name__ == "__main__":
    unittest.main()
