
import unittest

import cv2
import numpy as np

from src.data.degradation import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    downsample_image,
    degrade_image,
)


class TestDegradation(unittest.TestCase):

    def setUp(self):
        self.image = np.full((256, 256, 3), 128, dtype=np.uint8)

    def test_blur_preserves_shape_and_dtype(self):
        result = apply_gaussian_blur(self.image)

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_noise_is_reproducible(self):
        result1 = apply_gaussian_noise(self.image, std=10, seed=42)
        result2 = apply_gaussian_noise(self.image, std=10, seed=42)

        np.testing.assert_array_equal(result1, result2)

    def test_zero_noise_preserves_image(self):
        result = apply_gaussian_noise(self.image, std=0, seed=42)

        np.testing.assert_array_equal(result, self.image)

    def test_jpeg_preserves_shape_and_dtype(self):
        result = apply_jpeg_compression(self.image, quality=40)

        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_downsampling_dimensions(self):
        result = downsample_image(self.image, scale_factor=4)

        self.assertEqual(result.shape, (64, 64, 3))

    def test_downsampling_rejects_invalid_dimensions(self):
        invalid_image = np.zeros((255, 256, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            downsample_image(invalid_image, scale_factor=4)

    def test_downsampling_rejects_invalid_scale(self):
        with self.assertRaises(ValueError):
            downsample_image(self.image, scale_factor=0)

    def test_complete_degradation_pipeline(self):
        result = degrade_image(
            self.image,
            scale_factor=4,
            blur_kernel=5,
            blur_sigma=1.0,
            noise_std=10,
            jpeg_quality=40,
            seed=42,
        )

        self.assertEqual(result.shape, (64, 64, 3))
        self.assertEqual(result.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
