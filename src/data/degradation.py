
from __future__ import annotations

import cv2
import numpy as np


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """Apply Gaussian blur to a BGR image."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")

    if sigma < 0:
        raise ValueError("sigma cannot be negative")

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma
    )


def apply_gaussian_noise(
    image: np.ndarray,
    std: float = 10.0,
    seed: int | None = None
) -> np.ndarray:
    """Add Gaussian noise to an image."""

    if std < 0:
        raise ValueError("Noise standard deviation cannot be negative")

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=std,
        size=image.shape
    )

    noisy = image.astype(np.float32) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_jpeg_compression(
    image: np.ndarray,
    quality: int = 40
) -> np.ndarray:
    """Apply JPEG compression and decode the result."""

    if not 0 <= quality <= 100:
        raise ValueError("JPEG quality must be between 0 and 100")

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    if not success:
        raise RuntimeError("JPEG encoding failed")

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )

    if decoded is None:
        raise RuntimeError("JPEG decoding failed")

    return decoded


def downsample_image(
    image: np.ndarray,
    scale_factor: int = 4
) -> np.ndarray:
    """Downsample an image by an integer scale factor."""

    if scale_factor < 1:
        raise ValueError("Scale factor must be at least 1")

    height, width = image.shape[:2]

    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError(
            "Image dimensions must be divisible by scale_factor"
        )

    return cv2.resize(
        image,
        (width // scale_factor, height // scale_factor),
        interpolation=cv2.INTER_AREA
    )


def degrade_image(
    image: np.ndarray,
    scale_factor: int = 4,
    blur_kernel: int = 5,
    blur_sigma: float = 1.0,
    noise_std: float = 10.0,
    jpeg_quality: int = 40,
    seed: int | None = 42
) -> np.ndarray:
    """Generate an LR image using a combined degradation pipeline."""

    blurred = apply_gaussian_blur(
        image,
        kernel_size=blur_kernel,
        sigma=blur_sigma
    )

    noisy = apply_gaussian_noise(
        blurred,
        std=noise_std,
        seed=seed
    )

    compressed = apply_jpeg_compression(
        noisy,
        quality=jpeg_quality
    )

    return downsample_image(
        compressed,
        scale_factor=scale_factor
    )
