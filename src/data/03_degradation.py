
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# EXPERIMENT 03: IMAGE DEGRADATION PIPELINE
# =====================================================

print("=" * 60)
print("EXPERIMENT 03 - IMAGE DEGRADATION PIPELINE")
print("=" * 60)


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "images"
    / "experiment_03"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. CREATE SYNTHETIC HR IMAGE

height, width = 256, 256

hr_image = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

# Background gradient
for y in range(height):
    intensity = int(40 + (y / height) * 100)
    hr_image[y, :] = [
        intensity,
        intensity,
        intensity
    ]

# Geometric objects
cv2.rectangle(
    hr_image, (25, 25), (105, 105),
    (0, 0, 255), -1
)

cv2.circle(
    hr_image, (180, 75), 40,
    (0, 255, 0), -1
)

cv2.rectangle(
    hr_image, (35, 145), (115, 225),
    (255, 0, 0), -1
)

cv2.circle(
    hr_image, (180, 180), 35,
    (255, 255, 255), -1
)

cv2.putText(
    hr_image,
    "SURVEILLANCE",
    (55, 245),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 0),
    2
)


# 3. GAUSSIAN BLUR

blurred_image = cv2.GaussianBlur(
    hr_image,
    (5, 5),
    sigmaX=1.0
)


# 4. GAUSSIAN NOISE

rng = np.random.default_rng(seed=42)

noise = rng.normal(
    loc=0,
    scale=10,
    size=hr_image.shape
)

noisy_image = np.clip(
    hr_image.astype(np.float32) + noise,
    0,
    255
).astype(np.uint8)


# 5. JPEG COMPRESSION

jpeg_quality = 40

success, encoded_image = cv2.imencode(
    ".jpg",
    hr_image,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        jpeg_quality
    ]
)

if not success:
    raise RuntimeError("JPEG encoding failed")

compressed_image = cv2.imdecode(
    encoded_image,
    cv2.IMREAD_COLOR
)


# 6. DOWNSAMPLING

lr_image = cv2.resize(
    hr_image,
    (64, 64),
    interpolation=cv2.INTER_AREA
)


# 7. COMBINED DEGRADATION PIPELINE

combined_blur = cv2.GaussianBlur(
    hr_image,
    (5, 5),
    sigmaX=1.0
)

combined_noise = np.clip(
    combined_blur.astype(np.float32)
    + rng.normal(
        0,
        10,
        combined_blur.shape
    ),
    0,
    255
).astype(np.uint8)

success, combined_encoded = cv2.imencode(
    ".jpg",
    combined_noise,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        jpeg_quality
    ]
)

if not success:
    raise RuntimeError("Combined JPEG encoding failed")

combined_compressed = cv2.imdecode(
    combined_encoded,
    cv2.IMREAD_COLOR
)

combined_lr = cv2.resize(
    combined_compressed,
    (64, 64),
    interpolation=cv2.INTER_AREA
)


# 8. SAVE ALL OUTPUTS

images = {
    "original_hr.png": hr_image,
    "gaussian_blur.png": blurred_image,
    "gaussian_noise.png": noisy_image,
    "jpeg_compression.png": compressed_image,
    "downsampled_lr.png": lr_image,
    "combined_degraded_lr.png": combined_lr
}

print("\nSAVING OUTPUT IMAGES")

for filename, image in images.items():

    output_path = OUTPUT_DIR / filename

    saved = cv2.imwrite(
        str(output_path),
        image
    )

    if not saved:
        raise RuntimeError(
            f"Failed to save {filename}"
        )

    print("Saved:", filename)


# 9. VISUAL COMPARISON

comparison = [
    ("Original HR", hr_image),
    ("Gaussian Blur", blurred_image),
    ("Gaussian Noise", noisy_image),
    ("JPEG Compression", compressed_image),
    ("Downsampled LR", lr_image),
    ("Combined LR", combined_lr)
]

plt.figure(figsize=(16, 8))

for index, (title, image) in enumerate(
    comparison,
    start=1
):

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    plt.subplot(2, 3, index)
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()

comparison_path = (
    OUTPUT_DIR / "degradation_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=150
)

plt.show()


# 10. FINAL REPORT

print("\nIMAGE DIMENSIONS")
print("Original HR:", hr_image.shape)
print("Downsampled LR:", lr_image.shape)
print("Combined LR:", combined_lr.shape)

print("\nDEGRADATION PARAMETERS")
print("Gaussian kernel: 5 x 5")
print("Gaussian sigma: 1.0")
print("Noise standard deviation: 10")
print("JPEG quality:", jpeg_quality)
print("Downsampling: 4x")

print("\nOUTPUT DIRECTORY:", OUTPUT_DIR)
print("\nExperiment 03 completed successfully!")
