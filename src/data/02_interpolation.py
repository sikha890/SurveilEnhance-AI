
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


print("=" * 60)
print("EXPERIMENT 02 - IMAGE RESIZING AND INTERPOLATION")
print("=" * 60)


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "images"
    / "experiment_02"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. CREATE SYNTHETIC HIGH-RESOLUTION IMAGE

height, width = 256, 256

hr_image = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

for y in range(height):
    intensity = int(40 + (y / height) * 100)
    hr_image[y, :] = [
        intensity,
        intensity,
        intensity
    ]

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
    "SR TEST",
    (115, 245),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 0),
    2
)

cv2.line(
    hr_image, (0, 128), (255, 128),
    (255, 255, 255), 1
)


# 3. DOWNSAMPLE HR TO LR

lr_image = cv2.resize(
    hr_image,
    (64, 64),
    interpolation=cv2.INTER_AREA
)


# 4. UPSCALE USING THREE METHODS

nearest_image = cv2.resize(
    lr_image,
    (256, 256),
    interpolation=cv2.INTER_NEAREST
)

bilinear_image = cv2.resize(
    lr_image,
    (256, 256),
    interpolation=cv2.INTER_LINEAR
)

bicubic_image = cv2.resize(
    lr_image,
    (256, 256),
    interpolation=cv2.INTER_CUBIC
)


# 5. EVALUATE QUALITY

results = {
    "Nearest": nearest_image,
    "Bilinear": bilinear_image,
    "Bicubic": bicubic_image
}

print("\nIMAGE DIMENSIONS")
print("Original HR:", hr_image.shape)
print("Low Resolution:", lr_image.shape)

print("\nQUALITY METRICS")
print("-" * 60)

for method, image in results.items():

    psnr_value = peak_signal_noise_ratio(
        hr_image,
        image,
        data_range=255
    )

    ssim_value = structural_similarity(
        hr_image,
        image,
        channel_axis=2,
        data_range=255
    )

    print(
        f"{method:10} | "
        f"PSNR: {psnr_value:.4f} dB | "
        f"SSIM: {ssim_value:.4f}"
    )


# 6. SAVE IMAGES

images_to_save = {
    "original_hr.png": hr_image,
    "low_resolution.png": lr_image,
    "nearest.png": nearest_image,
    "bilinear.png": bilinear_image,
    "bicubic.png": bicubic_image
}

for filename, image in images_to_save.items():

    output_path = OUTPUT_DIR / filename

    success = cv2.imwrite(
        str(output_path),
        image
    )

    if not success:
        raise RuntimeError(
            f"Could not save image: {output_path}"
        )


# 7. CREATE COMPARISON FIGURE

comparison_images = [
    ("Original HR", hr_image),
    ("Low Resolution", lr_image),
    ("Nearest", nearest_image),
    ("Bilinear", bilinear_image),
    ("Bicubic", bicubic_image)
]

plt.figure(figsize=(15, 5))

for index, (title, image) in enumerate(
    comparison_images,
    start=1
):

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    plt.subplot(1, 5, index)
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()

comparison_path = (
    OUTPUT_DIR / "interpolation_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=150
)

plt.show()


# 8. FINAL VERIFICATION

print("\nOUTPUT DIRECTORY:", OUTPUT_DIR)

print("\nGENERATED FILES:")

for path in sorted(OUTPUT_DIR.glob("*.png")):
    print(path.name, "-", path.stat().st_size, "bytes")

print("\nExperiment 02 completed successfully!")
