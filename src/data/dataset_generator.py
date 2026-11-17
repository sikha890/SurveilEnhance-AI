
from pathlib import Path
import sys

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.degradation import degrade_image


CONFIG_PATH = PROJECT_ROOT / "configs" / "degradation.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():

    config = load_config()

    source_dir = PROJECT_ROOT / "data" / "source"

    hr_dir = PROJECT_ROOT / config["output"]["hr_directory"]
    lr_dir = PROJECT_ROOT / config["output"]["lr_directory"]

    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = tuple(
        ext.lower()
        for ext in config["dataset"]["supported_extensions"]
    )

    image_paths = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )

    if not image_paths:
        print("No source images found.")
        print("Add images to:", source_dir)
        return

    degradation = config["degradation"]
    scale_factor = config["dataset"]["scale_factor"]
    seed = config["reproducibility"]["random_seed"]

    print("=" * 60)
    print("HR-LR DATASET GENERATOR")
    print("=" * 60)

    processed = 0
    failed = 0

    for index, image_path in enumerate(image_paths):

        hr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if hr_image is None:
            print("Could not read:", image_path.name)
            failed += 1
            continue

        height, width = hr_image.shape[:2]

        if height % scale_factor or width % scale_factor:
            print("Skipping incompatible dimensions:", image_path.name)
            failed += 1
            continue

        lr_image = degrade_image(
            hr_image,
            scale_factor=scale_factor,
            blur_kernel=degradation["blur_kernel"],
            blur_sigma=degradation["blur_sigma"],
            noise_std=degradation["noise_std"],
            jpeg_quality=degradation["jpeg_quality"],
            seed=seed + index
        )

        hr_output = hr_dir / f"{image_path.stem}.png"
        lr_output = lr_dir / f"{image_path.stem}.png"

        if not cv2.imwrite(str(hr_output), hr_image):
            print("Failed to save HR:", image_path.name)
            failed += 1
            continue

        if not cv2.imwrite(str(lr_output), lr_image):
            hr_output.unlink(missing_ok=True)
            print("Failed to save LR:", image_path.name)
            failed += 1
            continue

        print(
            f"Processed: {image_path.name} | "
            f"HR: {width}x{height} | "
            f"LR: {lr_image.shape[1]}x{lr_image.shape[0]}"
        )

        processed += 1

    print("\nDATASET GENERATION SUMMARY")
    print("Total images:", len(image_paths))
    print("Successfully processed:", processed)
    print("Failed:", failed)

    print("HR output:", hr_dir)
    print("LR output:", lr_dir)


if __name__ == "__main__":
    main()
