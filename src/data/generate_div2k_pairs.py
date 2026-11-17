
"""
DIV2K HR-LR Dataset Generator.

Creates separate training and validation HR-LR image pairs.

Pipeline:
1. Discover source images recursively.
2. Read original HR images.
3. Crop dimensions to multiples of the scale factor.
4. Generate synthetic LR images using the reusable degradation module.
5. Save aligned HR-LR pairs.
6. Report processing statistics.
"""

from pathlib import Path

import cv2
import yaml

from src.data.degradation import degrade_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "div2k.yaml"


def load_config():
    """Load DIV2K generation configuration."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def discover_images(directory, extensions):
    """Recursively discover supported image files."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {directory}"
        )

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def crop_to_scale(image, scale_factor):
    """
    Crop the bottom and right edges so dimensions
    are divisible by the scale factor.

    This keeps HR and LR dimensions aligned.
    """

    height, width = image.shape[:2]

    new_height = (height // scale_factor) * scale_factor
    new_width = (width // scale_factor) * scale_factor

    if new_height == 0 or new_width == 0:
        raise ValueError("Image is too small for the selected scale factor.")

    return image[:new_height, :new_width]


def process_split(
    split_name,
    source_dir,
    hr_output_dir,
    lr_output_dir,
    config,
    seed_offset=0,
):
    """Generate paired images for one dataset split."""

    extensions = tuple(
        ext.lower()
        for ext in config["dataset"]["supported_extensions"]
    )

    image_paths = discover_images(source_dir, extensions)

    hr_output_dir.mkdir(parents=True, exist_ok=True)
    lr_output_dir.mkdir(parents=True, exist_ok=True)

    degradation = config["degradation"]
    scale_factor = config["dataset"]["scale_factor"]
    base_seed = config["reproducibility"]["random_seed"]

    processed = 0
    failed = 0

    print(f"\n{'=' * 65}")
    print(f"PROCESSING {split_name.upper()} DATASET")
    print(f"{'=' * 65}")
    print("Source images:", len(image_paths))

    for index, image_path in enumerate(image_paths):

        hr_image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if hr_image is None:
            print(f"[FAILED] Cannot read: {image_path.name}")
            failed += 1
            continue

        try:
            original_height, original_width = hr_image.shape[:2]

            hr_image = crop_to_scale(
                hr_image,
                scale_factor
            )

            lr_image = degrade_image(
                hr_image,
                scale_factor=scale_factor,
                blur_kernel=degradation["blur_kernel"],
                blur_sigma=degradation["blur_sigma"],
                noise_std=degradation["noise_std"],
                jpeg_quality=degradation["jpeg_quality"],
                seed=base_seed + seed_offset + index,
            )

            hr_path = hr_output_dir / f"{image_path.stem}.png"
            lr_path = lr_output_dir / f"{image_path.stem}.png"

            if not cv2.imwrite(str(hr_path), hr_image):
                raise RuntimeError("Could not save HR image.")

            if not cv2.imwrite(str(lr_path), lr_image):
                hr_path.unlink(missing_ok=True)
                raise RuntimeError("Could not save LR image.")

            processed += 1

            if processed % 25 == 0 or processed == len(image_paths):
                print(
                    f"[{processed}/{len(image_paths)}] "
                    f"{image_path.name} | "
                    f"Original: {original_width}x{original_height} | "
                    f"HR: {hr_image.shape[1]}x{hr_image.shape[0]} | "
                    f"LR: {lr_image.shape[1]}x{lr_image.shape[0]}"
                )

        except Exception as error:
            print(f"[FAILED] {image_path.name}: {error}")
            failed += 1

    print(f"\n{split_name.upper()} SUMMARY")
    print("Discovered:", len(image_paths))
    print("Processed:", processed)
    print("Failed:", failed)
    print("HR output:", hr_output_dir)
    print("LR output:", lr_output_dir)

    return processed, failed


def main():
    config = load_config()

    source = config["source"]
    output = config["output"]

    train_processed, train_failed = process_split(
        split_name="train",
        source_dir=PROJECT_ROOT / source["train_directory"],
        hr_output_dir=PROJECT_ROOT / output["train_hr_directory"],
        lr_output_dir=PROJECT_ROOT / output["train_lr_directory"],
        config=config,
        seed_offset=0,
    )

    validation_processed, validation_failed = process_split(
        split_name="validation",
        source_dir=PROJECT_ROOT / source["validation_directory"],
        hr_output_dir=PROJECT_ROOT / output["validation_hr_directory"],
        lr_output_dir=PROJECT_ROOT / output["validation_lr_directory"],
        config=config,
        seed_offset=10000,
    )

    print(f"\n{'=' * 65}")
    print("DIV2K DATASET GENERATION COMPLETE")
    print(f"{'=' * 65}")

    print("Training pairs:", train_processed)
    print("Training failures:", train_failed)

    print("Validation pairs:", validation_processed)
    print("Validation failures:", validation_failed)

    print(
        "Total successful pairs:",
        train_processed + validation_processed
    )


if __name__ == "__main__":
    main()
