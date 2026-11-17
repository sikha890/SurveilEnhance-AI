
from pathlib import Path

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "degradation.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_dataset():

    config = load_config()

    hr_dir = PROJECT_ROOT / config["output"]["hr_directory"]
    lr_dir = PROJECT_ROOT / config["output"]["lr_directory"]

    scale = config["dataset"]["scale_factor"]

    if not hr_dir.is_dir() or not lr_dir.is_dir():
        raise FileNotFoundError("HR or LR dataset directory is missing")

    extensions = tuple(
        ext.lower()
        for ext in config["dataset"]["supported_extensions"]
    )

    hr_files = {
        path.name: path
        for path in hr_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }

    lr_files = {
        path.name: path
        for path in lr_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }

    print("=" * 60)
    print("HR-LR DATASET VALIDATION")
    print("=" * 60)

    missing_lr = sorted(set(hr_files) - set(lr_files))
    missing_hr = sorted(set(lr_files) - set(hr_files))

    if missing_lr:
        print("Missing LR pairs:", missing_lr)

    if missing_hr:
        print("Missing HR pairs:", missing_hr)

    errors = len(missing_lr) + len(missing_hr)
    valid_pairs = 0

    for filename in sorted(set(hr_files) & set(lr_files)):

        hr = cv2.imread(str(hr_files[filename]), cv2.IMREAD_COLOR)
        lr = cv2.imread(str(lr_files[filename]), cv2.IMREAD_COLOR)

        if hr is None or lr is None:
            print("INVALID IMAGE:", filename)
            errors += 1
            continue

        hr_height, hr_width = hr.shape[:2]
        lr_height, lr_width = lr.shape[:2]

        expected_height = hr_height // scale
        expected_width = hr_width // scale

        if (
            hr_height % scale != 0
            or hr_width % scale != 0
            or lr_height != expected_height
            or lr_width != expected_width
        ):
            print("INVALID DIMENSIONS:", filename)
            print("HR:", hr.shape, "| LR:", lr.shape)
            errors += 1
            continue

        print(
            f"VALID: {filename} | "
            f"HR: {hr_width}x{hr_height} | "
            f"LR: {lr_width}x{lr_height}"
        )

        valid_pairs += 1

    print("\nVALIDATION SUMMARY")
    print("Valid pairs:", valid_pairs)
    print("Errors:", errors)

    if errors:
        raise SystemExit("Dataset validation failed")

    if valid_pairs == 0:
        raise SystemExit("No valid HR-LR pairs found")

    print("\nDataset validation completed successfully!")


if __name__ == "__main__":
    validate_dataset()
