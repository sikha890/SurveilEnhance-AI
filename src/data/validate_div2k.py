
"""
DIV2K HR-LR Dataset Validator.

Checks:
1. HR and LR filename matching.
2. Missing and extra image pairs.
3. Image readability.
4. Image dimensions.
5. Exact 4x spatial scale relationship.
6. Image channel count and data type.
"""

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCALE_FACTOR = 4

DATASETS = {
    "TRAIN": {
        "hr": PROJECT_ROOT / "data/DIV2K/train/HR",
        "lr": PROJECT_ROOT / "data/DIV2K/train/LR",
        "expected": 800,
    },
    "VALIDATION": {
        "hr": PROJECT_ROOT / "data/DIV2K/validation/HR",
        "lr": PROJECT_ROOT / "data/DIV2K/validation/LR",
        "expected": 100,
    },
}


def get_image_files(directory):
    """Return PNG files indexed by filename."""

    if not directory.exists():
        raise FileNotFoundError(f"Missing directory: {directory}")

    return {
        path.name: path
        for path in directory.glob("*.png")
        if path.is_file()
    }


def validate_split(name, hr_dir, lr_dir, expected):
    """Validate HR-LR pairs for one dataset split."""

    print(f"\n{'=' * 60}")
    print(f"VALIDATING {name} DATASET")
    print(f"{'=' * 60}")

    hr_files = get_image_files(hr_dir)
    lr_files = get_image_files(lr_dir)

    hr_names = set(hr_files)
    lr_names = set(lr_files)

    missing_lr = sorted(hr_names - lr_names)
    missing_hr = sorted(lr_names - hr_names)

    print("HR images:", len(hr_files))
    print("LR images:", len(lr_files))
    print("Expected pairs:", expected)

    errors = 0

    if len(hr_files) != expected:
        print("[ERROR] Unexpected HR image count.")
        errors += 1

    if len(lr_files) != expected:
        print("[ERROR] Unexpected LR image count.")
        errors += 1

    if missing_lr:
        print("[ERROR] Missing LR pairs:", missing_lr[:10])
        errors += len(missing_lr)

    if missing_hr:
        print("[ERROR] Missing HR pairs:", missing_hr[:10])
        errors += len(missing_hr)

    matched_names = sorted(hr_names & lr_names)

    for index, filename in enumerate(matched_names):

        hr = cv2.imread(
            str(hr_files[filename]),
            cv2.IMREAD_COLOR
        )

        lr = cv2.imread(
            str(lr_files[filename]),
            cv2.IMREAD_COLOR
        )

        if hr is None or lr is None:
            print(f"[ERROR] Cannot read image pair: {filename}")
            errors += 1
            continue

        if hr.ndim != 3 or lr.ndim != 3:
            print(f"[ERROR] Invalid image dimensions: {filename}")
            errors += 1
            continue

        hr_height, hr_width, hr_channels = hr.shape
        lr_height, lr_width, lr_channels = lr.shape

        expected_lr_height = hr_height // SCALE_FACTOR
        expected_lr_width = hr_width // SCALE_FACTOR

        if (
            hr_height % SCALE_FACTOR != 0
            or hr_width % SCALE_FACTOR != 0
        ):
            print(f"[ERROR] HR dimensions not divisible by 4: {filename}")
            errors += 1
            continue

        if (
            lr_height != expected_lr_height
            or lr_width != expected_lr_width
        ):
            print(f"[ERROR] Incorrect LR dimensions: {filename}")
            errors += 1
            continue

        if hr_channels != 3 or lr_channels != 3:
            print(f"[ERROR] Expected 3 channels: {filename}")
            errors += 1
            continue

        if hr.dtype != "uint8" or lr.dtype != "uint8":
            print(f"[ERROR] Unexpected image dtype: {filename}")
            errors += 1
            continue

        if index == 0:
            print("\nExample verified pair:")
            print("Filename:", filename)
            print("HR shape:", hr.shape)
            print("LR shape:", lr.shape)

    valid_pairs = len(matched_names) - sum(
        1 for filename in matched_names
        if not (
            cv2.imread(str(hr_files[filename]), cv2.IMREAD_COLOR) is not None
            and cv2.imread(str(lr_files[filename]), cv2.IMREAD_COLOR) is not None
        )
    )

    print("\nValidation errors:", errors)
    print("Matching filenames:", len(matched_names))

    if errors == 0:
        print(f"[PASS] {name} dataset validation successful.")
    else:
        print(f"[FAIL] {name} dataset has validation errors.")

    return errors


def main():
    total_errors = 0

    for name, paths in DATASETS.items():
        errors = validate_split(
            name=name,
            hr_dir=paths["hr"],
            lr_dir=paths["lr"],
            expected=paths["expected"],
        )
        total_errors += errors

    print(f"\n{'=' * 60}")
    print("FINAL DIV2K VALIDATION REPORT")
    print(f"{'=' * 60}")
    print("Total validation errors:", total_errors)

    if total_errors == 0:
        print("[SUCCESS] All DIV2K HR-LR pairs passed validation.")
    else:
        print("[FAILURE] Please investigate the reported errors.")


if __name__ == "__main__":
    main()
