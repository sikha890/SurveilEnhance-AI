
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "TRAIN": PROJECT_ROOT / "data/external/DIV2K/train/HR",
    "VALIDATION": PROJECT_ROOT / "data/external/DIV2K/validation/HR",
}


def inspect_images(label, directory):
    print(f"\n{'=' * 55}")
    print(f"{label} DATASET")
    print(f"{'=' * 55}")

    images = sorted(directory.rglob("*.png"))
    print("Total images:", len(images))

    for image_path in images[:5]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(image_path.name, "FAILED TO READ")
            continue

        height, width, channels = image.shape

        print(
            f"{image_path.name}: "
            f"Width={width}, Height={height}, "
            f"Channels={channels}, "
            f"Dtype={image.dtype}"
        )


def main():
    for label, directory in DATASETS.items():
        if not directory.exists():
            print(f"Directory missing: {directory}")
            continue

        inspect_images(label, directory)


if __name__ == "__main__":
    main()
