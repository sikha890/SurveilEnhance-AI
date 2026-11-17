
"""
Full-image tiled evaluation of trained SRCNN against bicubic interpolation.

Evaluates all DIV2K validation images without extracting center patches.
Uses overlapping tiles to handle SRCNN's valid-convolution boundaries.
"""

from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from src.models.srcnn import SRCNN
from src.training.train_srcnn import load_config, get_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_rgb_image(path):
    """Read an image with OpenCV and convert BGR to RGB."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_tensor(image, device):
    """Convert an RGB uint8 image into a normalized NCHW tensor."""

    tensor = (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    )

    return tensor.to(device)


def calculate_metrics(prediction, target):
    """Calculate PSNR and SSIM for two RGB images."""

    prediction = np.clip(prediction, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    psnr = peak_signal_noise_ratio(
        target,
        prediction,
        data_range=1.0,
    )

    ssim = structural_similarity(
        target,
        prediction,
        channel_axis=-1,
        data_range=1.0,
    )

    return float(psnr), float(ssim)


def tiled_srcnn_inference(
    model,
    image,
    device,
    tile_size=256,
    overlap=32,
):
    """
    Run SRCNN on a complete bicubic-upsampled image.

    Each tile has a 6-pixel context margin on every side.
    Only the valid central prediction is retained.

    Overlapping input tiles provide context near internal
    tile boundaries. The retained output regions do not overlap.
    """

    if tile_size <= 12:
        raise ValueError("Tile size must be greater than 12.")

    if overlap < 12:
        raise ValueError("Overlap must be at least 12 pixels.")

    if overlap >= tile_size:
        raise ValueError("Overlap must be smaller than tile size.")

    height, width = image.shape[:2]

    if height < 13 or width < 13:
        raise ValueError("Image dimensions must be at least 13 pixels.")

    crop = 6
    stride = tile_size - overlap

    image_tensor = image_to_tensor(image, device)

    # SRCNN predicts H-12 x W-12, so the output covers
    # the reference image cropped by 6 pixels on each edge.
    output_height = height - 2 * crop
    output_width = width - 2 * crop

    output = torch.empty(
        (1, 3, output_height, output_width),
        dtype=torch.float32,
        device="cpu",
    )

    # Generate tile starts, including the final image boundary.
    def tile_starts(length):
        if length <= tile_size:
            return [0]

        starts = list(range(0, length - tile_size + 1, stride))

        final_start = length - tile_size

        if starts[-1] != final_start:
            starts.append(final_start)

        return starts

    y_starts = tile_starts(height)
    x_starts = tile_starts(width)

    for y in y_starts:
        for x in x_starts:

            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)

            tile = image_tensor[
                :,
                :,
                y:y_end,
                x:x_end,
            ]

            prediction = model(tile)
            prediction = torch.clamp(prediction, 0.0, 1.0)

            # Valid SRCNN output coordinates in the full image.
            # The model output starts crop pixels inside each input tile.
            valid_y_start = max(y + crop, crop)
            valid_x_start = max(x + crop, crop)

            valid_y_end = min(y_end - crop, height - crop)
            valid_x_end = min(x_end - crop, width - crop)

            if (
                valid_y_start >= valid_y_end
                or valid_x_start >= valid_x_end
            ):
                continue

            # Corresponding coordinates inside the tile prediction.
            # SRCNN's valid output begins crop pixels inside the input tile.
            local_y_start = valid_y_start - y - crop
            local_x_start = valid_x_start - x - crop

            local_y_end = valid_y_end - y - crop
            local_x_end = valid_x_end - x - crop

            # Corresponding coordinates in the final output.
            output_y_start = valid_y_start - crop
            output_x_start = valid_x_start - crop

            output_y_end = valid_y_end - crop
            output_x_end = valid_x_end - crop

            output[
                :,
                :,
                output_y_start:output_y_end,
                output_x_start:output_x_end,
            ] = prediction[
                :,
                :,
                local_y_start:local_y_end,
                local_x_start:local_x_end,
            ].cpu()

    return (
        output.squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )


def main():

    config = load_config()
    device = get_device(config)

    print("=" * 70)
    print("FULL-IMAGE SRCNN VS BICUBIC EVALUATION")
    print("=" * 70)

    data_config = config["data"]

    hr_directory = PROJECT_ROOT / data_config["validation_hr_directory"]
    lr_directory = PROJECT_ROOT / data_config["validation_lr_directory"]

    if not hr_directory.is_dir() or not lr_directory.is_dir():
        raise FileNotFoundError("DIV2K validation directories not found.")

    hr_files = sorted(hr_directory.glob("*.png"))
    lr_files = sorted(lr_directory.glob("*.png"))

    hr_map = {path.name: path for path in hr_files}
    lr_map = {path.name: path for path in lr_files}

    if not hr_map:
        raise ValueError("No validation HR images found.")

    if set(hr_map) != set(lr_map):
        raise ValueError("HR and LR validation filenames do not match.")

    checkpoint_path = (
        PROJECT_ROOT
        / config["checkpoint"]["directory"]
        / "best_srcnn.pth"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = SRCNN().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Device: {device}")
    print(f"Validation images: {len(hr_map)}")
    print(f"Checkpoint: {checkpoint_path.name}")

    bicubic_psnr = []
    bicubic_ssim = []

    srcnn_psnr = []
    srcnn_ssim = []

    inference_times = []

    tile_size = 256
    overlap = 32
    scale_factor = data_config["scale_factor"]

    with torch.inference_mode():

        for index, filename in enumerate(sorted(hr_map), start=1):

            hr_image = load_rgb_image(hr_map[filename])
            lr_image = load_rgb_image(lr_map[filename])

            hr_height, hr_width = hr_image.shape[:2]

            expected_lr_shape = (
                hr_height // scale_factor,
                hr_width // scale_factor,
            )

            if (
                hr_height % scale_factor != 0
                or hr_width % scale_factor != 0
                or lr_image.shape[:2] != expected_lr_shape
            ):
                raise ValueError(
                    f"Invalid HR-LR dimensions: {filename}"
                )

            # Bicubic-upsample the complete LR image using the same
            # PyTorch interpolation settings used by the training dataset.
            lr_tensor = image_to_tensor(lr_image, device)

            bicubic_tensor = F.interpolate(
                lr_tensor,
                size=(hr_height, hr_width),
                mode="bicubic",
                align_corners=False,
            )

            bicubic_tensor = torch.clamp(
                bicubic_tensor,
                min=0.0,
                max=1.0,
            )

            bicubic_image = (
                bicubic_tensor.squeeze(0)
                .permute(1, 2, 0)
                .mul(255.0)
                .round()
                .byte()
                .cpu()
                .numpy()
            )

            # Measure SRCNN inference time only.
            start_time = time.perf_counter()

            prediction = tiled_srcnn_inference(
                model,
                bicubic_image,
                device,
                tile_size=tile_size,
                overlap=overlap,
            )

            inference_time = time.perf_counter() - start_time
            inference_times.append(inference_time)

            # Valid-convolution SRCNN output excludes 6 pixels
            # from every edge. Crop both references identically.
            target = hr_image[
                6:-6,
                6:-6,
            ].astype(np.float32) / 255.0

            bicubic_reference = bicubic_image[
                6:-6,
                6:-6,
            ].astype(np.float32) / 255.0

            srcnn_reference = prediction.astype(np.float32)

            if (
                target.shape != bicubic_reference.shape
                or target.shape != srcnn_reference.shape
            ):
                raise ValueError(
                    f"Output alignment mismatch: {filename}; "
                    f"target={target.shape}, "
                    f"bicubic={bicubic_reference.shape}, "
                    f"SRCNN={srcnn_reference.shape}"
                )

            p, s = calculate_metrics(
                bicubic_reference,
                target,
            )

            bicubic_psnr.append(p)
            bicubic_ssim.append(s)

            p, s = calculate_metrics(
                srcnn_reference,
                target,
            )

            srcnn_psnr.append(p)
            srcnn_ssim.append(s)

            print(
                f"[{index:03d}/{len(hr_map):03d}] "
                f"{filename} | "
                f"Bicubic PSNR: {bicubic_psnr[-1]:.4f} | "
                f"SRCNN PSNR: {srcnn_psnr[-1]:.4f} | "
                f"Time: {inference_time:.3f}s"
            )

    print("\n" + "=" * 70)
    print("FULL-IMAGE EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Images evaluated: {len(hr_map)}")

    print("\nBICUBIC BASELINE")
    print(f"Average PSNR: {np.mean(bicubic_psnr):.6f} dB")
    print(f"Average SSIM: {np.mean(bicubic_ssim):.6f}")

    print("\nTRAINED SRCNN")
    print(f"Average PSNR: {np.mean(srcnn_psnr):.6f} dB")
    print(f"Average SSIM: {np.mean(srcnn_ssim):.6f}")

    print("\nSRCNN IMPROVEMENT")
    print(
        f"PSNR difference: "
        f"{np.mean(srcnn_psnr) - np.mean(bicubic_psnr):+.6f} dB"
    )

    print(
        f"SSIM difference: "
        f"{np.mean(srcnn_ssim) - np.mean(bicubic_ssim):+.6f}"
    )

    print("\nINFERENCE PERFORMANCE")
    print(f"Total inference time: {np.sum(inference_times):.3f}s")
    print(f"Average per image: {np.mean(inference_times):.3f}s")

    print("\nFull-image evaluation completed.")


if __name__ == "__main__":
    main()
