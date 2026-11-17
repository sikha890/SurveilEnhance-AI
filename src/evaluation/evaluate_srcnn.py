
"""
Evaluate trained SRCNN against bicubic interpolation.

Uses the 100 deterministic DIV2K validation patches.
Reports average PSNR and SSIM for both methods.
"""

from pathlib import Path

import numpy as np
import torch

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from src.models.srcnn import SRCNN
from src.training.train_srcnn import (
    load_config,
    get_device,
    create_dataset,
    create_dataloader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def calculate_metrics(prediction, target):

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

    return psnr, ssim


def main():

    config = load_config()
    device = get_device(config)

    print("=" * 65)
    print("SRCNN VS BICUBIC EVALUATION")
    print("=" * 65)

    # Load validation dataset.
    data = config["data"]

    validation_dataset = create_dataset(
        data["validation_hr_directory"],
        data["validation_lr_directory"],
        config,
        training=False,
    )

    validation_loader = create_dataloader(
        validation_dataset,
        config,
        training=False,
    )

    # Load trained model.
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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    crop = data["input_crop"]

    srcnn_psnr = []
    srcnn_ssim = []

    bicubic_psnr = []
    bicubic_ssim = []

    # Evaluate both methods on identical patches.
    with torch.inference_mode():

        for lr_images, hr_images in validation_loader:

            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            predictions = model(lr_images)

            # Clip predictions for image-quality evaluation.
            predictions = torch.clamp(
                predictions,
                min=0.0,
                max=1.0,
            )

            # Crop bicubic input to match SRCNN output.
            bicubic = lr_images[
                :,
                :,
                crop:-crop,
                crop:-crop,
            ]

            for index in range(hr_images.size(0)):

                target = (
                    hr_images[index]
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )

                srcnn_image = (
                    predictions[index]
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )

                bicubic_image = (
                    bicubic[index]
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )

                p, s = calculate_metrics(
                    srcnn_image,
                    target,
                )

                srcnn_psnr.append(p)
                srcnn_ssim.append(s)

                p, s = calculate_metrics(
                    bicubic_image,
                    target,
                )

                bicubic_psnr.append(p)
                bicubic_ssim.append(s)

    print("\nEVALUATION SUMMARY")
    print("-" * 65)

    print("Validation images:", len(validation_dataset))

    print("\nBICUBIC BASELINE")
    print("Average PSNR:", np.mean(bicubic_psnr))
    print("Average SSIM:", np.mean(bicubic_ssim))

    print("\nTRAINED SRCNN")
    print("Average PSNR:", np.mean(srcnn_psnr))
    print("Average SSIM:", np.mean(srcnn_ssim))

    print("\nSRCNN IMPROVEMENT")
    print(
        "PSNR difference:",
        np.mean(srcnn_psnr) - np.mean(bicubic_psnr),
        "dB",
    )

    print(
        "SSIM difference:",
        np.mean(srcnn_ssim) - np.mean(bicubic_ssim),
    )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
