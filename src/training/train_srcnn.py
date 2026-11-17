
"""
SRCNN Training Engine.

Features:
- YAML configuration
- Reproducible training
- CPU/GPU device selection
- Separate DIV2K training and validation datasets
- Patch-based data loading
- Adam optimizer
- MSE loss
- Best validation checkpointing
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader

from src.data.srcnn_patch_dataset import SRCNNPatchDataset
from src.losses.srcnn_loss import SRCNNLoss
from src.models.srcnn import SRCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "srcnn.yaml"



def load_config(config_path=None):

    if config_path is None:
        config_path = CONFIG_PATH

    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):

    use_cuda = config["device"]["use_cuda"]

    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_dataset(hr_dir, lr_dir, config, training):

    data_config = config["data"]

    return SRCNNPatchDataset(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        patch_size=data_config["patch_size"],
        patches_per_image=data_config["patches_per_image"],
        scale_factor=data_config["scale_factor"],
        input_crop=data_config["input_crop"],
        training=training,
    )


def create_dataloader(dataset, config, training):

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=training,
        num_workers=config["training"]["num_workers"],
        pin_memory=False,
        drop_last=False,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    total_loss = 0.0
    total_samples = 0

    for lr_images, hr_images in loader:

        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(lr_images)

        loss = criterion(predictions, hr_images)

        loss.backward()
        optimizer.step()

        batch_size = lr_images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training DataLoader is empty.")

    return total_loss / total_samples


def validate(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():

        for lr_images, hr_images in loader:

            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            predictions = model(lr_images)

            loss = criterion(predictions, hr_images)

            batch_size = lr_images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Validation DataLoader is empty.")

    return total_loss / total_samples


def main():

    parser = argparse.ArgumentParser(
        description="Train SRCNN using a selected YAML configuration."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/srcnn.yaml",
        help="Path to the SRCNN YAML configuration.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config["experiment"]["seed"])

    device = get_device(config)

    print("=" * 65)
    print("SRCNN DIV2K PATCH-BASED TRAINING")
    print("=" * 65)

    print("Experiment:", config["experiment"]["name"])
    print("Device:", device)

    # Resolve separate training and validation directories.
    data_config = config["data"]

    train_hr_dir = PROJECT_ROOT / data_config["train_hr_directory"]
    train_lr_dir = PROJECT_ROOT / data_config["train_lr_directory"]

    validation_hr_dir = (
        PROJECT_ROOT / data_config["validation_hr_directory"]
    )

    validation_lr_dir = (
        PROJECT_ROOT / data_config["validation_lr_directory"]
    )

    # Create separate datasets.
    train_dataset = create_dataset(
        train_hr_dir,
        train_lr_dir,
        config,
        training=True,
    )

    validation_dataset = create_dataset(
        validation_hr_dir,
        validation_lr_dir,
        config,
        training=False,
    )

    print("\nDATASET INFORMATION")
    print("-" * 65)

    print("Training images:", len(train_dataset.filenames))
    print("Training patches per image:",
          data_config["patches_per_image"])
    print("Total training patches:", len(train_dataset))

    print("Validation images:", len(validation_dataset.filenames))
    print("Validation patches:", len(validation_dataset))

    # Create DataLoaders.
    train_loader = create_dataloader(
        train_dataset,
        config,
        training=True,
    )

    validation_loader = create_dataloader(
        validation_dataset,
        config,
        training=False,
    )

    print("Training batches per epoch:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    # Initialize model.
    model = SRCNN().to(device)

    # Initialize loss.
    criterion = SRCNNLoss()

    # Initialize optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Prepare checkpoint directory.
    checkpoint_dir = (
        PROJECT_ROOT / config["checkpoint"]["directory"]
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")

    print("\nTRAINING STARTED")
    print("=" * 65)

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss = validate(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.8f} "
            f"Validation Loss: {validation_loss:.8f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            checkpoint_path = (
                checkpoint_dir / "best_srcnn.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "config": config,
                },
                checkpoint_path,
            )

            print("Best checkpoint saved:", checkpoint_path)

    print("\n" + "=" * 65)
    print("SRCNN TRAINING COMPLETED")
    print("Best validation loss:", best_validation_loss)
    print("=" * 65)


if __name__ == "__main__":
    main()
