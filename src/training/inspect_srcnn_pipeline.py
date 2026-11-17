
"""
Inspect the SRCNN DIV2K patch-based training pipeline.

Loads one batch from each DataLoader.
Does not perform model training.
"""

import torch

from src.training.train_srcnn import (
    load_config,
    create_dataset,
    create_dataloader,
)


def main():

    config = load_config()

    data = config["data"]

    print("=" * 65)
    print("SRCNN DATA PIPELINE INSPECTION")
    print("=" * 65)

    # Training dataset
    train_dataset = create_dataset(
        data["train_hr_directory"],
        data["train_lr_directory"],
        config,
        training=True,
    )

    # Validation dataset
    validation_dataset = create_dataset(
        data["validation_hr_directory"],
        data["validation_lr_directory"],
        config,
        training=False,
    )

    # DataLoaders
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

    print("\nDATASET SUMMARY")
    print("-" * 65)

    print("Training images:", len(train_dataset.filenames))
    print("Training patches:", len(train_dataset))
    print("Validation images:", len(validation_dataset.filenames))
    print("Validation patches:", len(validation_dataset))

    print("\nDATALOADER SUMMARY")
    print("-" * 65)

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    # Inspect one training batch.
    train_lr, train_hr = next(iter(train_loader))

    # Inspect one validation batch.
    validation_lr, validation_hr = next(iter(validation_loader))

    print("\nTRAINING BATCH")
    print("-" * 65)

    print("LR shape:", tuple(train_lr.shape))
    print("HR shape:", tuple(train_hr.shape))
    print("LR dtype:", train_lr.dtype)
    print("HR dtype:", train_hr.dtype)
    print("LR range:", train_lr.min().item(), train_lr.max().item())
    print("HR range:", train_hr.min().item(), train_hr.max().item())

    print("\nVALIDATION BATCH")
    print("-" * 65)

    print("LR shape:", tuple(validation_lr.shape))
    print("HR shape:", tuple(validation_hr.shape))

    # Verify compatibility with SRCNN.
    assert train_lr.shape[1:] == (3, 128, 128)
    assert train_hr.shape[1:] == (3, 116, 116)

    assert validation_lr.shape[1:] == (3, 128, 128)
    assert validation_hr.shape[1:] == (3, 116, 116)

    assert torch.isfinite(train_lr).all()
    assert torch.isfinite(train_hr).all()

    assert torch.isfinite(validation_lr).all()
    assert torch.isfinite(validation_hr).all()

    print("\nRESULT: ALL PIPELINE CHECKS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
