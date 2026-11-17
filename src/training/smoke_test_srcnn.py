
"""
Controlled SRCNN training smoke test.

Checks:
1. Forward propagation
2. MSE loss
3. Backpropagation
4. Adam optimizer update
5. Validation without gradients
6. Checkpoint save and reload

Uses only one training batch and one validation batch.
"""

import tempfile
from pathlib import Path

import torch

from src.models.srcnn import SRCNN
from src.losses.srcnn_loss import SRCNNLoss
from src.training.train_srcnn import (
    load_config,
    set_seed,
    get_device,
    create_dataset,
    create_dataloader,
    train_one_epoch,
    validate,
)


def main():

    config = load_config()

    set_seed(config["experiment"]["seed"])

    device = get_device(config)

    print("=" * 65)
    print("SRCNN CONTROLLED TRAINING SMOKE TEST")
    print("=" * 65)

    print("Device:", device)

    data = config["data"]

    # Create datasets.
    train_dataset = create_dataset(
        data["train_hr_directory"],
        data["train_lr_directory"],
        config,
        training=True,
    )

    validation_dataset = create_dataset(
        data["validation_hr_directory"],
        data["validation_lr_directory"],
        config,
        training=False,
    )

    # Create loaders.
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

    # Obtain only one batch from each loader.
    train_lr, train_hr = next(iter(train_loader))
    validation_lr, validation_hr = next(iter(validation_loader))

    train_loader = [(train_lr, train_hr)]
    validation_loader = [(validation_lr, validation_hr)]

    # Initialize model and optimizer.
    model = SRCNN().to(device)
    criterion = SRCNNLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # 1. Forward propagation.
    print("\n1. FORWARD PROPAGATION")

    model.eval()

    with torch.inference_mode():
        predictions = model(train_lr.to(device))

    print("Prediction shape:", tuple(predictions.shape))
    print("Target shape:", tuple(train_hr.shape))

    assert predictions.shape == train_hr.shape

    print("Forward propagation: PASSED")

    # 2. Record model parameters before training.
    original_parameter = (
        next(model.parameters()).detach().clone()
    )

    # 3. Training step.
    print("\n2. TRAINING STEP")

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    print("Training loss:", train_loss)

    assert torch.isfinite(torch.tensor(train_loss))
    assert train_loss >= 0

    updated_parameter = next(model.parameters()).detach()

    assert not torch.equal(
        original_parameter,
        updated_parameter,
    )

    print("Gradient and optimizer update: PASSED")

    # 4. Validation.
    print("\n3. VALIDATION STEP")

    validation_loss = validate(
        model,
        validation_loader,
        criterion,
        device,
    )

    print("Validation loss:", validation_loss)

    assert torch.isfinite(torch.tensor(validation_loss))
    assert validation_loss >= 0

    print("Validation: PASSED")

    # 5. Checkpoint save and reload.
    print("\n4. CHECKPOINT SAVE AND RELOAD")

    with tempfile.TemporaryDirectory() as temp_dir:

        checkpoint_path = (
            Path(temp_dir) / "smoke_test_srcnn.pth"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            },
            checkpoint_path,
        )

        assert checkpoint_path.is_file()

        loaded_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        restored_model = SRCNN().to(device)

        restored_model.load_state_dict(
            loaded_checkpoint["model_state_dict"]
        )

        print("Checkpoint saved and loaded successfully.")

    print("\n" + "=" * 65)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
