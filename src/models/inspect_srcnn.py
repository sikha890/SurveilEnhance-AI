
"""
Inspect SRCNN architecture, parameters and tensor dimensions.
"""

import torch

from src.models.srcnn import SRCNN


def inspect_model():

    # Use CPU for this project
    device = torch.device("cpu")

    # Initialize model
    model = SRCNN().to(device)
    model.eval()

    print("=" * 60)
    print("SRCNN MODEL INSPECTION")
    print("=" * 60)

    # Display architecture
    print("\n1. MODEL ARCHITECTURE")
    print(model)

    # Count parameters
    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    parameter_memory = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    print("\n2. PARAMETER INFORMATION")
    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)
    print("Parameter memory:", parameter_memory, "bytes")
    print(
        "Parameter memory:",
        round(parameter_memory / 1024, 2),
        "KB"
    )

    # Test forward pass
    print("\n3. TENSOR DIMENSION ANALYSIS")

    input_tensor = torch.randn(
        1, 3, 256, 256,
        device=device
    )

    print("Input shape:", tuple(input_tensor.shape))

    with torch.inference_mode():

        features = model.feature_extraction(input_tensor)
        print("After feature extraction:", tuple(features.shape))

        mapped = model.nonlinear_mapping(features)
        print("After nonlinear mapping:", tuple(mapped.shape))

        output = model.reconstruction(mapped)
        print("After reconstruction:", tuple(output.shape))

    print("\n4. EXECUTION INFORMATION")
    print("Device:", device)
    print("Model training mode:", model.training)
    print("Output data type:", output.dtype)

    print("\n" + "=" * 60)
    print("SRCNN INSPECTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    inspect_model()
