
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.srcnn_dataset import SRCNNDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HR_DIR = PROJECT_ROOT / "data" / "generated" / "HR"
LR_DIR = PROJECT_ROOT / "data" / "generated" / "LR"


def inspect_dataset():

    dataset = SRCNNDataset(
        hr_dir=HR_DIR,
        lr_dir=LR_DIR,
        scale_factor=4,
        input_crop=6,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("=" * 60)
    print("SRCNN DATASET INSPECTION")
    print("=" * 60)

    print("Dataset size:", len(dataset))

    for lr_batch, hr_batch in loader:

        print("LR batch shape:", tuple(lr_batch.shape))
        print("HR batch shape:", tuple(hr_batch.shape))

        print("LR minimum:", lr_batch.min().item())
        print("LR maximum:", lr_batch.max().item())

        print("HR minimum:", hr_batch.min().item())
        print("HR maximum:", hr_batch.max().item())

        break

    print("\nDataset inspection completed successfully!")


if __name__ == "__main__":
    inspect_dataset()