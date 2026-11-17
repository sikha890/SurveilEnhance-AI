from pathlib import Path
import io
import time

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from src.models.srcnn import SRCNN

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints" / "srcnn_5_epochs" / "best_srcnn.pth"

MAX_IMAGE_PIXELS = 25_000_000
TILE_SIZE = 256
HALO = 6
SCALE = 4


class SuperResolutionService:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = None
        self.checkpoint_info = {}

    def load(self):
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"SRCNN checkpoint not found: {CHECKPOINT}"
            )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint has no model_state_dict")

        model = SRCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.checkpoint_info = {
            "path": str(CHECKPOINT.relative_to(ROOT)),
            "epoch": checkpoint.get("epoch"),
            "training_loss": checkpoint.get("train_loss"),
            "validation_loss": checkpoint.get("validation_loss"),
            "parameters": sum(p.numel() for p in model.parameters()),
        }

    @property
    def ready(self):
        return self.model is not None

    @torch.inference_mode()
    def enhance(self, image: Image.Image):
        if not self.ready:
            raise RuntimeError("SRCNN model is not loaded")

        image = image.convert("RGB")
        original_width, original_height = image.size

        if original_width * original_height > MAX_IMAGE_PIXELS:
            raise ValueError("Image exceeds the 25-megapixel limit")

        lr = pil_to_tensor(image).float().unsqueeze(0) / 255.0

        upscaled = F.interpolate(
            lr,
            scale_factor=SCALE,
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)

        _, _, height, width = upscaled.shape

        # SRCNN uses valid convolutions and removes 6 pixels per edge.
        # Pad the full image, then process overlapping halo regions.
        padded = F.pad(
            upscaled,
            (HALO, HALO, HALO, HALO),
            mode="replicate",
        )

        output = torch.empty_like(upscaled)

        start = time.perf_counter()

        for top in range(0, height, TILE_SIZE):
            for left in range(0, width, TILE_SIZE):
                bottom = min(top + TILE_SIZE, height)
                right = min(left + TILE_SIZE, width)

                tile = padded[
                    :,
                    :,
                    top:bottom + 2 * HALO,
                    left:right + 2 * HALO,
                ]

                prediction = self.model(tile)

                expected_h = bottom - top
                expected_w = right - left

                if prediction.shape[-2:] != (expected_h, expected_w):
                    raise RuntimeError("Unexpected SRCNN tile dimensions")

                output[:, :, top:bottom, left:right] = prediction

        elapsed = time.perf_counter() - start

        output = output.clamp(0.0, 1.0).squeeze(0).cpu()
        result = to_pil_image(output)

        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        buffer.seek(0)

        return buffer, {
            "inference_seconds": round(elapsed, 4),
            "input_width": original_width,
            "input_height": original_height,
            "output_width": result.width,
            "output_height": result.height,
            "scale_factor": SCALE,
            "device": str(self.device),
        }


service = SuperResolutionService()
