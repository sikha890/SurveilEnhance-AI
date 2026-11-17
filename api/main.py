from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from api.services.sr_service import service

MAX_UPLOAD_BYTES = 15 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="Surveillance Super-Resolution API",
    version="1.0.0",
    description="SRCNN image enhancement inference API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "Surveillance Super-Resolution API",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": service.ready,
        "device": "cpu",
    }


@app.get("/api/model")
def model_info():
    if not service.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "name": "SRCNN",
        "architecture": "9x9 -> 1x1 -> 5x5",
        "scale_factor": 4,
        "device": "cpu",
        "checkpoint": service.checkpoint_info,
    }


@app.post("/api/enhance")
async def enhance_image(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/bmp",
        "image/tiff",
    }:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type. Upload PNG, JPEG, WebP, BMP or TIFF.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Maximum upload size is 15 MB")

    try:
        image = Image.open(BytesIO(content))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid or unsafe image")

    try:
        buffer, metadata = service.enhance(image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    headers = {
        "X-Inference-Seconds": str(metadata["inference_seconds"]),
        "X-Input-Width": str(metadata["input_width"]),
        "X-Input-Height": str(metadata["input_height"]),
        "X-Output-Width": str(metadata["output_width"]),
        "X-Output-Height": str(metadata["output_height"]),
        "X-Scale-Factor": str(metadata["scale_factor"]),
        "X-Model-Name": "SRCNN",
        "Content-Disposition": 'attachment; filename="srcnn_enhanced.png"',
    }

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers=headers,
    )
