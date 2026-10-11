"""Local FastAPI service for the malaria-cell research demo."""

from __future__ import annotations

import os
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .inference import load_model, predict_image

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("MALARIA_MODEL_PATH", ROOT / "artifacts" / "best_model.pt"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
THRESHOLD = float(os.environ.get("MALARIA_THRESHOLD", "0.5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.image_size = 96
    app.state.model_ready = False
    if MODEL_PATH.is_file():
        try:
            app.state.model, app.state.image_size = load_model(MODEL_PATH, device="cpu")
            app.state.model_ready = True
        except Exception:
            app.state.model = None
            app.state.model_ready = False
    yield


app = FastAPI(
    title="Malaria Cell Screening Research API",
    version="1.0.0",
    description="Research and education demo for classifying an individual thin-smear cell image. Not for clinical diagnosis.",
    lifespan=lifespan,
)
ASSET_DIR = ROOT / "web" / "assets"
if ASSET_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="assets")


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ready" if app.state.model_ready else "model_missing",
        "model_ready": bool(app.state.model_ready),
        "service": "malaria-cell-screening-research-prototype",
    }


@app.get("/api/model-info")
async def model_info():
    return {
        "model_ready": bool(app.state.model_ready),
        "task": "single-cell binary image classification",
        "classes": ["Uninfected", "Parasitized"],
        "input_size": [app.state.image_size, app.state.image_size, 3],
        "default_threshold": THRESHOLD,
        "scope": "Research/education only; not validated for clinical use.",
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not app.state.model_ready or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Train the model first.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 5 MB upload limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(BytesIO(content))
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions are too large.")
            image.verify()
            image = Image.open(BytesIO(content)).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=415, detail="Upload a valid PNG, JPEG, or WebP image.") from exc
    result = predict_image(app.state.model, image, app.state.image_size, THRESHOLD, device="cpu")
    return {
        **result,
        "input_width": width,
        "input_height": height,
        "disclaimer": "Research/education output only. This cell-image score is not a patient diagnosis and must not guide treatment.",
        "upload_retained": False,
    }
