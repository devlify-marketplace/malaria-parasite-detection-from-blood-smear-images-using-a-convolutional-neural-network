"""Model loading and single-image prediction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .data import INDEX_TO_CLASS
from .model import MalariaCNN


def preprocess_image(image: Image.Image, image_size: int = 96) -> torch.Tensor:
    """Resize and convert an image to the model's normalized NCHW tensor format."""
    image = ImageOps.fit(
        image.convert("RGB"),
        (int(image_size), int(image_size)),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()).unsqueeze(0)


def load_model(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[MalariaCNN, int]:
    """Load a trusted project checkpoint containing only tensors and primitives."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MalariaCNN(dropout=float(payload.get("dropout", 0.30)))
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, int(payload.get("image_size", 96))


def predict_image(
    model: MalariaCNN,
    image: Image.Image,
    image_size: int = 96,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
) -> dict[str, float | str]:
    """Return a screening class and uncalibrated softmax scores, not a diagnosis."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    model = model.to(device).eval()
    batch = preprocess_image(image, image_size).to(device)
    with torch.inference_mode():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    parasitized_score = float(probabilities[1])
    prediction = "Parasitized" if parasitized_score >= threshold else "Uninfected"
    confidence = parasitized_score if prediction == "Parasitized" else float(probabilities[0])
    return {
        "prediction": prediction,
        "confidence": confidence,
        "parasitized_score": parasitized_score,
        "uninfected_score": float(probabilities[0]),
        "threshold": float(threshold),
        "class_scores_are_calibrated_probabilities": False,
    }
