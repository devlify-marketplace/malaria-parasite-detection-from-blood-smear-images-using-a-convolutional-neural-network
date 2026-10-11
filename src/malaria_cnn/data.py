"""Dataset parsing and patient-grouped splitting utilities."""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

CLASS_TO_INDEX = {"Uninfected": 0, "Parasitized": 1}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_mapping_file(path: str | Path, label_name: str) -> list[dict[str, str | int]]:
    """Parse an NIH/NLM patient mapping file (patient ID + Python-list cell names).

    The NLM files are CSV-like, with one patient per line and a Python list of
    filenames followed by empty comma-padded columns. ``ast.literal_eval`` is
    used rather than executing source text.
    """
    if label_name not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown label: {label_name}")
    path = Path(path)
    rows: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip(" ,\t\r\n"):
            continue
        if "," not in line:
            raise ValueError(f"Malformed mapping line {line_number} in {path}")
        patient_id, payload = line.split(",", 1)
        patient_id = patient_id.strip().strip("\"'")
        payload = payload.rstrip(", \t\r\n").strip()
        try:
            filenames = ast.literal_eval(payload)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid filename list on line {line_number} in {path}") from exc
        if not patient_id or not isinstance(filenames, (list, tuple)):
            raise ValueError(f"Invalid patient mapping on line {line_number} in {path}")
        for filename in filenames:
            name = Path(str(filename).replace("\\", "/")).name
            key = f"{label_name}/{name}"
            if not name or key in seen:
                raise ValueError(f"Empty or duplicate mapped filename: {key}")
            seen.add(key)
            rows.append({
                "filename": name,
                "patient_id": patient_id,
                "label": CLASS_TO_INDEX[label_name],
                "label_name": label_name,
            })
    return rows


def _image_index(image_root: str | Path) -> dict[tuple[int, str], Path]:
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    class_names = {name.casefold(): name for name in CLASS_TO_INDEX}
    index: dict[tuple[int, str], Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parent_class = class_names.get(path.parent.name.casefold())
        if parent_class is None:
            continue
        label = CLASS_TO_INDEX[parent_class]
        key = (label, path.name)
        if key in index:
            raise ValueError(f"Duplicate image filename within class: {path.name}")
        index[key] = path
    if not index:
        raise FileNotFoundError(
            f"No Parasitized/Uninfected images found below {root}; run scripts/download_dataset.py"
        )
    return index


def build_records(
    image_root: str | Path,
    parasitized_mapping: str | Path,
    uninfected_mapping: str | Path,
) -> list[dict[str, str | int]]:
    """Join the official mappings to image files and verify label consistency."""
    index = _image_index(image_root)
    mappings = (
        parse_mapping_file(parasitized_mapping, "Parasitized")
        + parse_mapping_file(uninfected_mapping, "Uninfected")
    )
    records: list[dict[str, str | int]] = []
    missing: list[str] = []
    for record in mappings:
        key = (int(record["label"]), str(record["filename"]))
        image_path = index.get(key)
        if image_path is None:
            missing.append(f"{record['label_name']}/{record['filename']}")
            continue
        records.append({**record, "filepath": str(image_path.resolve())})
    if missing:
        example = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} mapped images are missing (e.g. {example})")
    if not records:
        raise ValueError("No mapped image records were found")
    ids = {(int(row["label"]), str(row["filename"])) for row in records}
    if len(ids) != len(records):
        raise ValueError("Duplicate patient-map rows were found")
    return records


def _allocate_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [n * ratio for ratio in ratios]
    sizes = [int(value) for value in raw]
    remainder = n - sum(sizes)
    order = sorted(range(3), key=lambda i: (raw[i] - sizes[i], -i), reverse=True)
    for i in order[:remainder]:
        sizes[i] += 1
    return tuple(sizes)  # type: ignore[return-value]


def assign_patient_splits(
    records: Iterable[dict[str, str | int]],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> list[dict[str, str | int]]:
    """Assign entire patients to train/validation/test, stratified by infection status.

    All cells from a patient, including that patient's parasitized and
    uninfected cells, stay in one partition. This reduces within-patient
    leakage compared with a random image-level split.
    """
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must be three positive values summing to 1")
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("records must not be empty")
    patient_status: dict[str, bool] = {}
    for row in rows:
        patient_id = str(row["patient_id"])
        patient_status[patient_id] = patient_status.get(patient_id, False) or int(row["label"]) == 1
    strata: dict[bool, list[str]] = {False: [], True: []}
    for patient_id, has_parasite in patient_status.items():
        strata[has_parasite].append(patient_id)
    patient_to_split: dict[str, str] = {}
    split_names = ("train", "validation", "test")
    for stratum_index, status in enumerate((False, True)):
        patients = sorted(strata[status])
        if len(patients) < 3:
            raise ValueError("Need at least three patient groups in each infection-status stratum")
        random.Random(seed + stratum_index).shuffle(patients)
        n_train, n_validation, n_test = _allocate_sizes(len(patients), ratios)
        if min(n_train, n_validation, n_test) < 1:
            raise ValueError("Each infection-status stratum must contribute patients to every split")
        boundaries = (n_train, n_train + n_validation)
        chunks = (patients[:boundaries[0]], patients[boundaries[0]:boundaries[1]], patients[boundaries[1]:])
        for split, group in zip(split_names, chunks):
            patient_to_split.update({patient_id: split for patient_id in group})
    for row in rows:
        row["split"] = patient_to_split[str(row["patient_id"])]
    return rows


class CellImageDataset(Dataset):
    """Loads cell images as float32 RGB tensors in the [0, 1] range."""

    def __init__(self, rows: list[dict[str, str | int]], image_size: int = 96, augment: bool = False):
        if image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        self.rows = rows
        self.image_size = int(image_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with Image.open(str(row["filepath"])) as source:
            image = source.convert("RGB")
        image = ImageOps.fit(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR,
            centering=(0.5, 0.5),
        )
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.5:
                image = ImageOps.flip(image)
            image = image.rotate(random.choice((0, 90, 180, 270)))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tensor, label
