"""Train and evaluate the malaria-cell CNN using patient-grouped partitions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import CellImageDataset, INDEX_TO_CLASS
from .model import MalariaCNN, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_manifest(path: Path, root: Path) -> list[dict[str, str | int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared manifest not found: {path}; run scripts/prepare_data.py first")
    rows: list[dict[str, str | int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["label"] = int(row["label"])
            filepath = Path(row["filepath"])
            row["filepath"] = str(filepath if filepath.is_absolute() else (root / filepath).resolve())
            rows.append(row)
    return rows


def _loader(rows, image_size: int, batch_size: int, workers: int, training: bool, seed: int) -> DataLoader:
    dataset = CellImageDataset(rows, image_size=image_size, augment=training)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def _train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_count = labels.size(0)
        running_loss += float(loss.detach().item()) * batch_count
        seen += batch_count
    return running_loss / max(1, seen)


def _evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    seen = 0
    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_count = labels.size(0)
            running_loss += float(loss.item()) * batch_count
            seen += batch_count
            y_true.append(labels.cpu().numpy())
            y_score.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    truth = np.concatenate(y_true).astype(np.int64)
    scores = np.concatenate(y_score).astype(np.float64)
    predictions = (scores >= 0.5).astype(np.int64)
    return running_loss / max(1, seen), truth, scores, predictions


def calculate_metrics(truth: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> tuple[dict, np.ndarray, np.ndarray]:
    predictions = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    auc = float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else None
    mcc = float(matthews_corrcoef(truth, predictions)) if len(np.unique(truth)) == 2 else None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "sensitivity_parasitized": float(sensitivity),
        "specificity_uninfected": float(specificity),
        "precision_parasitized": float(precision),
        "f1_parasitized": float(f1),
        "roc_auc": auc,
        "matthews_correlation_coefficient": mcc,
        "confusion_matrix": {"true_uninfected": tn, "false_parasitized": fp, "missed_parasitized": fn, "true_parasitized": tp},
        "class_counts": {"uninfected": int((truth == 0).sum()), "parasitized": int((truth == 1).sum())},
    }, matrix, predictions


def _write_plots(history: list[dict], matrix: np.ndarray, output: Path) -> None:
    if history:
        fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=150)
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Training loss", color="#087e8b")
        ax.plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation loss", color="#d1495b")
        ax.set(title="Training and validation loss", xlabel="Epoch", ylabel="Weighted cross-entropy")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / "training_history.png", bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title="Patient-held-out test confusion matrix", xlabel="Predicted class", ylabel="Reference class")
    ax.set_xticks([0, 1], ["Uninfected", "Parasitized"])
    ax.set_yticks([0, 1], ["Uninfected", "Parasitized"])
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, f"{int(value):,}", ha="center", va="center", color="#132238", fontsize=11)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2], help="Project root")
    parser.add_argument("--config", type=Path, default=None, help="JSON configuration file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional reproducible subset for smoke tests; 0 uses all training cells")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="Optional reproducible subset for smoke tests; 0 uses all evaluation cells")
    parser.add_argument("--device", default=None, help="Override automatic cpu/cuda selection")
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = args.config or root / "configs" / "default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in (("epochs", args.epochs), ("batch_size", args.batch_size), ("learning_rate", args.learning_rate)):
        if value is not None:
            config[key] = value
    seed = int(config["seed"])
    set_seed(seed)
    cpu_threads = max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(cpu_threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest_path = root / "data" / "processed" / "manifest.csv"
    rows = read_manifest(manifest_path, root)
    splits = {name: [row for row in rows if row["split"] == name] for name in ("train", "validation", "test")}
    if any(not splits[name] for name in splits):
        raise ValueError("The prepared manifest must contain non-empty train, validation, and test splits")
    if args.max_train_samples > 0 and len(splits["train"]) > args.max_train_samples:
        splits["train"] = random.Random(seed).sample(splits["train"], args.max_train_samples)
    if args.max_eval_samples > 0:
        for name in ("validation", "test"):
            if len(splits[name]) > args.max_eval_samples:
                splits[name] = random.Random(seed + (1 if name == "validation" else 2)).sample(splits[name], args.max_eval_samples)
    workers = max(0, min(int(config.get("num_workers", 2)), 4))
    train_loader = _loader(splits["train"], int(config["image_size"]), int(config["batch_size"]), workers, True, seed)
    validation_loader = _loader(splits["validation"], int(config["image_size"]), int(config["batch_size"]), workers, False, seed + 1)
    test_loader = _loader(splits["test"], int(config["image_size"]), int(config["batch_size"]), workers, False, seed + 2)

    train_labels = np.array([int(row["label"]) for row in splits["train"]])
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = len(train_labels) / (2.0 * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    model = MalariaCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))

    output = root / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best_model.pt"
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    start = time.time()
    print(f"Device: {device}; threads={torch.get_num_threads()}; model parameters={count_parameters(model):,}", flush=True)
    print(f"Training/validation/test images: {len(splits['train']):,}/{len(splits['validation']):,}/{len(splits['test']):,}", flush=True)

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.time()
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, _, _, _ = _evaluate(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "elapsed_seconds": round(time.time() - epoch_start, 2)})
        print(f"Epoch {epoch}/{config['epochs']} - train_loss={train_loss:.5f} val_loss={validation_loss:.5f} elapsed={history[-1]['elapsed_seconds']:.1f}s", flush=True)
        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            payload = {
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "image_size": int(config["image_size"]),
                "dropout": 0.30,
                "class_names": [INDEX_TO_CLASS[0], INDEX_TO_CLASS[1]],
            }
            torch.save(payload, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                print(f"Early stopping after {epoch} epochs; best validation epoch was {best_epoch}.", flush=True)
                break

    if not best_path.exists():
        raise RuntimeError("Training ended without producing a checkpoint")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, truth, scores, _ = _evaluate(model, test_loader, criterion, device)
    metrics, matrix, predictions = calculate_metrics(truth, scores, float(config["threshold"]))
    metrics.update({
        "test_loss": float(test_loss),
        "best_epoch": int(best_epoch),
        "patient_grouped_holdout": True,
        "test_patient_count": len({str(row["patient_id"]) for row in splits["test"]}),
        "validation_patient_count": len({str(row["patient_id"]) for row in splits["validation"]}),
        "test_image_count": len(splits["test"]),
        "device": str(device),
        "seed": seed,
    })
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(history)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "true_label", "predicted_label", "parasitized_score"])
        writer.writeheader()
        for row, actual, score, predicted in zip(splits["test"], truth, scores, predictions):
            writer.writerow({
                "filename": row.get("filename", Path(str(row["filepath"])).name),
                "true_label": INDEX_TO_CLASS[int(actual)],
                "predicted_label": INDEX_TO_CLASS[int(predicted)],
                "parasitized_score": f"{float(score):.8f}",
            })
    _write_plots(history, matrix, output)
    split_summary = {}
    for split_name, split_rows in splits.items():
        split_summary[split_name] = {
            "images": len(split_rows),
            "patients": len({str(row["patient_id"]) for row in split_rows}),
            "uninfected_cells": sum(int(row["label"]) == 0 for row in split_rows),
            "parasitized_cells": sum(int(row["label"]) == 1 for row in split_rows),
        }
    metadata = {
        "project": "Malaria Parasite Detection from Blood Smear Images Using a Convolutional Neural Network",
        "architecture": "custom three-block CNN with adaptive average pooling",
        "parameters": count_parameters(model),
        "image_size": int(config["image_size"]),
        "class_order": [INDEX_TO_CLASS[0], INDEX_TO_CLASS[1]],
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_config": config,
        "patient_grouped_split": True,
        "split_summary": split_summary,
        "test_metrics_file": "test_metrics.json",
        "elapsed_training_seconds": round(time.time() - start, 2),
        "clinical_use": "research/education only; not validated for clinical use",
    }
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_plots(history, matrix, output)
    print("\nHeld-out test metrics (0.5 threshold):", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"\nSaved model and evaluation artifacts under {output}", flush=True)


if __name__ == "__main__":
    main()
