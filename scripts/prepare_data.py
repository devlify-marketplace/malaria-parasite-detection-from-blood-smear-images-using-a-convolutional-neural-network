#!/usr/bin/env python3
"""Join NIH/NLM cells to patient IDs and create reproducible group splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from malaria_cnn.data import assign_patient_splits, build_records, INDEX_TO_CLASS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--image-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    image_dir = args.image_dir or root / "data" / "images"
    raw_dir = root / "data" / "raw"
    records = build_records(
        image_dir,
        raw_dir / "patientid_cellmapping_parasitized.csv",
        raw_dir / "patientid_cellmapping_uninfected.csv",
    )
    records = assign_patient_splits(
        records,
        seed=args.seed,
        ratios=(args.train_fraction, args.validation_fraction, args.test_fraction),
    )
    output_dir = root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    fields = ["filepath", "filename", "patient_id", "label", "label_name", "split"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                **row,
                "filepath": Path(str(row["filepath"])).resolve().relative_to(root).as_posix(),
            })
    summary = {"seed": args.seed, "ratios": {"train": args.train_fraction, "validation": args.validation_fraction, "test": args.test_fraction}, "patient_grouped": True, "total_images": len(records), "splits": {}}
    for split in ("train", "validation", "test"):
        subset = [row for row in records if row["split"] == split]
        summary["splits"][split] = {
            "images": len(subset),
            "patients": len({str(row["patient_id"]) for row in subset}),
            "classes": {INDEX_TO_CLASS[label]: sum(int(row["label"]) == label for row in subset) for label in (0, 1)},
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records):,} rows to {manifest}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
