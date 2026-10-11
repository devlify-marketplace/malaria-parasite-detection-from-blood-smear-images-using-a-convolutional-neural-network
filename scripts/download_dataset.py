#!/usr/bin/env python3
"""Download and safely extract the official NLM malaria cell dataset."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "cell_images.zip": "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip",
    "patientid_cellmapping_parasitized.csv": "https://data.lhncbc.nlm.nih.gov/public/Malaria/patientid_cellmapping_parasitized.csv",
    "patientid_cellmapping_uninfected.csv": "https://data.lhncbc.nlm.nih.gov/public/Malaria/patientid_cellmapping_uninfected.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, force: bool = False) -> None:
    if target.exists() and target.stat().st_size > 0 and not force:
        print(f"Using existing {target} ({target.stat().st_size:,} bytes)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "malaria-smear-cnn/1.0 (research project)"})
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            last_report = -1
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                downloaded += len(block)
                percent = int(100 * downloaded / total) if total else 0
                if percent >= last_report + 10:
                    print(f"  {downloaded:,}/{total or '?'} bytes ({percent}%)", flush=True)
                    last_report = percent
        if total and downloaded != total:
            raise IOError(f"Incomplete download for {target}: got {downloaded}, expected {total}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Saved {target} ({target.stat().st_size:,} bytes; sha256={sha256(target)})")


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise IOError(f"Corrupt ZIP member: {bad_member}")
        for info in archive.infolist():
            candidate = (destination / info.filename).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError(f"Unsafe path in ZIP archive: {info.filename}")
        archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    parser.add_argument("--force", action="store_true", help="Redownload existing source files")
    parser.add_argument("--no-extract", action="store_true", help="Download sources without extracting images")
    args = parser.parse_args()
    raw = args.root / "data" / "raw"
    for filename, url in SOURCES.items():
        download(url, raw / filename, force=args.force)
    archive = raw / "cell_images.zip"
    if not args.no_extract:
        image_root = args.root / "data" / "images"
        existing = image_root.exists() and any(image_root.rglob("*.png"))
        if existing:
            print(f"Using existing extracted images under {image_root}")
        else:
            print(f"Extracting archive to {image_root}")
            safe_extract(archive, image_root)
        print("Download and extraction completed.")
    sums = raw / "SHA256SUMS.txt"
    sums.write_text("".join(f"{sha256(raw / filename)}  {filename}\n" for filename in SOURCES), encoding="utf-8")
    print(f"Wrote source checksums to {sums}")


if __name__ == "__main__":
    main()
