# Malaria Parasite Detection from Blood Smear Images Using a Convolutional Neural Network

**Topic 6** from the supplied final-year project list. This is a complete, runnable research prototype for classifying **individual segmented thin-smear cell images** as *Parasitized* or *Uninfected*. It includes an NIH/NLM dataset downloader, patient-grouped data preparation, a custom CNN, training and held-out evaluation, a saved model, a FastAPI service, a browser upload demo, tests, an academic report and provenance/model-card documentation.

> **Research-only scope:** this model classifies a cropped cell image. It does not examine a full slide, count parasites, estimate patient parasitaemia, identify all malaria species, or diagnose a person. It is not validated for clinical use and must not guide treatment. A trained microscopist and approved diagnostic workflow remain essential.

## Devlify

An academic project in the [Devlify GitHub collection](https://github.com/devlify-marketplace).

[Visit Devlify](https://devlify.com.ng) · [Browse the marketplace](https://devlify.com.ng/marketplace)

## Included

- `scripts/download_dataset.py` — downloads the NIH/NLM single-cell archive and its patient-to-cell mapping files, checks ZIP integrity, extracts safely, and records SHA-256 checksums.
- `scripts/prepare_data.py` and `src/malaria_cnn/data.py` — joins each cell to the official patient mapping and creates deterministic **patient-grouped** 70/15/15 train/validation/test partitions.
- `src/malaria_cnn/model.py` — compact three-block CNN with batch normalization, dropout and global average pooling.
- `src/malaria_cnn/train.py` — reproducible training, best-checkpoint selection by validation loss, and held-out metrics (sensitivity, specificity, F1, ROC-AUC, MCC, confusion matrix).
- `artifacts/best_model.pt` — trained CPU-inference checkpoint, included in the completed bundle.
- `src/malaria_cnn/api.py` and `web/index.html` — local FastAPI inference API and responsive browser demo. Uploads are processed in memory and are not saved.
- `tests/` — offline tests for model output, mapping parsing, split leakage and API behavior.
- `docs/` — project report, data provenance, model card, measured results, and architecture diagram.

## Delivered model result

The included checkpoint was trained on the official 27,558-cell collection using a patient-grouped split. On the **3,928-cell held-out test partition from 29 patient groups**, it achieved **95.04% accuracy**, **94.66% parasitized-cell sensitivity**, **95.40% uninfected-cell specificity**, and **0.9856 ROC-AUC** at the fixed 0.5 threshold. There were 103 missed parasitized cell images and 92 uninfected images flagged as parasitized. These are internal cell-level benchmark results, not patient-level diagnostic accuracy or external clinical validation; full results and plots are in [`docs/RESULTS.md`](docs/RESULTS.md).

The raw dataset is **not** included in the source bundle. Download it when you want to retrain. The completed model artifact is included so the browser demo runs immediately after installation.

## Requirements and install

Python 3.10–3.12, about 2 GB free space for the Python environment and model, plus roughly 1 GB for the downloaded/extracted data when retraining. CPU is supported; a CUDA-capable GPU is optional. On a CPU-only computer, install a CPU-compatible PyTorch wheel if your package index defaults to a CUDA build.

```bash
cd /workspace/topic-06-malaria-parasite-detection-cnn
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[test]'
```

## Retrain from the original data

```bash
python scripts/download_dataset.py
python scripts/prepare_data.py --seed 42
python -m malaria_cnn.train --root .
```

The downloader saves the NIH/NLM archive and mapping files under `data/raw/`, extracts images to `data/images/`, and creates `data/raw/SHA256SUMS.txt`. The preparation script writes `data/processed/manifest.csv` and `summary.json`. Those data folders are excluded from the source archive and Git by design.

Default experiment: 96×96 RGB inputs, three convolutional blocks, AdamW, batch size 64, learning rate 0.001, six epochs maximum, early stopping on validation loss, fixed threshold 0.5, seed 42. Entire patient groups are assigned to one partition; both infected and healthy patient groups are represented in every partition. Test data do not update weights or select the checkpoint.

For a short software smoke run only, you can limit sample counts, but do **not** report those results as the final experiment:

```bash
python -m malaria_cnn.train --root . --epochs 1 --max-train-samples 256 --max-eval-samples 128
```

## Run the browser demo and API

```bash
uvicorn malaria_cnn.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. API documentation: <http://127.0.0.1:8000/docs>.

- `GET /api/health` — reports whether the checkpoint loaded.
- `GET /api/model-info` — lists the image-level task and model input shape.
- `POST /api/predict` — multipart upload field named `file`; accepts readable image files up to 5 MB.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F 'file=@cell.png'
```

The returned softmax scores are **not calibrated diagnostic probabilities**. The API does not store uploaded image bytes or request history.

## Tests

```bash
pytest -q
```

Tests do not download the dataset. They verify model shape and preprocessing, safe mapping parsing, deterministic patient-level partitions, and API input/response behavior. To reproduce the measured run, download the data and run the training commands above.

## Project documents and sources

- [Project report](docs/PROJECT_REPORT.md)
- [Project report PDF](docs/PROJECT_REPORT.pdf)
- [Measured results](docs/RESULTS.md)
- [Data provenance and reuse notes](docs/DATA_PROVENANCE.md)
- [Model card and intended-use limits](docs/MODEL_CARD.md)
- [Architecture diagram source](docs/ARCHITECTURE.mmd)

Key sources include the [official NIH/NLM malaria data sheet](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html), the [TensorFlow Datasets malaria catalog](https://www.tensorflow.org/datasets/catalog/malaria), and Rajaraman et al. (2018), [Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images](https://doi.org/10.7717/peerj.4568).
