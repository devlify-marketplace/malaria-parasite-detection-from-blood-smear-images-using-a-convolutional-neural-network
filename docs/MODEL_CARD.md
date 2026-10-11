# Model card: Malaria thin-smear cell CNN

## Model details

- **Model:** custom three-block convolutional neural network (`MalariaCNN`).
- **Version:** 1.0.0 research prototype.
- **Task:** binary classification of one segmented RGB cell crop.
- **Input:** resized 96×96 RGB cell image.
- **Output labels:** `Uninfected` (index 0) and `Parasitized` (index 1), plus uncalibrated softmax scores.
- **Checkpoint:** `artifacts/best_model.pt`.

## Intended use

Educational demonstrations and controlled, non-clinical research on a public cell-image benchmark. The model may be used to illustrate CNN training, patient-grouped evaluation, and local image inference.

## Out-of-scope and prohibited interpretations

This model is **not** a medical device, a patient-level diagnostic test, a parasite counter, a full-slide detector, a species/stage classifier, or a treatment decision tool. Do not use it to diagnose malaria, rule malaria out, delay care, select medication, or replace a qualified microscopist/approved clinical method. A low score cannot exclude infection.

## Training data and evaluation

Training data are the NLM-Falciparum thin-smear cell images described in `DATA_PROVENANCE.md`. All official patient-mapped cells from one patient are kept within one partition, with approximately 70%/15%/15% group allocation. Validation loss chooses the checkpoint; the test partition is reserved for final reporting. Actual split counts and measured performance are in `artifacts/model_metadata.json` and `artifacts/test_metrics.json`.

## Performance

See `docs/RESULTS.md` for the measured patient-held-out test results. These figures describe one internal split from a single acquisition collection, not independent prospective validation. Softmax values are not probability-calibrated and the fixed 0.5 threshold has not been clinically optimized.

## Known limitations and risks

- Source images are cropped cells with labels, not complete slides or confirmed patient diagnoses.
- Domain shift from stain, microscope, camera, resolution, population or laboratory may materially change behavior.
- The benchmark is balanced at the cell level; operational prevalence and predictive values may differ.
- False negatives could create dangerous reassurance; false positives could prompt unnecessary follow-up. No clinical operating threshold is established.
- The implementation does not provide confidence calibration, uncertainty estimation, robustness certification, subgroup fairness testing, or human-factors validation.
- Image compression, focus, illumination, crop selection and staining artifacts may affect predictions.

## Ethical considerations

The source images are de-identified research data as described by the source paper. This project does not use patient identifiers as model input and does not retain browser uploads. Patient mapping identifiers are only used to make the split group-aware. Follow applicable institutional and dataset terms before using or redistributing data or weights.

## Maintenance

A scientifically meaningful update requires a versioned data source and checksum, documented patient-group split, validation-based model selection, untouched test set, external validation, and an updated model card. Do not overwrite the final test results when tuning; create a new experiment and preserve the prior artifacts.
