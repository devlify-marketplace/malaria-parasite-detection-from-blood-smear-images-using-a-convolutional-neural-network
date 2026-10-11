# Malaria Parasite Detection from Blood Smear Images Using a Convolutional Neural Network

## Abstract

This project develops a convolutional neural network (CNN) research prototype for classifying segmented, Giemsa-stained thin-smear cell images into two dataset labels: *Parasitized* and *Uninfected*. The implementation uses the publicly documented NIH/National Library of Medicine (NLM) single-cell malaria image collection, together with the official patient-to-cell mappings. The dataset contains 27,558 images, evenly divided between the two cell labels. Unlike a random image-level split, this project assigns all mapped images from a patient to a single train, validation or test partition. The custom CNN is trained with light geometric augmentation, class-weighted cross-entropy and AdamW; validation loss selects the saved checkpoint. Evaluation is performed on the patient-held-out test set at a fixed 0.5 score threshold using accuracy, sensitivity, specificity, F1, ROC-AUC and Matthews correlation coefficient. The delivered software also provides a local HTTP API and browser demo for one uploaded cell image. Measured results from the included run are summarized in [RESULTS.md](RESULTS.md). This is a computational screening study on cropped cell images, not a clinical diagnostic product: it is not prospectively validated, is not a full-slide parasite detector, and must not be used to make patient-care decisions.

## 1. Introduction and problem statement

Microscopic blood-film review requires trained personnel, consistent staining and careful examination. Computer vision may help prioritize or standardize review of image data, particularly where resources are limited. However, a cell-image classifier is only one component of a diagnostic workflow. It does not replace slide preparation, image acquisition, expert review, quality controls, clinical context or confirmatory testing.

The project problem is to build and evaluate a reproducible CNN that distinguishes the two labels supplied for a segmented red-blood-cell image. The technical challenge is to learn visual patterns despite variation in cell shape, stain intensity, illumination and image scale, while avoiding the overly optimistic estimate that can result when cells from the same patient appear in both training and test data.

## 2. Aim, objectives and research questions

**Aim:** Develop a reproducible CNN research prototype for two-class classification of thin-smear malaria cell images.

**Objectives**

1. Acquire the official cell-image archive and patient mapping files from NIH/NLM and record their checksums.
2. Join mapped cell images to patient identifiers and create deterministic patient-grouped data partitions.
3. Design and train a compact CNN with training-only geometric augmentation and validation-based checkpoint selection.
4. Evaluate the locked model on held-out patient groups using class-sensitive classification metrics.
5. Implement a local API and browser interface that returns a clearly qualified research result for an individual cell image.

**Research questions**

- Can a compact CNN learn to separate the dataset's *Parasitized* and *Uninfected* cell labels?
- What sensitivity and specificity does the selected model achieve on patients not used for training or model selection?
- What limitations prevent these measurements from being interpreted as clinical diagnostic performance?

## 3. Scope and significance

The input is one RGB crop of a segmented cell. The output is one of two dataset classes plus uncalibrated model scores. The project does not detect cells in a whole-slide image, count parasites, estimate parasite density, classify life-cycle stage or species, identify a patient, or provide treatment advice. The web demo is intended for supervised research and education only.

The engineering contribution is an end-to-end implementation with data provenance, patient-level partitioning, a saved model, evaluation artifacts, an API and tests. The central methodological safeguard is grouping all mapped cells from a patient into one partition. This reduces direct patient overlap, but does not establish performance on a different hospital, microscope, staining protocol, population or prospective clinical cohort.

## 4. Related work

Rajaraman et al. (2018) studied customized and pretrained CNNs for classifying parasitized and uninfected cells from thin blood-smear imagery. Their paper describes 27,558 segmented cell images with equal class counts and reports patient-level cross-validation as a way to reduce bias from patient overlap. The present project adopts patient grouping and a compact custom CNN but is an educational reimplementation, not a replication of their full experimental protocol or reported model. The NIH/NLM data sheet additionally documents cell-image data and patient-to-cell mapping files, enabling this project to construct group-aware partitions.

## 5. Materials and methodology

### 5.1 Data source and preparation

The training data are obtained from the NIH/NLM single-cell archive and mapping CSVs listed in [DATA_PROVENANCE.md](DATA_PROVENANCE.md). The upstream description reports 13,779 parasitized and 13,779 uninfected images. Files are decoded as RGB, aspect-preserving center-fit resized to 96×96 pixels, and scaled to the [0, 1] interval. The training loader applies random horizontal/vertical flips and quarter-turn rotations; validation and test images are not augmented.

The patient-to-cell mapping files are parsed as text literals with `ast.literal_eval`; no remote code is executed. Patient identifiers are used only to keep groups intact. A fixed seed assigns groups in approximately 70%/15%/15% train/validation/test proportions, stratified by whether a patient's mapped records include any parasitized cell. The exact image and patient counts are saved in `data/processed/summary.json` and included in `artifacts/model_metadata.json`.

### 5.2 CNN architecture

The model has three convolutional blocks with 32, 64 and 128 channels. Each block contains two 3×3 convolutions, batch normalization, ReLU activation, 2×2 max pooling and dropout. Adaptive average pooling reduces the spatial feature map to one vector. A 64-unit dense layer, ReLU, dropout and a two-logit output layer produce the class scores. The architecture is deliberately small enough for CPU inference and has fewer than one million trainable parameters.

### 5.3 Training procedure

The default run uses seed 42, 96×96 inputs, batch size 64, AdamW at learning rate 0.001, weight decay 0.0001, weighted cross-entropy, and a maximum of six epochs. Early stopping (patience two) and checkpoint selection use validation loss only. The test partition is not used for parameter updates, early stopping or threshold tuning. The final evaluation threshold is fixed at 0.5. A CUDA device is selected when available; otherwise the full workflow runs on CPU.

### 5.4 Evaluation measures

The held-out test evaluation reports accuracy, balanced accuracy, parasitized-cell sensitivity and precision, uninfected-cell specificity, F1, ROC-AUC, MCC, cross-entropy loss, class counts and a confusion matrix. Sensitivity and specificity are both necessary to show different error types. ROC-AUC summarizes ranking across thresholds; it does not identify a clinically appropriate operating point. Softmax scores are not calibrated probabilities. Cell-level metrics do not directly measure patient-level diagnostic accuracy.

## 6. System implementation

The project is implemented in Python with PyTorch, Pillow, NumPy and scikit-learn. `scripts/download_dataset.py` acquires the source files and checks archive integrity; `scripts/prepare_data.py` writes the group-aware manifest; `malaria_cnn.train` produces a checkpoint, metrics, prediction CSV and figures. The FastAPI service accepts a single image, validates basic size/format constraints, computes the prediction in memory and does not persist the uploaded bytes. The responsive browser client displays the input preview and result, and prominently states the research-only limitation.

## 7. Results and discussion

The official mappings joined all 27,558 images to 201 patient groups. The deterministic patient-held-out partitions contain 19,112 training images from 141 groups, 4,518 validation images from 31 groups, and 3,928 test images from 29 groups. The CNN has 295,842 parameters; validation loss selected epoch 5 of 6. At the fixed 0.5 test threshold, the model achieved:

| Metric | Patient-held-out result |
| --- | ---: |
| Accuracy | 95.04% |
| Balanced accuracy | 95.03% |
| Sensitivity (Parasitized recall) | 94.66% |
| Specificity (Uninfected recall) | 95.40% |
| Precision (Parasitized) | 95.20% |
| F1 (Parasitized) | 94.93% |
| ROC-AUC | 0.9856 |
| Matthews correlation coefficient | 0.9007 |

The confusion counts were 1,907 correctly classified uninfected cells, 92 uninfected cells predicted parasitized, 103 parasitized cells predicted uninfected, and 1,826 correctly classified parasitized cells. The 103 misses are important: a high benchmark result does not remove false-negative risk. Exact results, plots and split counts are in [RESULTS.md](RESULTS.md) and `artifacts/`.

Patient grouping is more defensible than random cell-level assignment because it prevents cells from a mapped patient appearing in multiple partitions. However, this is still an internal holdout from one research collection and acquisition setting, not external validation. The test set has 29 patient groups and near-balanced cell labels; values are point estimates without confidence intervals. Cell-level performance is not patient-level diagnostic accuracy and is not directly comparable to studies with different splits, image resolution, model, tuning or prevalence.

## 8. Limitations, ethics and safety

- The data are cropped cell images, not full-slide scans or prospective patient cases.
- The public collection comes from a specific research setting; generalization to other sites, devices, stain batches, species and populations is unknown.
- The archive is balanced by cell label; real screening prevalence can differ substantially, changing predictive values.
- Expert labels and mappings are inherited from the dataset; this project does not independently adjudicate images.
- A single train/validation/test partition has sampling variability. A stronger study would pre-register repeated patient-grouped cross-validation and test externally on independently collected sites.
- The implementation does not calibrate probabilities, define clinical thresholds, conduct bias/fairness analysis or validate an end-to-end microscope workflow.
- The API accepts images only and does not save them; users should nevertheless avoid uploading identifiable or confidential material to any network-accessible deployment.
- Automated outputs could cause harm if treated as a diagnosis or used to delay care. This prototype must not direct treatment or replace expert interpretation.

## 9. Conclusion and future work

This project delivers an end-to-end CNN prototype for the dataset's binary cell-image task, including a patient-grouped evaluation path and local inference interface. Its principal result is the measured performance on held-out patient groups reported in `RESULTS.md`; it should be treated as a research baseline rather than clinical evidence. Future work should include repeated patient-level cross-validation, external validation across laboratories and acquisition devices, calibrated operating-point analysis, full-slide localization/counting, prospective workflow evaluation and review by qualified microscopy experts before any clinical consideration.

## References

1. Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude RJ, Jaeger S, Thoma GR. (2018). *Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images*. PeerJ, 6:e4568. <https://doi.org/10.7717/peerj.4568>
2. Rajaraman S, Silamut K, Hossain MA, et al. (2018). *Understanding the learned behavior of customized convolutional neural networks toward malaria parasite detection in thin blood smear images*. Journal of Medical Imaging, 5(3):034501. <https://doi.org/10.1117/1.JMI.5.3.034501>
3. National Library of Medicine, Lister Hill National Center for Biomedical Communications. *Malaria Data Sheet*. <https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html>
4. TensorFlow Datasets. *Malaria dataset catalog entry*. <https://www.tensorflow.org/datasets/catalog/malaria>
5. He K, Zhang X, Ren S, Sun J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. <https://doi.org/10.1109/CVPR.2016.90>
