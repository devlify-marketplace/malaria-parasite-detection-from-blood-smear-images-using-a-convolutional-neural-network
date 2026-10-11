# Data provenance and reuse notes

## Training and evaluation data

The project uses the single-cell **NLM-Falciparum-Thin-Cell-Images** archive and patient-to-cell mappings linked from the National Library of Medicine's Lister Hill National Center for Biomedical Communications (LHNCBC) [malaria data sheet](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html).

- Cell archive: <https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip>
- Parasitized patient-to-cell map: <https://data.lhncbc.nlm.nih.gov/public/Malaria/patientid_cellmapping_parasitized.csv>
- Uninfected patient-to-cell map: <https://data.lhncbc.nlm.nih.gov/public/Malaria/patientid_cellmapping_uninfected.csv>
- Cross-reference: [TensorFlow Datasets malaria catalog](https://www.tensorflow.org/datasets/catalog/malaria)

The catalog describes 27,558 segmented cell images, balanced between 13,779 *Parasitized* and 13,779 *Uninfected*. The associated 2018 paper describes Giemsa-stained thin-smear images collected at Chittagong Medical College Hospital, Bangladesh, from an infected/healthy cohort, annotated by an expert reader, and segmented into red blood cells. See Rajaraman et al. (2018), [PeerJ article](https://doi.org/10.7717/peerj.4568), for the dataset and experimental context. The original training data and mapping files are downloaded locally by the script; they are intentionally excluded from the completed source archive because their terms should be checked at the upstream source before redistribution.

### Patient-level partitioning

The official mapping files associate cell filenames with patient identifiers. `scripts/prepare_data.py` uses those mappings to ensure that every mapped cell from a patient is assigned to only one of train, validation or test. Assignment is stratified by whether that patient's mapped cells include a parasitized example. Generated counts are saved in `data/processed/summary.json` and copied into the model metadata. This reduces within-patient leakage; it does not provide external validation across hospitals or imaging protocols.

### Reproducibility and checksums

After download, `scripts/download_dataset.py` validates the ZIP archive and writes `data/raw/SHA256SUMS.txt`. The checksums for the delivered local download are stored separately in `artifacts/source_checksums.json` when available. Because upstream archives may be updated, record the retrieval date, file size and checksum for any new run. The downloader does not execute code from the dataset.

## Web illustration

`web/assets/cdc-malaria-smear.png` is a blood-smear photomicrograph from CDC Public Health Image Library (PHIL) #2704, described on [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Plasmodium_falciparum_01.png) as *Plasmodium falciparum* gametocytes. The Commons file page identifies CDC/Dr. Mae Melvin as the content provider and marks the image **public domain** as a U.S. government work. Courtesy credit is included in the browser demo. It is an educational illustration, not a sample from the training/evaluation dataset and not evidence of model performance.

## Code and model artifacts

Project-authored source code is MIT-licensed; see `LICENSE`. This does not extend to the underlying dataset, patient mappings, public-domain CDC image or model weights. The included checkpoint was trained from the NIH/NLM data; check source data terms and institutional requirements before redistributing or using it beyond research/education. No patient names or direct identifiers are used by the project.
