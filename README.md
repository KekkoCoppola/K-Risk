# Machine Learning Pipeline for Diabetes Risk Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Bachelor's Thesis project focused on metabolic risk prediction and diabetes stratification using clinical and biomarker data.

---

## 📌 Overview
* **Objective:** Work in progress.
* **Dataset:** Bimodal clinical dataset from *Scientific Data* (Nature Portfolio) [DOI: 10.1038/s41597-026-06923-y].

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/KekkoCoppola/MetaRisk.git
cd MetaRisk
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
```

Tested with **Python 3.12.10** on **Windows 11**. `requirements-lock.txt` pins the exact package versions used to produce the results; `requirements.txt` lists the direct dependencies with minimum versions.

### 2. Data Acquisition
The original dataset (Zenodo v10, CC BY 4.0) is included unmodified in [`data/raw/`](data/raw/). Source, license and checksums are documented in [`data/README.md`](data/README.md).

### 3. Train/Test Split and Figures
```bash
python -m src.data.split                  # writes data/processed/train.csv, test.csv
python -m src.analytics.dataset_overview  # figures in analytics/dataset/
python -m src.analytics.split_report      # figures in analytics/split/
python -m pytest                          # split checks
```

---

## 🗂️ Repository Structure
```
analytics/          generated figures (dataset/, split/)
configs/            config.yaml: paths, seed, split parameters
data/
  raw/              original dataset (included, CC BY 4.0)
  processed/        train/test split (not versioned)
  augmented/        balanced training sets (not versioned)
docs/thesis/        thesis material
src/
  config.py         configuration loader
  data/             loading and splitting
  analytics/        figure generation
tests/              automated checks
```

---

## 📖 Citation & Acknowledgements
See [`CITATION.cff`](CITATION.cff) and the Data Attribution section in [`data/README.md`](data/README.md).

This project is built upon the open-access clinical data provided by Li et al. (2026):
* **Dataset Paper:** *A bimodal dataset for diabetes research*, *Scientific Data* (Nature Portfolio). [DOI: 10.1038/s41597-026-06923-y](https://doi.org/10.1038/s41597-026-06923-y).
* **Data Archive:** Zenodo repository [10.5281/zenodo.18270337](https://doi.org/10.5281/zenodo.18270337).

The dataset is distributed under the terms of the Creative Commons Attribution 4.0 International License (CC BY 4.0).