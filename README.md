# Machine Learning Pipeline for Diabetes Risk Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Progetto di tesi triennale sulla predizione del rischio metabolico e sulla stratificazione del diabete a partire da dati clinici e biomarcatori.

---

## 📌 Panoramica
* **Obiettivo:** in corso di definizione.
* **Dataset:** dataset clinico bimodale pubblicato su *Scientific Data* (Nature Portfolio) [DOI: 10.1038/s41597-026-06923-y].

---

## 🚀 Avvio rapido

### 1. Configurazione dell'ambiente
```bash
git clone https://github.com/KekkoCoppola/MetaRisk.git
cd MetaRisk
python -m venv .venv
source .venv/bin/activate  # Su Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
```

Testato con **Python 3.12.10** su **Windows 11**. `requirements-lock.txt` fissa le versioni esatte dei pacchetti utilizzati per produrre i risultati; `requirements.txt` elenca le dipendenze dirette con le versioni minime.

### 2. Dati
Il dataset originale (Zenodo v10, CC BY 4.0) è incluso senza modifiche in [`data/raw/`](data/raw/). Fonte, licenza e checksum sono documentati in [`data/README.md`](data/README.md).

### 3. Split train/test e figure
```bash
python -m src.data.split                  # genera data/processed/train.csv e test.csv
python -m src.analytics.dataset_overview  # figure in analytics/dataset/
python -m src.analytics.split_report      # figure in analytics/split/
python -m pytest                          # verifiche sullo split
```

---

## 🗂️ Struttura del repository
```
analytics/          figure generate (dataset/, split/)
configs/            config.yaml: percorsi, seed, parametri dello split
data/
  raw/              dataset originale (incluso, CC BY 4.0)
  processed/        split train/test (non versionato)
  augmented/        training set bilanciati (non versionato)
docs/thesis/        materiale della tesi
src/
  config.py         caricamento della configurazione
  data/             caricamento e split dei dati
  analytics/        generazione delle figure
tests/              verifiche automatiche
```

---

## 📖 Citazione e riconoscimenti
Vedi [`CITATION.cff`](CITATION.cff) e la sezione "Provenienza e Citazione" in [`data/README.md`](data/README.md).

Il progetto si basa sui dati clinici open access pubblicati da Li et al. (2026):
* **Articolo del dataset:** *A bimodal dataset for diabetes research*, *Scientific Data* (Nature Portfolio). [DOI: 10.1038/s41597-026-06923-y](https://doi.org/10.1038/s41597-026-06923-y).
* **Archivio dei dati:** repository Zenodo [10.5281/zenodo.18270337](https://doi.org/10.5281/zenodo.18270337).

Il dataset è distribuito secondo i termini della licenza Creative Commons Attribution 4.0 International (CC BY 4.0).
