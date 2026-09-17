# Dataset Documentation & Data Dictionary

Questo modulo gestisce l'acquisizione, l'ispezione e il preprocessing dei dati clinici utilizzati nel progetto di tesi.

---

## 1. Provenienza e Citazione

I dati provengono dal Data Descriptor open access pubblicato su *Scientific Data* (Nature Portfolio) e archiviato su Zenodo.

* **Titolo articolo:** *A bimodal dataset for diabetes research*
* **Journal:** Scientific Data (2026)
* **DOI Articolo:** [10.1038/s41597-026-06923-y](https://doi.org/10.1038/s41597-026-06923-y)
* **PubMed ID:** [41813689](https://pubmed.ncbi.nlm.nih.gov/41813689/)
* **Zenodo Record:** [10.5281/zenodo.18270337](https://doi.org/10.5281/zenodo.18270337)
* **Supplementary Information (DOCX):** [Media Object MOESM1](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41597-026-06923-y/MediaObjects/41597_2026_6923_MOESM1_ESM.docx)

Se utilizzi o estendi questo repository, cita la fonte primaria:

```bibtex
@article{li2026bimodal,
  title   = {A bimodal dataset for diabetes research},
  journal = {Scientific Data},
  year    = {2026},
  doi     = {10.1038/s41597-026-06923-y},
  url     = {https://doi.org/10.1038/s41597-026-06923-y}
}
```

---

## 2. File inclusi nel repository

Il dataset originale è incluso in `data/raw/` senza modifiche, come consentito dalla licenza **Creative Commons Attribution 4.0 International (CC BY 4.0)**, citando la fonte indicata sopra.

| File | Origine | SHA-256 |
|------|---------|---------|
| `raw/Dataset for diabetes research.xlsx` | Zenodo, record 10.5281/zenodo.18270337 (v10) | `4e0545be6ba61381368100556f7f18980b19a5d4d444ee6fb75263e207bd5380` |
| `raw/Supplementary Information.pdf` | Supplementary Information dell'articolo | `7d89ffb7c83ac74c70b9410eb3800ad674614c9dd8d6741e278c46a17a8589ea` |

L'impronta SHA-256 permette di verificare che il file sia identico a quello usato nel progetto:

```bash
sha256sum "data/raw/Dataset for diabetes research.xlsx"
```

Le cartelle `processed/` (split train/test) e `augmented/` (training set bilanciati) contengono solo dati generati dagli script in `src/` e non sono versionate.