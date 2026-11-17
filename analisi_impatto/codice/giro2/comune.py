"""Funzioni comuni del giro 2 (fattibilità delle idee, agente auditor-dati).

Regole:
- legge solo data/processed/train.csv e i record delle previsioni in analytics/ (esclusa
  analytics/test/, di cui non si apre nessun file); data/processed/test.csv non viene MAI letto;
- solo inventario, conteggi e tempi: nessuna metrica di esito delle idee (niente AUROC nuove,
  niente disaccordo o correlazione fra modelli, niente copertura conformal);
- le uscite sono solo aggregati (nessuna riga di paziente); seed 42.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.data.folds import inner_folds, outer_folds  # noqa: E402
from src.data.kidney import egfr, kdigo_level, target  # noqa: E402

SEED = 42
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
ANALYTICS = ROOT / "analytics"
TRAIN_FILE = ROOT / "data" / "processed" / "train.csv"
NUMERIC = CONFIG["features"]["numeric"]
CATEGORICAL = CONFIG["features"]["categorical"]
INPUTS = NUMERIC + CATEGORICAL


def load_train():
    """Training set (4.350 righe). Il test set non viene letto."""
    df = pd.read_csv(TRAIN_FILE, parse_dates=["Data"])
    assert len(df) == 4350, len(df)
    return df


def components(df):
    """Bersaglio composito e componenti (come src/data/kidney.py e la Fase D)."""
    acr = (df[CONFIG["columns"]["acr"]] >= CONFIG["target"]["acr_threshold"]).to_numpy()
    low = (egfr(df) < CONFIG["target"]["egfr_threshold"]).to_numpy()
    y = target(df).to_numpy()
    assert np.array_equal(y, (acr | low).astype(int))
    return {"y": y, "only_alb": acr & ~low, "low_egfr": low, "negative": ~acr & ~low}
