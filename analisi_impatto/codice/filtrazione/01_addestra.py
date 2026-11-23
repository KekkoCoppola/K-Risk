"""Filtrazione, passo 1 [PREDITTIVO]: CV annidata 5 x 5 sui due criteri KDIGO separati.

Piano: analisi_impatto/04_esperimenti/filtrazione/piano.md. Riusa senza modificarlo il codice della
Fase A (run_fold, fold imputati, spazi di ricerca): cambia solo il vettore bersaglio.
  G = eGFR < 60 (CKD-EPI 2021)      A = UMAUCR >= 30
Varianti degli ingressi: tutte (74 colonne), senza_sua, senza_eta.
Solo fold esterni 0-4: nessun modello finale, nessuna scrittura in models/. Il test set non viene letto.

Uso: python analisi_impatto/codice/filtrazione/01_addestra.py --bersaglio G --modelli xgboost lr_scored
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from src.config import CONFIG  # noqa: E402
from src.data.folds import OUTER  # noqa: E402
from src.data.kidney import egfr  # noqa: E402
from src.data.preprocess import FORBIDDEN  # noqa: E402
from src.data.split import PROCESSED  # noqa: E402
from src.models.phase_a import record_path, run_fold, sensitivity_spaces, xy  # noqa: E402

OUT = ROOT / "analisi_impatto" / "04_esperimenti" / "filtrazione" / "oof"
SPACES = sensitivity_spaces("depth_1_12")  # XGBoost con profondità 1-12, come dalla Fase B in poi
VARIANTI = {"tutte": [], "senza_sua": ["SUA"], "senza_eta": ["Age"]}


def bersaglio(df, sigla):
    if sigla == "G":
        return (egfr(df) < CONFIG["target"]["egfr_threshold"]).astype(int)
    if sigla == "A":
        return (df[CONFIG["columns"]["acr"]] >= CONFIG["target"]["acr_threshold"]).astype(int)
    raise ValueError(sigla)


def caricatore(nome, y, tolte):
    def carica(outer, inner):
        X_tr, y_tr, X_va, y_va = xy(nome, "main", y, outer, inner)
        tieni = [c for c in X_tr.columns if c not in tolte]
        assert not set(tieni) & set(FORBIDDEN), "colonna renale fra i predittori"
        return X_tr[tieni], y_tr, X_va[tieni], y_va
    return carica


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bersaglio", required=True, choices=["G", "A"])
    parser.add_argument("--modelli", nargs="+", required=True)
    parser.add_argument("--variante", default="tutte", choices=list(VARIANTI))
    parser.add_argument("--tentativi", type=int, default=CONFIG["phase_a"]["n_trials"])
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    train = pd.read_csv(PROCESSED / "train.csv")
    y = bersaglio(train, args.bersaglio)
    cartella = OUT / args.bersaglio / args.variante
    print(f"bersaglio {args.bersaglio}: {int(y.sum())} positivi su {len(y)}; variante {args.variante}", flush=True)
    for nome in args.modelli:
        for outer in range(OUTER):
            if record_path(cartella, nome, "main", outer).exists():
                continue
            record = run_fold(nome, "main", outer, y, args.tentativi, space=SPACES.get(nome),
                              output=cartella, loader=caricatore(nome, y, VARIANTI[args.variante]),
                              meta={"bersaglio": args.bersaglio, "variante": args.variante})
            print(f"{args.bersaglio} {args.variante} {nome} outer{outer}: {record['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
