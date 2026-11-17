"""Confronto post-hoc (non preregistrato), split G.

[ESPLICATIVO] Punteggi non supervisionati di G (78 colonne, con le misure renali grezze, senza
etichetta) contro i supervisionati della Fase A (74 colonne, senza misure renali, con etichetta),
sugli stessi 4.350 soggetti e sugli stessi 5 fold esterni. Non stima una capacità predittiva:
dice quanto vale avere le misure grezze senza la formula, rispetto ad avere la formula (come
etichetta) senza le misure.

Uscita: `04_esperimenti/non_supervisionato/parteB/G/posthoc_contro_fase_a.csv`.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import prep

ROOT = prep.ROOT
from src.data import kidney  # noqa: E402
from src.models.evaluation import delong  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load("01_parteA")
B = _load("03_parteB")
S = _load("04_supervisionato_G")
OUT = ROOT / "analisi_impatto/04_esperimenti/non_supervisionato/parteB/G"
SUP = {"fase_a_random_forest": "phase_a/main/random_forest", "fase_a_xgboost": "phase_a/main/xgboost",
       "fase_a_lr_penalized": "phase_a/main/lr_penalized"}


def main():
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    y = kidney.target(df).to_numpy()
    n = len(y)
    z = np.load(prep.CACHE / "G_out" / "punteggi_oof.npz")
    fold_u = z["fold"]
    uns = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
    rows = []
    for sname, member in SUP.items():
        p, fold_s = A.load_scores(member, n)
        assert (fold_s == fold_u).all(), "fold diversi fra Fase A e split G"
        a_s = delong(y, p)[0]
        for u, s in uns.items():
            d, lo, hi, pv = S.delong_paired(y, s, p)
            rows.append({"non_supervisionato_G": u, "supervisionato_fase_a": sname,
                         "auroc_non_supervisionato": delong(y, s)[0], "auroc_supervisionato": a_s,
                         "differenza_auroc": d, "ic_basso": lo, "ic_alto": hi, "p": pv,
                         **{f"positivi_{int(f * 100)}pct_non_sup": B.at_fraction(s, fold_u, y, f) for f in (0.05, 0.10, 0.20)},
                         **{f"positivi_{int(f * 100)}pct_sup": B.at_fraction(p, fold_u, y, f) for f in (0.05, 0.10, 0.20)}})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "posthoc_contro_fase_a.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
