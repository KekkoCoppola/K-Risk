"""Punteggio dell'AI senza etichetta per gruppi di ACR, split G (post-hoc, dopo la revisione NS-04).

[ESPLICATIVO] Sostituisce la lettura "confine a 7 mg/g" della regressione segmentata. Per ogni
configurazione: punteggio standardizzato dentro il fold, media con IC 95% (normale) nei gruppi di
ACR fissati qui (< 4, 4–7, 7–13, 13–26, 26–30, ≥ 30), e due confronti con Mann-Whitney:
7–13 contro 4–7 (c'è un salto a 7?) e 13–26 contro 7–13 (c'è un salto sotto 30, escludendo 26–30).
Solo aggregati.

Uscite: `04_esperimenti/non_supervisionato/parteB/G/posthoc_fasce_acr.csv` e
`posthoc_fasce_acr_confronti.csv`.
"""
import numpy as np
import pandas as pd
from scipy import stats

import prep

ROOT = prep.ROOT
from src.data import kidney  # noqa: E402

OUT = ROOT / "analisi_impatto/04_esperimenti/non_supervisionato/parteB/G"
GROUPS = [("< 4", 0, 4), ("4–7", 4, 7), ("7–13", 7, 13), ("13–26", 13, 26), ("26–30", 26, 30), ("≥ 30", 30, np.inf)]
CONFIGS = ["kmeans_k4", "kmeans_gap", "kmeans_k2", "autoencoder", "mahalanobis", "gmm_k4", "gmm_k2"]


def main():
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    acr = pd.to_numeric(df["UMAUCR"], errors="coerce").to_numpy()
    y = kidney.target(df).to_numpy()
    z = np.load(prep.CACHE / "G_out" / "punteggi_oof.npz")
    rows, tests = [], []
    for cfg in CONFIGS:
        s = z[f"s_{cfg}"].astype(float).copy()
        for k in range(5):
            r = z["fold"] == k
            s[r] = (s[r] - s[r].mean()) / s[r].std()
        masks = {}
        for name, lo, hi in GROUPS:
            m = (acr >= lo) & (acr < hi)
            masks[name] = m
            v = s[m]
            half = 1.96 * v.std() / np.sqrt(len(v))
            rows.append({"configurazione": cfg, "gruppo_acr_mg_g": name, "n": int(m.sum()), "positivi_kdigo": int(y[m].sum()),
                         "punteggio_medio_z": v.mean(), "ic_basso": v.mean() - half, "ic_alto": v.mean() + half})
        for a, b in [("4–7", "7–13"), ("7–13", "13–26")]:
            va, vb = s[masks[a]], s[masks[b]]
            tests.append({"configurazione": cfg, "confronto": f"{b} contro {a}", "differenza_medie": vb.mean() - va.mean(),
                          "p_mann_whitney": stats.mannwhitneyu(vb, va).pvalue})
    pd.DataFrame(rows).to_csv(OUT / "posthoc_fasce_acr.csv", index=False)
    t = pd.DataFrame(tests)
    t.to_csv(OUT / "posthoc_fasce_acr_confronti.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
