"""Ablazione post-hoc (non preregistrata), split T.

[ESPLICATIVO] Quanto dell'AUROC dei punteggi di T viene dalle colonne che codificano il target?
Con gli stessi modelli già stimati (nessuna nuova stima), il punteggio si ricalcola sul fold
esterno usando (a) tutte le colonne, (b) tutte tranne i gruppi `leakage` e `renal_exams` di
`configs/config.yaml` (18 colonne), (b2) tutte tranne le 14 di quei gruppi che lo split G non ha
(si tengono `SCRE`, `UmALB`, `UCRE`, `BUN`; variante aggiunta dopo la revisione NS-04), (c) solo
le 18. Per k-means e k-prototypes: distanza dal
centroide più numeroso sulle sole dimensioni scelte; autoencoder: errore di ricostruzione medio
sulle sole colonne scelte; Mahalanobis: distanza marginale (sottomatrice della covarianza
robusta). Le misture gaussiane non sono incluse.

Uscita: `04_esperimenti/non_supervisionato/parteB/T/posthoc_ablazione_target.csv`.
"""
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import prep

ROOT = prep.ROOT
from src.config import CONFIG  # noqa: E402
from src.data import kidney  # noqa: E402
from src.models.evaluation import delong  # noqa: E402

_spec = importlib.util.spec_from_file_location("parteB", Path(__file__).parent / "03_parteB.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
KPrototypes = B.KPrototypes  # il pickle cerca la classe in __main__
_spec_s = importlib.util.spec_from_file_location("supG", Path(__file__).parent / "04_supervisionato_G.py")
S = importlib.util.module_from_spec(_spec_s)
_spec_s.loader.exec_module(S)

OUT = ROOT / "analisi_impatto/04_esperimenti/non_supervisionato/parteB/T"
EX = CONFIG["features"]["exclude"]
TARGET_COLS = set(EX["leakage"]) | set(EX["renal_exams"])
# le 10 classificazioni derivate del gruppo `leakage` (CLAUDE.md, "Variabili escluse per leakage")
CLASSIFICATIONS = {"highCr", "HighUmALB", "HighACR", "ACR3degree", "GFR5lev", "GFR6levG5", "EUGFRabACR012",
                   "EUGFR90UACR01", "EUGFR60abACR012", "UACRGFR"}


def scores(pack, X, keep):
    """Punteggi di ogni configurazione sulle sole colonne `keep` (insieme di nomi)."""
    mods, cont = pack["models"], pack["continuous"]
    cols = list(X.columns)
    A = X.to_numpy()
    mask = np.array([c in keep for c in cols])
    out = {}
    for key, m in mods["kmeans"].items():
        big = int(np.bincount(m.labels_).argmax())
        out[f"kmeans_{key}"] = np.sqrt((((A - m.cluster_centers_[big]) ** 2)[:, mask]).sum(1))
    kp = mods["kprototypes"]
    catc = kp["categoriche"]
    mn = np.array([c in keep for c in cont])
    mc = np.array([c in keep for c in catc])
    Xn, Xc = X[cont].to_numpy(), X[catc].to_numpy().round(6)
    for key, m in kp.items():
        if key == "categoriche":
            continue
        big = int(np.bincount(m.labels_, minlength=m.k).argmax())
        dn = ((Xn - m.cn[big]) ** 2)[:, mn].sum(1)
        dc = (Xc != m.cc[big])[:, mc].sum(1)
        out[f"kproto_{key}"] = np.sqrt(dn + m.gamma * dc)
    ae = mods["autoencoder"]
    out["autoencoder"] = ((ae.predict(A) - A) ** 2)[:, mask].mean(1)
    mcd = mods["mahalanobis"]
    if mn.any():
        D = Xn[:, mn] - mcd.location_[mn]
        P = np.linalg.pinv(mcd.covariance_[np.ix_(mn, mn)])
        out["mahalanobis"] = np.einsum("ij,jk,ik->i", D, P, D)
    return out


def main():
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    y = kidney.target(df).to_numpy()
    store = prep.CACHE / "T_out"
    pooled, per_fold = {}, []
    n_target = None
    for k in range(5):
        pack = joblib.load(store / f"modelli_outer{k}.joblib")
        _, X_va = prep.load("T", f"outer{k}")
        cols = set(X_va.columns)
        tcols = cols & TARGET_COLS
        n_target = len(tcols)
        yv = y[X_va.index]
        g_cols = set(prep.RENAL)  # SCRE, UmALB, UCRE, BUN: presenti anche in G
        variants = [("tutte", cols), ("senza_colonne_del_target", cols - tcols), ("solo_colonne_del_target", tcols),
                    ("senza_le_14_assenti_in_G", cols - (tcols - g_cols)),
                    # richieste dello studente dopo NS-05: solo la colonna-diagnosi, poi le 10 classificazioni
                    ("senza_solo_EUGFR90UACR01", cols - {"EUGFR90UACR01"}),
                    ("senza_le_10_classificazioni", cols - CLASSIFICATIONS)]
        for variant, keep in variants:
            for cfg, s in scores(pack, X_va, keep).items():
                per_fold.append({"fold": k, "variante": variant, "configurazione": cfg, "auroc": roc_auc_score(yv, s)})
                z = (s - s.mean()) / (s.std() or 1.0)
                acc = pooled.setdefault((variant, cfg), [[], []])
                acc[0].append(z)
                acc[1].append(yv)
    pf = pd.DataFrame(per_fold)
    # confronto appaiato con lo split G (stessa configurazione, punteggi z dentro il fold, stesso ordine di righe)
    zg = np.load(prep.CACHE / "G_out" / "punteggi_oof.npz")
    order = np.concatenate([prep.load("T", f"outer{k}")[1].index.to_numpy() for k in range(5)])
    fold_g = zg["fold"]
    rows = []
    for (variant, cfg), (s, yy) in pooled.items():
        s, yy = np.concatenate(s), np.concatenate(yy)
        a, lo, hi = delong(yy, s)
        g = pf[(pf.variante == variant) & (pf.configurazione == cfg)].auroc
        row = {"variante": variant, "configurazione": cfg, "auroc_unita_z": a, "ic_basso": lo, "ic_alto": hi,
               "auroc_media_fold": g.mean(), "auroc_min_fold": g.min(), "auroc_max_fold": g.max(),
               "colonne_del_target": n_target}
        if f"s_{cfg}" in zg.files:
            sg = zg[f"s_{cfg}"].astype(float).copy()
            for k in range(5):
                r = fold_g == k
                sg[r] = (sg[r] - sg[r].mean()) / (sg[r].std() or 1.0)
            d, dlo, dhi, p = S.delong_paired(yy, s, sg[order])
            row.update({"auroc_G_unita_z": delong(yy, sg[order])[0], "differenza_contro_G": d,
                        "ic_basso_differenza": dlo, "ic_alto_differenza": dhi, "p_contro_G": p})
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["configurazione", "variante"])
    res.to_csv(OUT / "posthoc_ablazione_target.csv", index=False)
    print(sorted(TARGET_COLS))
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
