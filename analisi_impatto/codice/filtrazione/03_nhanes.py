"""Filtrazione, passo 3 [PREDITTIVO]: replica esterna su NHANES 2005-2018.

Stessa coorte del punto 4 del confine AI (funzione load() di punto4_nhanes.py, non modificata).
Bersaglio G = eGFR < 60 (CKD-EPI 2021); A = ACR >= 30. Nessun esame renale fra i predittori.
XGBoost con gli iperparametri tipici (mediana dei 5 fold esterni) scelti sugli Jinuo per G:
nessuna ottimizzazione su NHANES. CV a 5 fold stratificata, seme 42. Solo aggregati.
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.models.evaluation import delong  # noqa: E402

warnings.filterwarnings("ignore")
BASE = ROOT / "analisi_impatto" / "04_esperimenti" / "filtrazione"
SEED = 42
N_BOOT = 1000
FEAT = ["RIDAGEYR", "female", "BMXBMI", "BMXWAIST", "SBP", "DBP", "LBXGH", "LBXGLU", "LBXTR", "LBDHDD",
        "LBXSUA", "LBXHGB", "diab_noto", "iper_nota"]
C3 = ["RIDAGEYR", "female", "LBXHGB", "SBP", "DM"]


def modulo(nome, path):
    spec = importlib.util.spec_from_file_location(nome, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parametri_tipici():
    params = [json.loads((BASE / "oof" / "G" / "tutte" / "main" / f"xgboost_outer{k}.json").read_text(encoding="utf-8"))["params"]
              for k in range(5)]
    return {k: (int(round(np.median([p[k] for p in params]))) if isinstance(params[0][k], int)
                else float(np.median([p[k] for p in params]))) for k in params[0]}


def oof(modello, X, y, pieghe):
    p = np.zeros(len(y))
    for tr, te in pieghe:
        p[te] = modello().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def componenti(y, p):
    y = y == 1
    pos, neg = p[y], p[~y]
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)] - stats.rankdata(pos)) / len(neg), 1 - (r[len(pos):] - stats.rankdata(neg)) / len(pos)


def delong_appaiato(y, p1, p2):
    a10, a01 = componenti(y, p1)
    b10, b01 = componenti(y, p2)
    d = a10.mean() - b10.mean()
    se = np.sqrt(np.var(a10 - b10, ddof=1) / len(a10) + np.var(a01 - b01, ddof=1) / len(a01))
    return d, d - 1.96 * se, d + 1.96 * se, 2 * stats.norm.sf(abs(d) / se)


def scegli(score, fold, quota, rng):
    sel = np.zeros(len(score), bool)
    for k in np.unique(fold):
        idx = np.flatnonzero(fold == k)
        ordine = np.lexsort((rng.random(len(idx)), -score[idx]))
        sel[idx[ordine[: int(round(quota * len(idx)))]]] = True
    return sel


def main():
    p4 = modulo("punto4_nhanes", ROOT / "analisi_impatto" / "codice" / "confine_ai" / "punto4_nhanes.py")
    d, _ = p4.load()
    d["female"] = d["female"].astype(float)
    d["DM"] = (d["diab_noto"].eq(1) | (d["LBXGH"] >= 6.5) | (d["LBXGLU"] >= 126)).astype(float)
    yG = (d["eGFR"] < 60).astype(int).to_numpy()
    yA = (d["URDACT"] >= 30).astype(int).to_numpy()
    n = len(d)
    tip = parametri_tipici()
    print(f"NHANES: n {n}; G {yG.sum()} ({yG.mean():.2%}); A {yA.sum()} ({yA.mean():.2%}); iperparametri {tip}")
    rng = np.random.default_rng(SEED)
    xgb = lambda: XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED, **tip)  # noqa: E731
    logit = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000))  # noqa: E731
    col = lambda nomi: d[nomi].astype(float).to_numpy()  # noqa: E731

    righe, confronti, parita = [], [], []
    for b, y in [("G", yG), ("A", yA)]:
        pieghe = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(n), y))
        fold = np.zeros(n, int)
        for k, (_, te) in enumerate(pieghe):
            fold[te] = k
        P = {"XGBoost (14 variabili)": oof(xgb, col(FEAT), y, pieghe),
             "C3 SCORED-5": oof(logit, col(C3), y, pieghe),
             "C1 sola età": oof(logit, col(["RIDAGEYR"]), y, pieghe),
             "C2 età + sesso": oof(logit, col(["RIDAGEYR", "female"]), y, pieghe),
             "C4 SCORED-5 + SUA": oof(logit, col(C3 + ["LBXSUA"]), y, pieghe),
             "età + sesso + SUA": oof(logit, col(["RIDAGEYR", "female", "LBXSUA"]), y, pieghe),
             "XGBoost senza SUA": oof(xgb, col([c for c in FEAT if c != "LBXSUA"]), y, pieghe),
             "XGBoost senza età": oof(xgb, col([c for c in FEAT if c != "RIDAGEYR"]), y, pieghe)}
        for nome, p in P.items():
            auc, lo, hi = delong(y, p)
            righe.append({"bersaglio": b, "modello": nome, "n": n, "positivi": int(y.sum()), "auroc": auc, "ic_basso": lo, "ic_alto": hi})
        if b != "G":
            continue
        pX = P["XGBoost (14 variabili)"]
        for nome, a, r in [("PRIMARIA: XGBoost − C3 SCORED-5", pX, P["C3 SCORED-5"]),
                           ("XGBoost − C1 sola età", pX, P["C1 sola età"]),
                           ("XGBoost − C2 età + sesso", pX, P["C2 età + sesso"]),
                           ("C4 (SCORED-5 + SUA) − C3", P["C4 SCORED-5 + SUA"], P["C3 SCORED-5"]),
                           ("XGBoost − XGBoost senza SUA", pX, P["XGBoost senza SUA"]),
                           ("XGBoost − XGBoost senza età", pX, P["XGBoost senza età"]),
                           ("XGBoost senza età − C3", P["XGBoost senza età"], P["C3 SCORED-5"])]:
            diff, lo, hi, pv = delong_appaiato(y, a, r)
            confronti.append({"confronto": nome, "diff_auroc": diff, "ic_basso": lo, "ic_alto": hi, "p": pv})
        eta = d["RIDAGEYR"].to_numpy(float)
        for quota in [0.05, 0.10, 0.20]:
            sel = {"XGBoost": scegli(pX, fold, quota, rng), "C3 SCORED-5": scegli(P["C3 SCORED-5"], fold, quota, rng),
                   "più anziani": scegli(eta, fold, quota, rng)}
            riga = {"quota": quota, "testati": int(sel["XGBoost"].sum()), "positivi_totali": int(y.sum()),
                    **{k: int(y[v].sum()) for k, v in sel.items()}}
            for rif in ["C3 SCORED-5", "più anziani"]:
                dd = (sel["XGBoost"] & (y == 1)).astype(int) - (sel[rif] & (y == 1)).astype(int)
                boot = [dd[rng.integers(0, n, n)].sum() for _ in range(N_BOOT)]
                riga[f"diff_vs_{rif}"] = int(dd.sum())
                riga[f"ic_basso_vs_{rif}"], riga[f"ic_alto_vs_{rif}"] = np.percentile(boot, [2.5, 97.5])
            parita.append(riga)
        # fasce di età: SCORED è nato su NHANES, il confronto si guarda anche dentro le fasce (descrittivo)
        for nome, m in [("età < 60", eta < 60), ("età ≥ 60", eta >= 60)]:
            for mod in ["XGBoost (14 variabili)", "C3 SCORED-5"]:
                auc, lo, hi = delong(y[m], P[mod][m])
                righe.append({"bersaglio": f"G, {nome}", "modello": mod, "n": int(m.sum()), "positivi": int(y[m].sum()),
                              "auroc": auc, "ic_basso": lo, "ic_alto": hi})
    pd.set_option("display.width", 250)
    for nome, tab in [("N1_discriminazione", righe), ("N2_confronti", confronti), ("N5_parita_di_esami", parita)]:
        tab = pd.DataFrame(tab)
        tab.to_csv(BASE / f"{nome}.csv", index=False)
        print(f"\n{nome}\n", tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
