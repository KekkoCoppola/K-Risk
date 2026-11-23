"""Filtrazione, passo 4 [PREDITTIVO]: validazione esterna VERA su NHANES 2005-2018.

Regola dello studente (10/11/2026): nessun riaddestramento su NHANES. Ogni modello è stimato una
volta sola sui 4.350 Jinuo del training (14 variabili comuni, unità convertite come al punto 6B)
e applicato a NHANES così com'è. Anche mediana per i mancanti e standardizzazione vengono dagli Jinuo.
Bersagli: G = eGFR < 60 (CKD-EPI 2021); A = ACR >= 30. Nessun esame renale fra i predittori.
Primaria fissata prima dell'esecuzione: XGBoost − C3 (SCORED-5) su G; secondaria chiave: C4 − C3.
"""
import importlib
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analisi_impatto" / "codice" / "confine_ai"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.data import kidney  # noqa: E402
from src.models.evaluation import delong  # noqa: E402

warnings.filterwarnings("ignore")
BASE = ROOT / "analisi_impatto" / "04_esperimenti" / "filtrazione"
SEED = 42
N_BOOT = 1000
C3 = ["eta", "female", "emoglobina", "PAS", "diabete"]


def parametri_tipici():
    params = [json.loads((BASE / "oof" / "G" / "tutte" / "main" / f"xgboost_outer{k}.json").read_text(encoding="utf-8"))["params"]
              for k in range(5)]
    return {k: (int(round(np.median([p[k] for p in params]))) if isinstance(params[0][k], int)
                else float(np.median([p[k] for p in params]))) for k in params[0]}


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


def primi(score, k, rng):
    sel = np.zeros(len(score), bool)
    sel[np.lexsort((rng.random(len(score)), -score))[:k]] = True
    return sel


def main():
    p6 = importlib.import_module("punto6_nonsup_e_validazione")
    Xj, _, _ = p6.jinuo_comuni()
    dj = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    d, _ = p6.p4.load()
    Xn = p6.nhanes_comuni(d)
    bersagli = {"G": ((kidney.egfr(dj) < 60).astype(int).to_numpy(), (d["eGFR"] < 60).astype(int).to_numpy()),
                "A": ((dj["UMAUCR"] >= 30).astype(int).to_numpy(), (d["URDACT"] >= 30).astype(int).to_numpy())}
    tip = parametri_tipici()
    logit = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=np.inf, max_iter=5000))  # noqa: E731
    tutte = list(Xj.columns)
    modelli = {
        "XGBoost (14 variabili)": (lambda: XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED, **tip), tutte),
        "Random Forest (14 variabili, come al punto 6B)": (lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=500, min_samples_leaf=20, n_jobs=-1, random_state=SEED)), tutte),
        "logistica (14 variabili)": (logit, tutte),
        "C3 SCORED-5": (logit, C3),
        "C4 SCORED-5 + SUA": (logit, C3 + ["acido_urico"]),
        "età + sesso + SUA": (logit, ["eta", "female", "acido_urico"]),
        "C2 età + sesso": (logit, ["eta", "female"]),
        "C1 sola età": (logit, ["eta"]),
    }
    rng = np.random.default_rng(SEED)
    righe, confronti, parita = [], [], []
    n = len(d)
    eta = Xn["eta"].to_numpy()
    for b, (yj, yn) in bersagli.items():
        P = {nome: costruisci().fit(Xj[col].to_numpy(), yj).predict_proba(Xn[col].to_numpy())[:, 1]
             for nome, (costruisci, col) in modelli.items()}
        for nome, p in P.items():
            auc, lo, hi = delong(yn, p)
            righe.append({"bersaglio": b, "modello": nome, "addestrato_su": f"Jinuo ({int(yj.sum())} positivi)", "n_nhanes": n,
                          "positivi_nhanes": int(yn.sum()), "auroc": auc, "ic_basso": lo, "ic_alto": hi,
                          "rischio_medio_predetto": p.mean(), "prevalenza_nhanes": yn.mean()})
        for nome, m in [("età < 60", eta < 60), ("età ≥ 60", eta >= 60)]:
            for mod in ["XGBoost (14 variabili)", "C3 SCORED-5", "C4 SCORED-5 + SUA"]:
                auc, lo, hi = delong(yn[m], P[mod][m])
                righe.append({"bersaglio": f"{b}, {nome}", "modello": mod, "addestrato_su": "Jinuo", "n_nhanes": int(m.sum()),
                              "positivi_nhanes": int(yn[m].sum()), "auroc": auc, "ic_basso": lo, "ic_alto": hi})
        coppie = [("XGBoost − C3 SCORED-5", "XGBoost (14 variabili)", "C3 SCORED-5"),
                  ("C4 (SCORED-5 + SUA) − C3", "C4 SCORED-5 + SUA", "C3 SCORED-5"),
                  ("Random Forest − C3 SCORED-5", "Random Forest (14 variabili, come al punto 6B)", "C3 SCORED-5"),
                  ("logistica 14 variabili − C3 SCORED-5", "logistica (14 variabili)", "C3 SCORED-5"),
                  ("XGBoost − C4 (SCORED-5 + SUA)", "XGBoost (14 variabili)", "C4 SCORED-5 + SUA"),
                  ("(età + sesso + SUA) − C2 età + sesso", "età + sesso + SUA", "C2 età + sesso"),
                  ("C3 SCORED-5 − C1 sola età", "C3 SCORED-5", "C1 sola età")]
        for nome, a, r in coppie:
            diff, lo, hi, pv = delong_appaiato(yn, P[a], P[r])
            confronti.append({"bersaglio": b, "confronto": nome, "diff_auroc": diff, "ic_basso": lo, "ic_alto": hi, "p": pv})
        for quota in [0.05, 0.10, 0.20]:
            k = int(round(quota * n))
            sel = {nome: primi(P[nome], k, rng) for nome in ["XGBoost (14 variabili)", "C4 SCORED-5 + SUA", "C3 SCORED-5"]}
            sel["più anziani"] = primi(eta, k, rng)
            riga = {"bersaglio": b, "quota": quota, "testati": k, "positivi_totali": int(yn.sum()),
                    **{nome: int(yn[s].sum()) for nome, s in sel.items()}}
            for cand in ["XGBoost (14 variabili)", "C4 SCORED-5 + SUA"]:
                dd = (sel[cand] & (yn == 1)).astype(int) - (sel["C3 SCORED-5"] & (yn == 1)).astype(int)
                boot = [dd[rng.integers(0, n, n)].sum() for _ in range(N_BOOT)]
                riga[f"{cand} − C3"] = int(dd.sum())
                riga[f"{cand} − C3 ic_basso"], riga[f"{cand} − C3 ic_alto"] = np.percentile(boot, [2.5, 97.5])
            parita.append(riga)
    pd.set_option("display.width", 260)
    print(f"Jinuo: {len(dj)} persone; NHANES: {n}; iperparametri XGBoost {tip}")
    for nome, tab in [("T1_trasporto_discriminazione", righe), ("T2_trasporto_confronti", confronti), ("T5_trasporto_parita_di_esami", parita)]:
        tab = pd.DataFrame(tab)
        tab.to_csv(BASE / f"{nome}.csv", index=False)
        print(f"\n{nome}\n", tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
