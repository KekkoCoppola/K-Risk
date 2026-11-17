"""B7 del piano preregistrato: riferimento supervisionato sullo split G.

[ESPLICATIVO] G contiene SCRE, UmALB, UCRE: il target è quasi una funzione degli ingressi,
quindi l'AUROC qui misura quanto un modello con l'etichetta ricostruisce KDIGO, non una
capacità predittiva. Serve a misurare il valore dell'etichetta rispetto al non supervisionato.

Logistica elastic net, Random Forest, XGBoost: 5×5 annidato come la Fase A (spazi, tentativi,
pruner, PR-AUC importati da `src/models`), sulla cache imputata di G (`prep.py`).
Previsioni per paziente: `data/processed/non_supervisionato/G_out/supervisionato_*.json`.
Metriche aggregate: `04_esperimenti/non_supervisionato/parteB/G/B7_*.csv`.
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import prep

ROOT = prep.ROOT
from src.data import kidney  # noqa: E402
from src.models.evaluation import delong  # noqa: E402
from src.models.phase_a import N_TRIALS, fit, tune  # noqa: E402
from src.models.zoo import SPACES  # noqa: E402

warnings.filterwarnings("ignore")
MODELS = ["lr_penalized", "random_forest", "xgboost"]
STORE = prep.CACHE / "G_out"
OUT = ROOT / "analisi_impatto/04_esperimenti/non_supervisionato/parteB/G"


def delong_paired(y, p1, p2):
    """Differenza di AUROC (p1 − p2) con IC e p di DeLong per campioni appaiati."""
    y = np.asarray(y) == 1
    comps = []
    for p in (p1, p2):
        pos, neg = p[y], p[~y]
        allr = stats.rankdata(np.concatenate([pos, neg]))
        m, n = len(pos), len(neg)
        v10 = (allr[:m] - stats.rankdata(pos)) / n
        v01 = 1 - (allr[m:] - stats.rankdata(neg)) / m
        comps.append((v10.mean(), v10, v01))
    d = comps[0][0] - comps[1][0]
    s10 = np.cov(np.vstack([comps[0][1], comps[1][1]]))
    s01 = np.cov(np.vstack([comps[0][2], comps[1][2]]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / y.sum() + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / (~y).sum()
    se = np.sqrt(var)
    return d, d - 1.96 * se, d + 1.96 * se, 2 * stats.norm.sf(abs(d / se))


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    y = pd.Series(kidney.target(df).to_numpy())
    n = len(y)
    timing = []
    for name in MODELS:
        path = STORE / f"supervisionato_{name}.json"
        if path.exists():
            continue
        prob = np.full(n, np.nan)
        params_all = []
        for k in range(5):
            t0 = time.perf_counter()
            data = []
            for j in range(5):
                X_tr, X_va = prep.load("G", f"outer{k}_inner{j}")
                data.append((X_tr, y.iloc[X_tr.index], X_va, y.iloc[X_va.index]))
            study = tune(name, data, SPACES.get(name), N_TRIALS)
            X_tr, X_va = prep.load("G", f"outer{k}")
            model = fit(name, study.best_params, X_tr, y.iloc[X_tr.index])
            prob[X_va.index.to_numpy()] = model.predict_proba(X_va)[:, 1]
            params_all.append(study.best_params)
            timing.append({"modello": name, "fold": k, "secondi": round(time.perf_counter() - t0, 1)})
            print(f"{name} fold {k}: {timing[-1]['secondi']} s", flush=True)
        path.write_text(json.dumps({"probability": prob.tolist(), "params": params_all}), encoding="utf-8")
    if timing:
        pd.DataFrame(timing).to_csv(OUT / "B7_tempi.csv", index=False)

    # confronto: supervisionato contro ogni punteggio non supervisionato di G
    sup = {m: np.asarray(json.loads((STORE / f"supervisionato_{m}.json").read_text())["probability"]) for m in MODELS}
    z = np.load(STORE / "punteggi_oof.npz")
    uns = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
    yy = y.to_numpy()
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("parteB", Path(__file__).parent / "03_parteB.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    rows = [{"modello": m, **dict(zip(["auroc", "ic_basso", "ic_alto"], delong(yy, p))),
             **{f"positivi_al_{int(f * 100)}pct": B.at_fraction(p, z["fold"], yy, f) for f in (0.05, 0.10, 0.20)}}
            for m, p in sup.items()]
    pd.DataFrame(rows).to_csv(OUT / "B7_supervisionato_auroc.csv", index=False)
    comp = []
    for m, p in sup.items():
        for u, s in uns.items():
            d, lo, hi, pv = delong_paired(yy, p, s)
            comp.append({"supervisionato": m, "non_supervisionato": u, "differenza_auroc": d,
                         "ic_basso": lo, "ic_alto": hi, "p": pv})
    pd.DataFrame(comp).to_csv(OUT / "B7_valore_etichetta.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
