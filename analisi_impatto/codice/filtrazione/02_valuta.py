"""Filtrazione, passo 2 [PREDITTIVO; C5 ESPLICATIVO]: analisi primaria e secondarie del piano.

Legge le previsioni OOF scritte da 01_addestra.py. Solo training, solo aggregati, test set non letto.
Uscite: analisi_impatto/04_esperimenti/filtrazione/*.csv
"""
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.config import CONFIG  # noqa: E402
from src.data.folds import OUTER  # noqa: E402
from src.data.imputed import load_fold  # noqa: E402
from src.data.kidney import egfr, is_female  # noqa: E402
from src.data.preprocess import FORBIDDEN  # noqa: E402
from src.data.split import PROCESSED  # noqa: E402
from src.models.evaluation import delong, holm  # noqa: E402

warnings.filterwarnings("ignore")
BASE = ROOT / "analisi_impatto" / "04_esperimenti" / "filtrazione"
OOF = BASE / "oof"
SEED = CONFIG["seed"]
FEATURES = CONFIG["features"]["numeric"] + CONFIG["features"]["categorical"]
SCORED = CONFIG["phase_a"]["scored_predictors"]
N_BOOT = 2000


def leggi(bersaglio, variante, modello, n):
    """(probabilità OOF, fold, parametri per fold) oppure None se manca un fold."""
    p, fold, params = np.full(n, np.nan), np.full(n, -1), []
    for k in range(OUTER):
        path = OOF / bersaglio / variante / "main" / f"{modello}_outer{k}.json"
        if not path.exists():
            return None
        r = json.loads(path.read_text(encoding="utf-8"))
        p[r["rows"]], fold[r["rows"]] = r["probability"], k
        params.append(r["params"])
    assert not np.isnan(p).any()
    return p, fold, params


def componenti(y, p):
    y = np.asarray(y) == 1
    pos, neg = p[y], p[~y]
    m, n = len(pos), len(neg)
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:m] - stats.rankdata(pos)) / n, 1 - (r[m:] - stats.rankdata(neg)) / m


def delong_appaiato(y, p1, p2):
    """Differenza di AUROC (p1 - p2) con IC 95% e p, DeLong et al. 1988 per curve correlate."""
    a10, a01 = componenti(y, p1)
    b10, b01 = componenti(y, p2)
    d = a10.mean() - b10.mean()
    se = np.sqrt(np.var(a10 - b10, ddof=1) / len(a10) + np.var(a01 - b01, ddof=1) / len(a01))
    return d, d - 1.96 * se, d + 1.96 * se, 2 * stats.norm.sf(abs(d) / se)


def logistica_oof(colonne, y, extra=None):
    """Logistica senza penalizzazione sugli stessi fold esterni imputati (come lr_scored)."""
    p = np.full(len(y), np.nan)
    for k in range(OUTER):
        X_tr, X_va = load_fold("main", k)
        A_tr, A_va = X_tr[colonne].to_numpy(), X_va[colonne].to_numpy()
        if extra is not None:
            A_tr = np.column_stack([A_tr, extra[X_tr.index]])
            A_va = np.column_stack([A_va, extra[X_va.index]])
        m = LogisticRegression(C=np.inf, max_iter=5000).fit(A_tr, y[X_tr.index])
        p[X_va.index] = m.predict_proba(A_va)[:, 1]
    return p


def scegli(score, fold, quota, rng):
    """Seleziona la quota più alta dentro ogni fold (pareggi rotti a caso)."""
    sel = np.zeros(len(score), bool)
    for k in np.unique(fold):
        idx = np.flatnonzero(fold == k)
        ordine = np.lexsort((rng.random(len(idx)), -score[idx]))
        sel[idx[ordine[: int(round(quota * len(idx)))]]] = True
    return sel


def parametri_tipici(params):
    """Mediana degli iperparametri scelti nei 5 fold esterni (interi arrotondati)."""
    out = {}
    for k in params[0]:
        v = [p[k] for p in params]
        out[k] = int(round(np.median(v))) if isinstance(v[0], int) else float(np.median(v))
    return out


def xgb(params):
    return XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED, **params)


def riga_auc(b, variante, nome, y, p, fold):
    auc, lo, hi = delong(y, p)
    per_fold = [roc_auc_score(y[fold == k], p[fold == k]) for k in range(OUTER)]
    return {"bersaglio": b, "variante": variante, "modello": nome, "positivi": int(y.sum()), "auroc": auc,
            "ic_basso": lo, "ic_alto": hi, "auroc_media_fold": np.mean(per_fold), "auroc_min_fold": np.min(per_fold),
            "auroc_max_fold": np.max(per_fold), "pr_auc": average_precision_score(y, p)}


def main():
    d = pd.read_csv(PROCESSED / "train.csv")
    n = len(d)
    yG = (egfr(d) < 60).astype(int).to_numpy()
    yA = (d["UMAUCR"] >= 30).astype(int).to_numpy()
    eta = d["Age"].to_numpy(float)
    assert not set(FEATURES) & set(FORBIDDEN)
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------ 1. discriminazione di tutto ciò che è pronto
    righe, P = [], {}
    for b, y in [("G", yG), ("A", yA)]:
        for variante in ["tutte", "senza_sua", "senza_eta"]:
            for modello in ["xgboost", "lr_penalized", "random_forest", "lr_scored", "tabpfn"]:
                r = leggi(b, variante, modello, n)
                if r is not None:
                    P[(b, variante, modello)] = r
                    righe.append(riga_auc(b, variante, modello, y, r[0], r[1]))
    fold = P[("G", "tutte", "lr_scored")][1]
    comp = {}
    for b, y in [("G", yG), ("A", yA)]:
        comp[(b, "C1 sola età")] = logistica_oof(["Age"], y)
        comp[(b, "C2 età + sesso")] = logistica_oof(["Age", "Gender"], y)
        comp[(b, "C4 SCORED-5 + SUA")] = logistica_oof(SCORED + ["SUA"], y)
        comp[(b, "età + sesso + SUA")] = logistica_oof(["Age", "Gender", "SUA"], y)
    comp[("G", "C5 SCORED-5 + albuminuria [ESPLICATIVO]")] = logistica_oof(SCORED, yG, extra=yA.astype(float))
    for (b, nome), p in comp.items():
        righe.append(riga_auc(b, "comparatore", nome, yG if b == "G" else yA, p, fold))
    pd.DataFrame(righe).to_csv(BASE / "F1_discriminazione.csv", index=False)
    print(pd.DataFrame(righe).round(3).to_string(index=False))
    if ("G", "tutte", "xgboost") not in P:
        print("\nXGBoost su G non ancora pronto: mi fermo qui.")
        return
    pX, _, parX = P[("G", "tutte", "xgboost")]
    pC3 = P[("G", "tutte", "lr_scored")][0]

    # ------------------------------------------------ 2. primaria e confronti secondari (DeLong appaiato)
    conf = [("PRIMARIA: XGBoost − C3 SCORED-5", pX, pC3, "primaria")]
    for nome in ["C1 sola età", "C2 età + sesso"]:
        conf.append((f"XGBoost − {nome}", pX, comp[("G", nome)], "secondaria"))
    conf.append(("C4 (SCORED-5 + SUA) − C3", comp[("G", "C4 SCORED-5 + SUA")], pC3, "secondaria"))
    conf.append(("XGBoost − C5 SCORED-5 + albuminuria [ESPLICATIVO]", pX,
                 comp[("G", "C5 SCORED-5 + albuminuria [ESPLICATIVO]")], "secondaria"))
    for variante, etichetta in [("senza_sua", "senza SUA"), ("senza_eta", "senza età")]:
        if ("G", variante, "xgboost") in P:
            pv = P[("G", variante, "xgboost")][0]
            conf.append((f"XGBoost − XGBoost {etichetta}", pX, pv, "secondaria"))
            conf.append((f"XGBoost {etichetta} − C3", pv, pC3, "secondaria"))
    for modello in ["lr_penalized", "random_forest", "tabpfn"]:
        if ("G", "tutte", modello) in P:
            conf.append((f"{modello} − C3 SCORED-5", P[("G", "tutte", modello)][0], pC3, "secondaria"))
    giorni = d["Data"].to_numpy()
    lista_giorni = np.unique(giorni)
    idx_giorno = {g: np.flatnonzero(giorni == g) for g in lista_giorni}
    campioni = [np.concatenate([idx_giorno[g] for g in rng.choice(lista_giorni, len(lista_giorni))]) for _ in range(N_BOOT)]
    rc = []
    for nome, a, b_, tipo in conf:
        diff, lo, hi, pv = delong_appaiato(yG, a, b_)
        boot = [roc_auc_score(yG[i], a[i]) - roc_auc_score(yG[i], b_[i]) for i in campioni if yG[i].sum() > 1]
        rc.append({"confronto": nome, "tipo": tipo, "diff_auroc": diff, "ic_basso": lo, "ic_alto": hi, "p": pv,
                   "ic_basso_per_giornata": np.percentile(boot, 2.5), "ic_alto_per_giornata": np.percentile(boot, 97.5)})
    rc = pd.DataFrame(rc)
    sec = rc["tipo"].eq("secondaria")
    rc.loc[sec, "p_holm"] = holm(rc.loc[sec, "p"].to_numpy())
    rc.to_csv(BASE / "F2_confronti.csv", index=False)
    print("\n", rc.round(4).to_string(index=False))

    # ------------------------------------------------ 3. sottogruppi (confondenti dell'acido urico)
    donna = is_female(d).to_numpy()
    gruppi = {"uomini": ~donna, "donne": donna,
              "beve regolarmente": d["Drinking"].eq(3).to_numpy(), "non beve regolarmente": d["Drinking"].isin([1, 2]).to_numpy(),
              "BMI < 24": (d["BMI"] < 24).to_numpy(), "BMI ≥ 24": (d["BMI"] >= 24).to_numpy(),
              "età < 60": eta < 60, "età ≥ 60": eta >= 60}
    rs = []
    for nome, m in gruppi.items():
        pos = int(yG[m].sum())
        if pos < 2:
            continue
        a, lo, hi = delong(yG[m], pX[m])
        c, clo, chi = delong(yG[m], pC3[m])
        rs.append({"gruppo": nome, "n": int(m.sum()), "positivi": pos, "auroc_xgboost": a, "ic_basso": lo, "ic_alto": hi,
                   "auroc_C3": c, "C3_ic_basso": clo, "C3_ic_alto": chi, "solo_descrittivo": pos < 20})
    pd.DataFrame(rs).to_csv(BASE / "F3_sottogruppi.csv", index=False)
    print("\n", pd.DataFrame(rs).round(3).to_string(index=False))

    # ------------------------------------------------ 5. a parità di persone mandate all'esame
    rp = []
    for quota in [0.05, 0.10, 0.20]:
        sel = {"XGBoost": scegli(pX, fold, quota, rng), "C3 SCORED-5": scegli(pC3, fold, quota, rng),
               "più anziani": scegli(eta, fold, quota, rng)}
        riga = {"quota": quota, "testati": int(sel["XGBoost"].sum()), "positivi_totali": int(yG.sum()),
                **{k: int(yG[v].sum()) for k, v in sel.items()}}
        for rif in ["C3 SCORED-5", "più anziani"]:
            dd = (sel["XGBoost"] & (yG == 1)).astype(int) - (sel[rif] & (yG == 1)).astype(int)
            boot = [dd[rng.integers(0, n, n)].sum() for _ in range(N_BOOT)]
            riga[f"diff_vs_{rif}"] = int(dd.sum())
            riga[f"ic_basso_vs_{rif}"], riga[f"ic_alto_vs_{rif}"] = np.percentile(boot, [2.5, 97.5])
            solo_a, solo_b = int((dd == 1).sum()), int((dd == -1).sum())
            riga[f"mcnemar_p_vs_{rif}"] = stats.binomtest(solo_a, solo_a + solo_b, 0.5).pvalue if solo_a + solo_b else 1.0
        rp.append(riga)
    pd.DataFrame(rp).to_csv(BASE / "F5_parita_di_esami.csv", index=False)
    print("\n", pd.DataFrame(rp).round(3).to_string(index=False))

    # ------------------------------------------------ 6. calibrazione e net benefit
    rk = []
    for nome, p in [("XGBoost", pX), ("C3 SCORED-5", pC3)]:
        q = np.clip(p, 1e-6, 1 - 1e-6)
        cal = LogisticRegression(C=np.inf).fit(np.log(q / (1 - q)).reshape(-1, 1), yG)
        riga = {"modello": nome, "pendenza": cal.coef_[0, 0], "intercetta": cal.intercept_[0],
                "rischio_medio_predetto": p.mean(), "prevalenza": yG.mean(), "brier": np.mean((p - yG) ** 2)}
        for t in [0.01, 0.02, 0.05]:
            tp, fp = ((p >= t) & (yG == 1)).sum(), ((p >= t) & (yG == 0)).sum()
            riga[f"net_benefit_{t}"] = tp / n - fp / n * t / (1 - t)
            riga[f"testa_tutti_{t}"] = yG.mean() - (1 - yG.mean()) * t / (1 - t)
        rk.append(riga)
    pd.DataFrame(rk).to_csv(BASE / "F6_calibrazione_utilita.csv", index=False)
    print("\n", pd.DataFrame(rk).round(4).to_string(index=False))

    # ------------------------------------------------ 8. etichetta con l'età fissata a 39 anni
    d39 = d.copy()
    d39["Age"] = 39
    y39 = (egfr(d39) < 60).astype(int).to_numpy()
    r8 = []
    for nome_y, y in [("G (eGFR < 60)", yG), ("G con età fissata a 39", y39)]:
        riga = {"etichetta": nome_y, "positivi": int(y.sum())}
        for nome, p in [("XGBoost", pX), ("C3 SCORED-5", pC3), ("sola età", eta)]:
            riga[f"auroc_{nome}"], riga[f"{nome}_ic_basso"], riga[f"{nome}_ic_alto"] = delong(y, p)
        r8.append(riga)
    pd.DataFrame(r8).to_csv(BASE / "F8_eta_fissata.csv", index=False)
    print("\n", pd.DataFrame(r8).round(3).to_string(index=False))

    # ------------------------------------------------ 4 e stabilità: iperparametri tipici, dati grezzi (XGBoost gestisce i mancanti)
    tip = parametri_tipici(parX)
    X = d[FEATURES].astype(float)
    rst = []
    for schema, semi, cv in [("5 fold stratificati", 10, StratifiedKFold), ("5 fold raggruppati per giornata", 5, StratifiedGroupKFold)]:
        for seme in range(semi):
            p = np.zeros(n)
            for tr, te in cv(5, shuffle=True, random_state=seme).split(X, yG, giorni):
                p[te] = xgb(tip).fit(X.iloc[tr], yG[tr]).predict_proba(X.iloc[te])[:, 1]
            rst.append({"schema": schema, "seme": seme, "auroc": roc_auc_score(yG, p)})
    rst = pd.DataFrame(rst)
    rst.to_csv(BASE / "F4_stabilita_e_giornate.csv", index=False)
    print("\niperparametri tipici:", tip)
    print(rst.groupby("schema")["auroc"].agg(["mean", "min", "max"]).round(3).to_string())

    # ------------------------------------------------ 9. SHAP (TreeSHAP di XGBoost) sui fold esterni, G e A
    rshap = []
    for b, y in [("G", yG), ("A", yA)]:
        if (b, "tutte", "xgboost") not in P:
            continue
        tot, colonne = 0, None
        for k in range(OUTER):
            X_tr, X_va = load_fold("main", k)
            m = xgb(P[(b, "tutte", "xgboost")][2][k]).fit(X_tr, y[X_tr.index])
            contrib = m.get_booster().predict(xgboost.DMatrix(X_va), pred_contribs=True)[:, :-1]
            tot, colonne = tot + np.abs(contrib).sum(0), list(X_tr.columns)
        s = pd.Series(tot / n, index=colonne).sort_values(ascending=False)
        for rango, (col, v) in enumerate(s.items(), 1):
            rshap.append({"bersaglio": b, "rango": rango, "variabile": col, "shap_medio_assoluto": v, "quota": v / s.sum()})
    rshap = pd.DataFrame(rshap)
    rshap.to_csv(BASE / "F9_shap.csv", index=False)
    for b in rshap["bersaglio"].unique():
        print(f"\nSHAP {b}:", ", ".join(f"{r.variabile} {r.quota:.1%}" for r in rshap[rshap.bersaglio == b].head(8).itertuples()))


if __name__ == "__main__":
    main()
