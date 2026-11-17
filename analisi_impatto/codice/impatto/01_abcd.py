"""Controlli sui dati per i quattro contributi candidati (A, B, C, D). Solo training, solo aggregati.

A [PREDITTIVO] Screening a parità di testati: Random Forest della Fase A contro la regola
  "diabete o ipertensione" (R-a) e contro regole che aggiungono l'età (> 55, > 60).
  Persone da testare per trovare un malato (NNS) e McNemar esatto sui discordanti.
B [PREDITTIVO] Chi la linea guida non testa: soggetti senza `DM` e senza `HypertenHis`.
  Quanti malati, di che tipo, e quanto li trovano il Random Forest, l'età e la pressione misurata.
C [ESPLICATIVO, qualità del dato] Prevalenza per giornata di raccolta (`Data`) e misure renali per
  giornata.
D [ESPLICATIVO] Zona grigia dell'albuminuria: soggetti con eGFR ≥ 60 e ACR 13–30 mg/g contro
  ACR < 13 mg/g (tutti negativi per KDIGO). Profilo degli esami in ingresso, grezzo e corretto
  per età e sesso.

Uscite: `analisi_impatto/04_esperimenti/impatto/`. Script `analisi_impatto/codice/impatto/01_abcd.py`.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data import kidney  # noqa: E402
from src.data.preprocess import CATEGORICAL, NUMERIC  # noqa: E402
from src.models.evaluation import delong  # noqa: E402

_spec = importlib.util.spec_from_file_location("parteA", ROOT / "analisi_impatto/codice/non_supervisionato/01_parteA.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

OUT = ROOT / "analisi_impatto/04_esperimenti/impatto"
SEED = 42


def loglik(X, y):
    """Log-verosimiglianza di una logistica senza penalità (sostituisce statsmodels, non installato)."""
    m = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ols_coef(X, z, j):
    """Coefficiente j e sua p (t di Student) da minimi quadrati con intercetta."""
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    res = z - X @ beta
    dof = len(z) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (res @ res) / dof)
    t = beta[j + 1] / se[j + 1]
    return float(beta[j + 1]), float(2 * stats.t.sf(abs(t), dof))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    n = len(df)
    y = kidney.target(df).to_numpy().astype(bool)
    egfr = kidney.egfr(df).to_numpy()
    acr = pd.to_numeric(df["UMAUCR"], errors="coerce").to_numpy()
    age = df["Age"].to_numpy(dtype=float)
    female = kidney.is_female(df).to_numpy()
    dm = df["DM"].eq(1).to_numpy()
    ht = df["HypertenHis"].fillna(0).eq(1).to_numpy()
    bp = ((df["Bpsys"] >= 140) | (df["Bpdia"] >= 90)).to_numpy()
    rf, fold = A.load_scores("phase_a/main/random_forest", n)
    tiebreak = np.random.default_rng(SEED).permutation(n).astype(float)
    alb_only = (acr >= 30) & (egfr >= 60)
    low_gfr = egfr < 60

    # ---------------- A ----------------
    rules = {"R-a: diabete o ipertensione": dm | ht,
             "R-a + età > 55": dm | ht | (age > 55),
             "R-a + età > 60": dm | ht | (age > 60),
             "R-b: R-a + pressione misurata ≥ 140/90": dm | ht | bp}
    rows = []
    for name, rule in rules.items():
        sel_rf = A.mask_of(A.select(rf, fold, rule, tiebreak), n)
        sel_age = A.mask_of(A.select(age, fold, rule, tiebreak), n)
        b, c, p = A.mcnemar(sel_rf, rule, y)
        rows.append({"regola": name, "testati": int(rule.sum()), "quota_testati": rule.mean(),
                     "malati_regola": int((rule & y).sum()), "malati_rf": int((sel_rf & y).sum()),
                     "malati_solo_eta": int((sel_age & y).sum()),
                     "nns_regola": rule.sum() / max((rule & y).sum(), 1), "nns_rf": sel_rf.sum() / max((sel_rf & y).sum(), 1),
                     "solo_rf": b, "solo_regola": c, "p_mcnemar": p,
                     "malati_eGFR60_regola": int((rule & low_gfr).sum()), "malati_eGFR60_rf": int((sel_rf & low_gfr).sum())})
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "A_parita_di_testati.csv", index=False)
    print("== A\n", a.round(3).to_string(index=False))

    # ---------------- B ----------------
    out = ~(dm | ht)
    yo = y[out]
    desc = {"soggetti_fuori_regola": int(out.sum()), "quota_popolazione": out.mean(),
            "malati_fuori_regola": int(yo.sum()), "quota_dei_malati": yo.sum() / y.sum(),
            "prevalenza_fuori_regola": yo.mean(), "prevalenza_dentro_regola": y[~out].mean(),
            "malati_fuori_sola_albuminuria": int((out & y & alb_only).sum()),
            "malati_fuori_eGFR_sotto_60": int((out & y & low_gfr).sum()),
            "eta_mediana_malati_fuori": float(np.median(age[out & y])),
            "eta_mediana_sani_fuori": float(np.median(age[out & ~y])),
            "pressione_alta_misurata_malati_fuori": bp[out & y].mean(),
            "pressione_alta_misurata_sani_fuori": bp[out & ~y].mean()}
    pd.DataFrame([desc]).to_csv(OUT / "B_fuori_linea_guida_descrizione.csv", index=False)
    print("== B\n", pd.Series(desc).round(3).to_string())
    rowsb = []
    idx_out = np.flatnonzero(out)
    for name, s in [("Random Forest", rf), ("solo età", age), ("pressione sistolica", df["Bpsys"].to_numpy(dtype=float))]:
        ok = out & ~np.isnan(s)
        auc, lo, hi = delong(y[ok].astype(int), s[ok])
        r = {"punteggio": name, "auroc_fuori_regola": auc, "ic_basso": lo, "ic_alto": hi}
        for frac in (0.10, 0.20):
            quota = np.zeros(n, bool)
            for k in range(5):
                idx = np.flatnonzero(out & (fold == k))
                quota[idx[: int(round(frac * len(idx)))]] = True  # quanti testare per fold
            sel = A.mask_of(A.select(np.nan_to_num(s, nan=-np.inf), fold, quota, tiebreak, idx=idx_out), n)
            r[f"testati_{int(frac * 100)}pct_fuori_regola"] = int(sel.sum())
            r[f"malati_trovati_{int(frac * 100)}pct_fuori_regola"] = int((sel & y).sum())
        rowsb.append(r)
    bp_out = out & bp
    sel_rf = A.mask_of(A.select(rf, fold, bp_out, tiebreak, idx=idx_out), n)
    sel_age = A.mask_of(A.select(age, fold, bp_out, tiebreak, idx=idx_out), n)
    b, c, p = A.mcnemar(sel_rf, bp_out, y)
    b2, c2, p2 = A.mcnemar(sel_rf, sel_age, y)
    rowsb.append({"punteggio": "regola pressione misurata ≥ 140/90 contro RF a parità (fuori regola)",
                  "testati_regola_pressione": int(bp_out.sum()), "malati_regola_pressione": int((bp_out & y).sum()),
                  "malati_rf_parita": int((sel_rf & y).sum()), "malati_eta_parita": int((sel_age & y).sum()),
                  "solo_rf": b, "solo_regola": c, "p_mcnemar": p,
                  "rf_contro_eta_solo_rf": b2, "rf_contro_eta_solo_eta": c2, "p_mcnemar_rf_contro_eta": p2})
    bdf = pd.DataFrame(rowsb)
    bdf.to_csv(OUT / "B_fuori_linea_guida_modelli.csv", index=False)
    print(bdf.round(3).to_string(index=False))

    # ---------------- C ----------------
    day = pd.to_datetime(df["Data"]).dt.date.astype(str).to_numpy()
    num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy()  # noqa: E731
    g = pd.DataFrame({"giornata": day, "y": y, "acr30": acr >= 30, "egfr60": low_gfr, "eta": age, "donna": female,
                      "UCRE": num("UCRE"), "UmALB": num("UmALB"), "SCRE": num("SCRE"), "DM": dm})
    per = g.groupby("giornata").agg(n=("y", "size"), prevalenza=("y", "mean"), quota_acr30=("acr30", "mean"),
                                   quota_egfr60=("egfr60", "mean"), eta_mediana=("eta", "median"),
                                   quota_donne=("donna", "mean"), quota_DM=("DM", "mean"),
                                   UCRE_mediana=("UCRE", "median"), UmALB_mediana=("UmALB", "median"),
                                   SCRE_mediana=("SCRE", "median")).reset_index()
    per.to_csv(OUT / "C_per_giornata.csv", index=False)
    big = per[per.n >= 30].reset_index(drop=True)
    kw = lambda col: stats.kruskal(*[x.dropna() for _, x in g.groupby("giornata")[col]])[1]  # noqa: E731
    tests = {"giornate": len(per), "giornate_con_almeno_30": len(big),
             "prevalenza_min_giornate_30": big.prevalenza.min(), "prevalenza_max_giornate_30": big.prevalenza.max(),
             "chi2_p_prevalenza": stats.chi2_contingency(pd.crosstab(g.giornata, g.y))[1],
             "chi2_p_acr30": stats.chi2_contingency(pd.crosstab(g.giornata, g.acr30))[1],
             "chi2_p_egfr60": stats.chi2_contingency(pd.crosstab(g.giornata, g.egfr60))[1],
             "kruskal_p_UCRE": kw("UCRE"), "kruskal_p_UmALB": kw("UmALB"), "kruskal_p_SCRE": kw("SCRE"),
             "kruskal_p_eta": kw("eta"),
             "spearman_prevalenza_UCRE_mediana": stats.spearmanr(big.prevalenza, big.UCRE_mediana)[0],
             "spearman_prevalenza_UmALB_mediana": stats.spearmanr(big.prevalenza, big.UmALB_mediana)[0],
             "spearman_prevalenza_eta_mediana": stats.spearmanr(big.prevalenza, big.eta_mediana)[0],
             "spearman_prevalenza_ordine_giornata": stats.spearmanr(np.arange(len(big)), big.prevalenza)[0]}
    X0 = pd.DataFrame({"eta": age, "donna": female.astype(float), "DM": dm.astype(float)})
    D = pd.get_dummies(pd.Series(day), drop_first=True, dtype=float)
    X0s = (X0 - X0.mean()) / X0.std()
    lr = 2 * (loglik(pd.concat([X0s, D], axis=1).to_numpy(), y.astype(float)) - loglik(X0s.to_numpy(), y.astype(float)))
    tests.update({"lrt_giornata_oltre_eta_sesso_DM_chi2": lr, "lrt_df": D.shape[1], "lrt_p": stats.chi2.sf(lr, D.shape[1])})
    pd.DataFrame([tests]).to_csv(OUT / "C_test.csv", index=False)
    print("== C\n", pd.Series(tests).round(4).to_string())

    # ---------------- D ----------------
    base = egfr >= 60
    gl, gh = base & (acr < 13), base & (acr >= 13) & (acr < 30)
    rowsd = []
    for c in NUMERIC + list(CATEGORICAL):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        a_, b_ = v[gl & ~np.isnan(v)], v[gh & ~np.isnan(v)]
        if len(a_) < 10 or len(b_) < 10 or np.nanstd(v[gl | gh]) == 0:
            continue
        sd = np.sqrt(((len(a_) - 1) * a_.var(ddof=1) + (len(b_) - 1) * b_.var(ddof=1)) / (len(a_) + len(b_) - 2))
        ok = (gl | gh) & ~np.isnan(v)
        Xr = np.column_stack([gh[ok].astype(float), age[ok], female[ok].astype(float)])
        z = (v[ok] - v[ok].mean()) / v[ok].std()
        coef, pc = ols_coef(Xr, z, 0)
        rowsd.append({"variabile": c, "n_sotto_13": len(a_), "n_13_30": len(b_),
                      "media_sotto_13": a_.mean(), "media_13_30": b_.mean(), "d_cohen": (b_.mean() - a_.mean()) / sd,
                      "p_mann_whitney": stats.mannwhitneyu(b_, a_).pvalue,
                      "diff_corretta_eta_sesso_in_ds": coef, "p_corretta": pc})
    d = pd.DataFrame(rowsd)
    d["p_holm"] = A.holm(d.p_mann_whitney)
    d["p_corretta_holm"] = A.holm(d.p_corretta)
    d = d.sort_values("p_corretta")
    d.to_csv(OUT / "D_zona_grigia_profilo.csv", index=False)
    info = {"n_sotto_13": int(gl.sum()), "n_13_30": int(gh.sum()), "eta_mediana_sotto_13": float(np.median(age[gl])),
            "eta_mediana_13_30": float(np.median(age[gh])), "quota_donne_sotto_13": female[gl].mean(),
            "quota_donne_13_30": female[gh].mean(),
            "variabili_significative_holm_grezze": int((d.p_holm < 0.05).sum()),
            "variabili_significative_holm_corrette": int((d.p_corretta_holm < 0.05).sum()), "variabili_testate": len(d)}
    pd.DataFrame([info]).to_csv(OUT / "D_zona_grigia_gruppi.csv", index=False)
    print("== D\n", pd.Series(info).round(3).to_string())
    print(d.head(15)[["variabile", "media_sotto_13", "media_13_30", "d_cohen", "p_holm", "diff_corretta_eta_sesso_in_ds",
                      "p_corretta_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
