"""Parte A del piano preregistrato: modelli supervisionati contro la linea guida.

[PREDITTIVO] Robustezza sugli stessi dati del passo 0 (già visto), non conferma indipendente.
Piano: `analisi_impatto/04_esperimenti/non_supervisionato/piano.md`, parte A.
Solo training, 5 fold esterni; test set non letto. Uscite aggregate in
`analisi_impatto/04_esperimenti/non_supervisionato/parteA/`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data import kidney  # noqa: E402
from src.models.phase_a import xy  # noqa: E402
from src.models.zoo import build  # noqa: E402

ANALYTICS = ROOT / "analytics"
OUT = ROOT / "analisi_impatto/04_esperimenti/non_supervisionato/parteA"
SEED = 42
N_BOOT = 2000
REPLICA_SEEDS = list(range(1, 11))
MODELS = {"random_forest": "phase_a/main/random_forest", "tabpfn": "phase_d/tabpfn"}
PRIMARY = ("random_forest", "R-a_DM")


def record_paths(member_id):
    p = Path(member_id)
    a = [ANALYTICS / p.parent / f"{p.name}_outer{k}.json" for k in range(5)]
    if all(x.exists() for x in a):
        return a
    return [ANALYTICS / p / f"outer{k}.json" for k in range(5)]


def load_scores(member_id, n):
    score, fold = np.full(n, np.nan), np.full(n, -1)
    for k, path in enumerate(record_paths(member_id)):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows = np.asarray(r["rows"], dtype=int)
        score[rows] = np.asarray(r["probability" if "probability" in r else "score"], dtype=float)
        fold[rows] = k
    assert not np.isnan(score).any() and (fold >= 0).all()
    return score, fold


def select(score, fold, quota_mask, tiebreak, idx=None):
    """Dentro ogni fold i punteggi più alti, tanti quanti i True di quota_mask nel fold.
    idx: campione (anche con ripetizioni) su cui lavorare; restituisce un vettore di posizioni."""
    idx = np.arange(len(score)) if idx is None else idx
    chosen = []
    for k in range(5):
        sub = idx[fold[idx] == k]
        m = int(quota_mask[sub].sum())
        order = np.lexsort((tiebreak[sub], -score[sub]))
        chosen.append(sub[order[:m]])
    return np.concatenate(chosen)


def mask_of(pos, n):
    m = np.zeros(n, dtype=bool)
    m[pos] = True
    return m


def mcnemar(sel_m, sel_r, yy):
    b = int((sel_m & ~sel_r & yy).sum())
    c = int((~sel_m & sel_r & yy).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(running, 1.0)
    return adj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    n = len(df)
    y = kidney.target(df).to_numpy().astype(bool)
    egfr = kidney.egfr(df).to_numpy()
    acr = df["UMAUCR"].to_numpy()
    level = kidney.kdigo_level(df).astype(str).to_numpy()
    female = kidney.is_female(df).to_numpy()
    age = df["Age"].to_numpy(dtype=float)

    ht = df["HypertenHis"].fillna(0).eq(1).to_numpy()
    bp = ((df["Bpsys"] >= 140) | (df["Bpdia"] >= 90)).to_numpy()
    dm = df["DM"].eq(1).to_numpy()
    dmhis = df["DMhis"].eq(1).to_numpy()
    rules = {"R-a_DM": dm | ht, "R-b_DM": dm | ht | bp,
             "R-a_DMhis": dmhis | ht, "R-b_DMhis": dmhis | ht | bp}

    tiebreak = np.random.default_rng(SEED).permutation(n).astype(float)
    scores, fold = {}, None
    for name, mid in MODELS.items():
        scores[name], f = load_scores(mid, n)
        assert fold is None or np.array_equal(f, fold)
        fold = f

    # baseline: età; informazioni del clinico (logistica in CV sugli stessi fold)
    scores["solo_eta"] = age.copy()
    clin_cols = ["Age", "Gender", "DM", "HypertenHis", "Bpsys", "Bpdia"]
    Xc = df[clin_cols].copy()
    Xc["HypertenHis"] = Xc["HypertenHis"].fillna(0)
    Xc["Gender"] = female.astype(float)
    clin = np.full(n, np.nan)
    for k in range(5):
        tr, va = fold != k, fold == k
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(penalty=None, max_iter=2000))
        pipe.fit(Xc[tr], y[tr])
        clin[va] = pipe.predict_proba(Xc[va])[:, 1]
    scores["clinico_logistica"] = clin

    sels = {(m, r): mask_of(select(scores[m], fold, rule, tiebreak), n)
            for m in scores for r, rule in rules.items()}

    # A1 + A3: McNemar (primario; secondari e modello contro clinico con Holm)
    tests = []
    for m in ["random_forest", "tabpfn"]:
        for r, rule in rules.items():
            b, c, p = mcnemar(sels[(m, r)], rule, y)
            tests.append({"confronto": f"{m} vs regola", "modello": m, "regola": r,
                          "solo_modello": b, "solo_altro": c, "p": p,
                          "primario": (m, r) == PRIMARY})
    for r in rules:
        b, c, p = mcnemar(sels[("random_forest", r)], sels[("clinico_logistica", r)], y)
        tests.append({"confronto": "random_forest vs clinico_logistica", "modello": "random_forest",
                      "regola": r, "solo_modello": b, "solo_altro": c, "p": p, "primario": False})
    tests = pd.DataFrame(tests)
    sec = ~tests["primario"]
    tests["p_holm"] = np.nan
    tests.loc[sec, "p_holm"] = holm(tests.loc[sec, "p"])
    tests.loc[~sec, "p_holm"] = tests.loc[~sec, "p"]
    tests.to_csv(OUT / "A1_mcnemar.csv", index=False)

    # conteggi a parità per tutti i punteggi, regola e baseline
    found = []
    for m in scores:
        for r, rule in rules.items():
            s = sels[(m, r)]
            found.append({"punteggio": m, "regola": r, "testati": int(rule.sum()),
                          "pos_regola": int((rule & y).sum()), "pos_punteggio": int((s & y).sum()),
                          "pos_entrambi": int((s & rule & y).sum()),
                          "pos_unione": int(((s | rule) & y).sum()), "testati_unione": int((s | rule).sum())})
    found = pd.DataFrame(found)

    # A3.3: regola oppure i più anziani, stessa quota totale di "regola oppure RF"
    extra = []
    for r, rule in rules.items():
        union_rf = sels[("random_forest", r)] | rule
        target_n = np.zeros(n, dtype=bool)
        for k in range(5):
            idx = np.flatnonzero(fold == k)
            need = int(union_rf[idx].sum()) - int(rule[idx].sum())
            out = idx[~rule[idx]]
            order = np.lexsort((tiebreak[out], -age[out]))
            target_n[out[order[:need]]] = True
        sel = rule | target_n
        extra.append({"regola": r, "testati": int(sel.sum()),
                      "pos_regola_o_anziani": int((sel & y).sum()),
                      "pos_regola_o_rf": int((union_rf & y).sum())})
    pd.DataFrame(extra).to_csv(OUT / "A3_regola_o_anziani.csv", index=False)

    # A2: bootstrap con riselezione
    rng = np.random.default_rng(SEED)
    boots = []
    draws = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    for m in ["random_forest", "tabpfn", "clinico_logistica", "solo_eta"]:
        for r, rule in rules.items():
            d = np.empty(N_BOOT)
            for i, idx in enumerate(draws):
                pos_m = y[select(scores[m], fold, rule, tiebreak, idx)].sum()
                pos_r = (y[idx] & rule[idx]).sum()
                d[i] = pos_m - pos_r
            boots.append({"punteggio": m, "regola": r, "diff_media_boot": d.mean(),
                          "ic_basso": np.percentile(d, 2.5), "ic_alto": np.percentile(d, 97.5)})
    found = found.merge(pd.DataFrame(boots), on=["punteggio", "regola"], how="left")
    found["diff"] = found["pos_punteggio"] - found["pos_regola"]
    found.to_csv(OUT / "A2_conteggi_e_bootstrap.csv", index=False)

    # A2: repliche per seed del Random Forest (iperparametri della Fase A, cache imputata)
    ys = pd.Series(y.astype(int))
    rep = {s: np.full(n, np.nan) for s in [SEED, *REPLICA_SEEDS]}
    for k in range(5):
        params = json.loads((ANALYTICS / f"phase_a/main/random_forest_outer{k}.json").read_text())["params"]
        X_tr, y_tr, X_va, _ = xy("random_forest", "main", ys, k)
        for s in rep:
            model = build("random_forest", params).set_params(random_state=s)
            rep[s][X_va.index.to_numpy()] = model.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
    check = float(np.max(np.abs(rep[SEED] - scores["random_forest"])))
    rows = []
    for s in REPLICA_SEEDS:
        for r, rule in rules.items():
            sm = mask_of(select(rep[s], fold, rule, tiebreak), n)
            rows.append({"seed": s, "regola": r, "pos_rf": int((sm & y).sum()),
                         "diff": int((sm & y).sum() - (rule & y).sum())})
    reps = pd.DataFrame(rows)
    reps.to_csv(OUT / "A2_repliche_seed.csv", index=False)
    reps.groupby("regola")["diff"].agg(["mean", "min", "max"]).assign(
        controllo_seed42_max_diff=check).to_csv(OUT / "A2_repliche_sintesi.csv")

    # A4: componenti ed etichette alternative (stesse selezioni)
    egfr39 = kidney.egfr(df.assign(Age=39)).to_numpy()
    labels = {
        "composito": y,
        "sola_albuminuria": (acr >= 30) & (egfr >= 60),
        "egfr_sotto_60": egfr < 60,
        "eta_fissa_39": (acr >= 30) | (egfr39 < 60),
        "acr_45_o_egfr": (acr >= 45) | (egfr < 60),
        "acr_300_o_egfr": (acr >= 300) | (egfr < 60),
    }
    comp = []
    for m in ["random_forest", "tabpfn", "clinico_logistica", "solo_eta"]:
        for r, rule in rules.items():
            s = sels[(m, r)]
            for lab, yy in labels.items():
                b, c, p = mcnemar(s, rule, yy)
                comp.append({"punteggio": m, "regola": r, "etichetta": lab, "positivi": int(yy.sum()),
                             "pos_punteggio": int((s & yy).sum()), "pos_regola": int((rule & yy).sum()),
                             "solo_punteggio": b, "solo_regola": c, "p_mcnemar_descrittivo": p})
    pd.DataFrame(comp).to_csv(OUT / "A4_componenti.csv", index=False)

    # A5: sottogruppi (Random Forest, descrittivo)
    groups = {"tutti": np.ones(n, bool), "non_diabetici": ~dm, "eta_<50": age < 50,
              "eta_50-64": (age >= 50) & (age < 65), "eta_>=65": age >= 65,
              "donne": female, "uomini": ~female}
    sub = []
    for r, rule in rules.items():
        s = sels[("random_forest", r)]
        for g, gm in groups.items():
            sub.append({"regola": r, "gruppo": g, "soggetti": int(gm.sum()), "positivi": int((gm & y).sum()),
                        "testati_regola": int((gm & rule).sum()), "testati_rf": int((gm & s).sum()),
                        "pos_regola": int((gm & rule & y).sum()), "pos_rf": int((gm & s & y).sum())})
    pd.DataFrame(sub).to_csv(OUT / "A5_sottogruppi.csv", index=False)

    # A6: profili e SHAP (modello finale della Fase A, SHAP in-sample sull'intero training)
    z = np.load(ANALYTICS / "phase_a/interpretation/shap_random_forest_main.npz")
    cols = list(z["columns"])
    # X dello SHAP è standardizzata: l'allineamento per riga si controlla con la correlazione
    assert np.corrcoef(z["X"][:, cols.index("Age")], age)[0, 1] > 0.9999, "SHAP non allineato a train.csv"
    shap_abs = pd.DataFrame(np.abs(z["values"]), columns=cols)
    prof, shp = [], []
    for r in ["R-a_DM", "R-b_DM"]:
        rule, s = rules[r], sels[("random_forest", r)]
        for g, gm in {"solo_modello": s & ~rule, "solo_regola": ~s & rule,
                      "entrambi": s & rule, "nessuno": ~s & ~rule}.items():
            m = gm & y
            prof.append({"regola": r, "gruppo": g, "n": int(m.sum()),
                         "eta_mediana": float(np.median(age[m])), "quota_donne": float(female[m].mean()),
                         "quota_DM": float(dm[m].mean()), "quota_ipertensione_nota": float(ht[m].mean()),
                         "BMI_mediana": float(df.loc[m, "BMI"].median()),
                         "HbA1c_mediana": float(df.loc[m, "HbA1c"].median()),
                         "Bpsys_mediana": float(df.loc[m, "Bpsys"].median()),
                         "quota_sola_albuminuria": float(labels["sola_albuminuria"][m].mean()),
                         "quota_egfr_sotto_60": float(labels["egfr_sotto_60"][m].mean()),
                         "quota_acr_almeno_300": float((acr[m] >= 300).mean()),
                         **{f"quota_{l.replace(' ', '_')}": float((level[m] == l).mean()) for l in kidney.LEVELS}})
            imp = shap_abs[m].mean().sort_values(ascending=False)
            for rank, (feat, v) in enumerate(imp.items(), 1):
                shp.append({"regola": r, "gruppo": g, "rango": rank, "variabile": feat, "shap_assoluto_medio": v})
    pd.DataFrame(prof).to_csv(OUT / "A6_profili.csv", index=False)
    pd.DataFrame(shp).to_csv(OUT / "A6_shap_gruppi.csv", index=False)

    print(tests.round(4).to_string(index=False))
    print(found[["punteggio", "regola", "pos_regola", "pos_punteggio", "diff", "ic_basso", "ic_alto"]].to_string(index=False))
    print(pd.read_csv(OUT / "A2_repliche_sintesi.csv").to_string(index=False))


if __name__ == "__main__":
    main()
