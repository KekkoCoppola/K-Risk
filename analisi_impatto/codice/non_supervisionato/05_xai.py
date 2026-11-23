"""Parte C del piano preregistrato: XAI dei modelli non supervisionati (split G o T).

[ESPLICATIVO] Si spiega l'uscita del modello (punteggio o cluster), mai la CKD.
Metodi C1: profili, permutazione sul modello, surrogato XGBoost + TreeSHAP con interazioni
(deviazione dichiarata: XGBoost invece di HistGradientBoosting, perché calcola le interazioni
SHAP in modo nativo), contributi diretti di autoencoder e Mahalanobis, famiglie di variabili.
C2: criteri fissati nel piano. C3: stabilità fra fold, accordo fra metodi, controllo di sanità
(sul fold 0).

Uso: python 05_xai.py G
Uscite aggregate: `04_esperimenti/non_supervisionato/xai/<split>/`.
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, r2_score

import prep

ROOT = prep.ROOT
from src.config import CONFIG  # noqa: E402
from src.data import kidney  # noqa: E402
from src.data.preprocess import CATEGORICAL  # noqa: E402

_spec = importlib.util.spec_from_file_location("parteB", Path(__file__).parent / "03_parteB.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
# 03_parteB.py è stato lanciato come __main__: il pickle cerca la classe in __main__
KPrototypes = B.KPrototypes

SEED = 42
N_PERM = 10
N_SHAP = 1000
FIDELITY = 0.80
CONFIGS = ["gmm_bic", "gmm_k2", "gmm_k4", "kmeans_gap", "kmeans_k2", "kmeans_k4", "mahalanobis", "autoencoder"]
CLUSTER = {"gmm_bic", "gmm_k2", "gmm_k4", "kmeans_gap", "kmeans_k2", "kmeans_k4"}
FAMILIES = {
    "antropometria e parametri vitali": ["Age", "Height", "Weight", "BMI", "waist1", "hip", "WHR", "Fat", "HR", "Bpsys", "Bpdia"],
    "glicemia e insulina": ["GA", "FPG", "PG2h", "HbA1c", "FINS", "FCP", "INS2h", "CP2h", "HomaIR", "ISIGutt"],
    "fegato e proteine": ["ALT", "AST", "ASTALT", "GGT", "ALP", "TBIL", "DBIL", "TP", "ALB", "AG", "FIB4"],
    "lipidi e acido urico": ["CHOL", "TG", "HDL", "LDL", "SUA"],
    "misure renali": ["SCRE", "UmALB", "UCRE", "BUN"],
    "anamnesi, abitudini e sesso": list(CATEGORICAL),
}
SURR = dict(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=SEED, n_jobs=-1)


def family_of(col, split):
    if split == "T":
        if col in ("NO", "Data"):
            return "ordine e giornata"
        for g, cols in CONFIG["features"]["exclude"].items():
            if col in cols:
                return f"gruppo {g}"
    for f, cols in FAMILIES.items():
        if col in cols:
            return f
    return "emocromo" if split == "G" else "ingressi: emocromo"


def outputs(pack, cfg, X, female):
    """Punteggio e (se cluster) etichette del modello `cfg` sulle righe X."""
    mods, cont = pack["models"], pack["continuous"]
    if cfg.startswith("gmm"):
        key = cfg.split("_")[1]
        s, lab = np.empty(len(X)), np.empty(len(X), int)
        for sex in [False, True]:
            m = mods[f"gmm_{int(sex)}"][key]
            r = female == sex
            A = X[cont].to_numpy()[r]
            s[r] = 1 - m.predict_proba(A)[:, int(np.argmax(m.weights_))]
            lab[r] = m.predict(A) + 10 * int(sex)
        return s, lab
    if cfg.startswith("kproto"):
        kp = mods["kprototypes"]
        m, catc = kp[cfg.split("_")[1]], kp["categoriche"]
        Xn, Xc = X[cont].to_numpy(), X[catc].to_numpy().round(6)
        return m.score(Xn, Xc), m.predict(Xn, Xc)
    if cfg.startswith("kmeans"):
        m = mods["kmeans"][cfg.split("_")[1]]
        big = int(np.bincount(m.labels_).argmax())
        A = X.to_numpy()
        return np.linalg.norm(A - m.cluster_centers_[big], axis=1), m.predict(A)
    if cfg == "mahalanobis":
        return mods["mahalanobis"].mahalanobis(X[cont].to_numpy()), None
    ae = mods["autoencoder"]
    A = X.to_numpy()
    return ((ae.predict(A) - A) ** 2).mean(axis=1), None


def refit_pack(pack, X, female, cov):
    """Stessi algoritmi e stessi k del pacchetto originale, stimati su X (per il controllo di sanità)."""
    mods, cont = pack["models"], pack["continuous"]
    new = {}
    for sex in [False, True]:
        r = female == sex
        new[f"gmm_{int(sex)}"] = {key: B.fit_gmm(X[cont].to_numpy()[r], m.n_components, cov)
                                  for key, m in mods[f"gmm_{int(sex)}"].items()}
    new["kmeans"] = {key: KMeans(m.n_clusters, n_init=20, random_state=SEED).fit(X.to_numpy())
                     for key, m in mods["kmeans"].items()}
    if "kprototypes" in mods:
        catc = mods["kprototypes"]["categoriche"]
        Xn, Xc = X[cont].to_numpy(), X[catc].to_numpy().round(6)
        new["kprototypes"] = {key: B.KPrototypes(m.k).fit(Xn, Xc)
                              for key, m in mods["kprototypes"].items() if key != "categoriche"}
        new["kprototypes"]["categoriche"] = catc
    new["mahalanobis"] = B.robust_cov(X[cont].to_numpy())[0]
    new["autoencoder"] = B.autoencoder(X.to_numpy())[0]
    return {"models": new, "continuous": cont, "columns": pack["columns"]}


def perm_importance(pack, cfg, X, female, rng):
    s0, _ = outputs(pack, cfg, X, female)
    sd = s0.std() or 1.0
    res = {}
    for c in X.columns:
        d = []
        for _ in range(N_PERM):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            d.append(np.abs(outputs(pack, cfg, Xp, female)[0] - s0).mean() / sd)
        res[c] = float(np.mean(d))
    return pd.Series(res)


def native_contrib(pack, cfg, X):
    """Contributo per variabile: errore di ricostruzione (AE) o termine della distanza (Mahalanobis)."""
    if cfg == "autoencoder":
        A = X.to_numpy()
        return pd.Series(((pack["models"]["autoencoder"].predict(A) - A) ** 2).mean(axis=0), index=X.columns)
    mc, cont = pack["models"]["mahalanobis"], pack["continuous"]
    D = X[cont].to_numpy() - mc.location_
    return pd.Series((D * (D @ mc.precision_)).mean(axis=0), index=cont)


def surrogate(X_tr, t_tr, X_va, t_va, is_cluster):
    if is_cluster:
        classes, t_tr_c = np.unique(t_tr, return_inverse=True)
        if len(classes) < 2:
            return None, np.nan, None
        m = xgb.XGBClassifier(**SURR, objective="multi:softprob" if len(classes) > 2 else "binary:logistic")
        m.fit(X_tr, t_tr_c)
        mapping = {c: i for i, c in enumerate(classes)}
        known = np.isin(t_va, classes)
        pred = m.predict(X_va)
        fid = balanced_accuracy_score(np.array([mapping[c] for c in t_va[known]]), pred[known])
        return m, fid, len(classes)
    m = xgb.XGBRegressor(**SURR).fit(X_tr, t_tr)
    return m, r2_score(t_va, m.predict(X_va)), 1


def shap_tables(model, X):
    dm = xgb.DMatrix(X)
    b = model.get_booster()
    contrib = b.predict(dm, pred_contribs=True)
    inter = b.predict(dm, pred_interactions=True)
    if contrib.ndim == 3:  # multiclasse: media dei valori assoluti sulle classi
        imp = np.abs(contrib[:, :, :-1]).mean(axis=(0, 1))
        sv = contrib[:, :, :-1].mean(axis=1)
        it = np.abs(inter[:, :, :-1, :-1]).mean(axis=(0, 1))
    else:
        imp = np.abs(contrib[:, :-1]).mean(axis=0)
        sv = contrib[:, :-1]
        it = np.abs(inter[:, :-1, :-1]).mean(axis=0)
    return imp, sv, it


def main(split):
    out = ROOT / f"analisi_impatto/04_esperimenti/non_supervisionato/xai/{split}"
    out.mkdir(parents=True, exist_ok=True)
    store = prep.CACHE / f"{split}_out"
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    female = kidney.is_female(df).to_numpy()
    cov = "full" if split == "G" else "diag"
    rng = np.random.default_rng(SEED)
    global CONFIGS
    if split == "T":
        CONFIGS = CONFIGS + ["kproto_gap", "kproto_k2", "kproto_k4"]
        CLUSTER.update({"kproto_gap", "kproto_k2", "kproto_k4"})

    imp_rows, int_rows, fid_rows, prof_rows, dep_rows, sanity = [], [], [], [], [], []
    for k in range(5):
        pack = joblib.load(store / f"modelli_outer{k}.joblib")
        X_tr, X_va = prep.load(split, f"outer{k}")
        f_tr, f_va = female[X_tr.index], female[X_va.index]
        cols = list(X_va.columns)
        sub = rng.choice(len(X_va), size=min(N_SHAP, len(X_va)), replace=False)
        for cfg in CONFIGS:
            s_tr, l_tr = outputs(pack, cfg, X_tr, f_tr)
            s_va, l_va = outputs(pack, cfg, X_va, f_va)
            targets = [("punteggio", s_tr, s_va, False)]
            if cfg in CLUSTER:
                targets.append(("cluster", l_tr, l_va, True))

            # C1.2 permutazione sul modello stesso
            pi = perm_importance(pack, cfg, X_va, f_va, rng)
            for c, v in pi.items():
                imp_rows.append({"fold": k, "configurazione": cfg, "uscita": "punteggio", "metodo": "permutazione",
                                 "variabile": c, "importanza": v})
            if l_va is not None:
                for c in cols:
                    d_l = []
                    for _ in range(N_PERM):
                        Xp = X_va.copy()
                        Xp[c] = rng.permutation(Xp[c].to_numpy())
                        d_l.append(1 - adjusted_rand_score(l_va, outputs(pack, cfg, Xp, f_va)[1]))
                    imp_rows.append({"fold": k, "configurazione": cfg, "uscita": "cluster", "metodo": "permutazione",
                                     "variabile": c, "importanza": float(np.mean(d_l))})

            # C3 controllo di sanità (fold 0): colonne permutate in modo indipendente, modello ristimato
            if k == 0:
                # G: ristima per ogni configurazione (come eseguito il 06/11). T: una sola ristima
                # riusata da tutte le configurazioni, perché la ristima su 189 colonne è lenta
                # (deviazione dichiarata nel piano; stessa procedura, un solo estratto casuale).
                if split == "G" or cfg == CONFIGS[0]:
                    Xr_tr = X_tr.apply(lambda col: rng.permutation(col.to_numpy()))
                    Xr_va = X_va.apply(lambda col: rng.permutation(col.to_numpy()))
                    packr = refit_pack(pack, Xr_tr, f_tr, cov)
                pr = perm_importance(packr, cfg, Xr_va, f_va, rng)
                sanity.append({"configurazione": cfg, "importanza_max_originale": float(pi.max()),
                               "importanza_max_permutato": float(pr.max()),
                               "rapporto": float(pr.max() / pi.max()) if pi.max() else np.nan,
                               "passa": bool(pr.max() <= 0.5 * pi.max())})

            # C1.4 contributi diretti
            if cfg in ("autoencoder", "mahalanobis"):
                for c, v in native_contrib(pack, cfg, X_va).items():
                    imp_rows.append({"fold": k, "configurazione": cfg, "uscita": "punteggio", "metodo": "diretto",
                                     "variabile": c, "importanza": float(v)})

            # C1.1 profili dei cluster (d di Cohen sui dati standardizzati)
            if l_va is not None:
                for cl in np.unique(l_va):
                    m = l_va == cl
                    if m.sum() < 2 or (~m).sum() < 2:
                        continue
                    a, b = X_va[m], X_va[~m]
                    pooled = np.sqrt((a.var() * (len(a) - 1) + b.var() * (len(b) - 1)) / (len(X_va) - 2))
                    dd = (a.mean() - b.mean()) / pooled.replace(0, np.nan)
                    for c in cols:
                        prof_rows.append({"fold": k, "configurazione": cfg, "cluster": int(cl), "n": int(m.sum()),
                                          "variabile": c, "media_standardizzata": float(a[c].mean()),
                                          "d_cohen": float(dd[c])})

            # C1.3 surrogato + TreeSHAP con interazioni
            for name, t_tr, t_va, is_c in targets:
                model, fid, ncls = surrogate(X_tr, t_tr, X_va, t_va, is_c)
                fid_rows.append({"fold": k, "configurazione": cfg, "uscita": name, "fedelta": fid,
                                 "metrica": "accuratezza bilanciata" if is_c else "R2", "fedele": bool(fid >= FIDELITY)})
                if model is None:
                    continue
                rows_shap = sub if split == "G" else sub[:300]
                imp, sv, it = shap_tables(model, X_va.iloc[rows_shap])
                for c, v in zip(cols, imp):
                    imp_rows.append({"fold": k, "configurazione": cfg, "uscita": name, "metodo": "shap",
                                     "variabile": c, "importanza": float(v)})
                pairs = [(f"{cols[i]} x {cols[j]}", float(it[i, j] + it[j, i]))
                         for i, j in itertools.combinations(range(len(cols)), 2)]
                if split == "T":
                    pairs = sorted(pairs, key=lambda p: -p[1])[:200]
                for pname, v in pairs:
                    int_rows.append({"fold": k, "configurazione": cfg, "uscita": name, "coppia": pname, "interazione": v})
                if not is_c:
                    raw = df.iloc[X_va.index[rows_shap]]
                    for c in ["UmALB", "UCRE", "SCRE", "BUN", "Age"]:
                        if c in cols:
                            jx = cols.index(c)
                            dep_rows.append({"fold": k, "configurazione": cfg, "variabile": c,
                                             "spearman_valore_shap": float(stats.spearmanr(raw[c], sv[:, jx], nan_policy="omit")[0])})
        print(f"{split} XAI fold {k} fatto", flush=True)

    imp = pd.DataFrame(imp_rows)
    imp["famiglia"] = imp["variabile"].map(lambda c: family_of(c, split))
    imp.to_csv(out / "C1_importanze.csv", index=False)
    inter = pd.DataFrame(int_rows)
    inter.to_csv(out / "C1_interazioni.csv", index=False)
    fid = pd.DataFrame(fid_rows)
    fid.to_csv(out / "C1_fedelta_surrogati.csv", index=False)
    pd.DataFrame(prof_rows).to_csv(out / "C1_profili_cluster.csv", index=False)
    dep = pd.DataFrame(dep_rows)
    dep.to_csv(out / "C1_direzione_shap.csv", index=False)
    pd.DataFrame(sanity).to_csv(out / "C3_controllo_sanita.csv", index=False)

    # famiglie (somma delle importanze)
    fam = imp.groupby(["fold", "configurazione", "uscita", "metodo", "famiglia"])["importanza"].sum().reset_index()
    fam["quota"] = fam["importanza"] / fam.groupby(["fold", "configurazione", "uscita", "metodo"])["importanza"].transform("sum")
    fam.to_csv(out / "C1_famiglie.csv", index=False)

    # C2 criteri (punteggio, SHAP del surrogato, per fold; vale se ≥ 4 fold su 5 e surrogato fedele)
    crit = []
    sh = imp[(imp.metodo == "shap") & (imp.uscita == "punteggio")]
    it = inter[inter.uscita == "punteggio"]
    for cfg in CONFIGS:
        for k in range(5):
            ok = fid[(fid.fold == k) & (fid.configurazione == cfg) & (fid.uscita == "punteggio")]["fedele"].any()
            s = sh[(sh.fold == k) & (sh.configurazione == cfg)].sort_values("importanza", ascending=False)
            top5, top = s.variabile.head(5).tolist(), s.set_index("variabile")["importanza"]
            ii = it[(it.fold == k) & (it.configurazione == cfg)].sort_values("interazione", ascending=False).coppia.tolist()
            d = dep[(dep.fold == k) & (dep.configurazione == cfg)].set_index("variabile")["spearman_valore_shap"]
            pair = lambda a, b: f"{a} x {b}" if f"{a} x {b}" in ii else f"{b} x {a}"  # noqa: E731
            r = {"fold": k, "configurazione": cfg, "surrogato_fedele": bool(ok)}
            if split == "G":
                r["variabili_del_clinico"] = sum(c in top5 for c in ["SCRE", "UmALB", "UCRE"]) >= 2
                r["rapporto_acr"] = bool((pair("UmALB", "UCRE") in ii[:5]) and d.get("UmALB", 0) > 0 and d.get("UCRE", 0) < 0)
                r["correzione_egfr"] = bool((pair("SCRE", "Age") in ii[:10]) or (pair("SCRE", "Gender") in ii[:10]))
                r["bun_prima_di_scre"] = bool(top.get("BUN", 0) > top.get("SCRE", 0))
            fam_k = fam[(fam.fold == k) & (fam.configurazione == cfg) & (fam.uscita == "punteggio") & (fam.metodo == "shap")]
            fam_rank = fam_k.sort_values("quota", ascending=False).famiglia.tolist()
            r["famiglia_prima"] = fam_rank[0] if fam_rank else None
            if split == "T":
                r["peso_su_target_codificato"] = bool(fam_rank and fam_rank[0] == "gruppo leakage")
                r["gruppi_sono_giornate"] = "ordine e giornata" in fam_rank[:3]
            crit.append(r)
    crit = pd.DataFrame(crit)
    crit.to_csv(out / "C2_criteri_per_fold.csv", index=False)
    flags = [c for c in crit.columns if crit[c].dtype == bool and c != "surrogato_fedele"]
    summary = []
    for cfg, g in crit.groupby("configurazione"):
        row = {"configurazione": cfg, "fold_con_surrogato_fedele": int(g.surrogato_fedele.sum())}
        for f in flags:
            n_ok = int((g[f] & g.surrogato_fedele).sum())
            row[f"{f}_fold"] = n_ok
            row[f] = n_ok >= 4
        row["famiglia_prima_piu_frequente"] = g.famiglia_prima.mode().iat[0] if g.famiglia_prima.notna().any() else None
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "C2_criteri_sintesi.csv", index=False)

    # C3 affidabilità: stabilità fra fold e accordo fra metodi
    rel = []
    for (cfg, uscita, metodo), g in imp.groupby(["configurazione", "uscita", "metodo"]):
        piv = g.pivot_table(index="variabile", columns="fold", values="importanza")
        taus = [stats.kendalltau(piv[a], piv[b])[0] for a, b in itertools.combinations(piv.columns, 2)]
        tops = [set(piv[c].nlargest(10).index) for c in piv.columns]
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
        rel.append({"configurazione": cfg, "uscita": uscita, "metodo": metodo,
                    "tau_kendall_medio": float(np.nanmean(taus)), "jaccard_top10_medio": float(np.mean(jac))})
    agree = []
    mean_imp = imp.groupby(["configurazione", "uscita", "metodo", "variabile"])["importanza"].mean().reset_index()
    for (cfg, uscita), g in mean_imp.groupby(["configurazione", "uscita"]):
        piv = g.pivot_table(index="variabile", columns="metodo", values="importanza")
        for a, b in itertools.combinations(piv.columns, 2):
            ok = piv[[a, b]].dropna()
            agree.append({"configurazione": cfg, "uscita": uscita, "metodo_1": a, "metodo_2": b,
                          "spearman": float(stats.spearmanr(ok[a], ok[b])[0])})
    pd.DataFrame(rel).to_csv(out / "C3_stabilita.csv", index=False)
    pd.DataFrame(agree).to_csv(out / "C3_accordo_metodi.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(sanity).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
