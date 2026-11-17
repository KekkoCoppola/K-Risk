"""Parte B del piano preregistrato: modelli non supervisionati su uno split (G o T).

[ESPLICATIVO] Gli ingressi contengono le misure che definiscono il target: l'accordo con
KDIGO non stima una capacità predittiva. Piano: `04_esperimenti/non_supervisionato/piano.md`,
B2–B6 e B8. CV 5×1 sui fold esterni del progetto; l'etichetta si usa solo nella valutazione.

Uso: python 03_parteB.py G
Uscite aggregate: `04_esperimenti/non_supervisionato/parteB/<split>/`.
Punteggi per paziente e modelli (per la parte C): `data/processed/non_supervisionato/<split>_out/`
(non versionati).
"""
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor

import prep

ROOT = prep.ROOT
from src.data import kidney  # noqa: E402
from src.models.evaluation import delong  # noqa: E402

SEED = 42
KS = range(1, 7)
N_REF = 20
FRACTIONS = [0.05, 0.10, 0.20]
N_BOOT = 1000


# ---------- algoritmi ----------

def gap_statistic(X, rng):
    """Tibshirani et al. 2001: riferimenti uniformi nel riquadro delle componenti principali."""
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ vt.T
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    refs = [rng.uniform(lo, hi, size=Z.shape) @ vt for _ in range(N_REF)]

    def logw(data, k):
        if k == 1:
            return np.log(((data - data.mean(axis=0)) ** 2).sum())
        return np.log(KMeans(k, n_init=20, random_state=SEED).fit(data).inertia_)

    rows = []
    for k in KS:
        ref = np.array([logw(r, k) for r in refs])
        rows.append({"k": k, "gap": ref.mean() - logw(Xc, k), "s": ref.std() * np.sqrt(1 + 1 / N_REF)})
    g = pd.DataFrame(rows)
    chosen = next((int(g.k[i]) for i in range(len(g) - 1) if g.gap[i] >= g.gap[i + 1] - g.s[i + 1]), int(g.k.iloc[-1]))
    return chosen, g


class KPrototypes:
    """Huang (1998): media per le numeriche, moda per le categoriche; γ = DS media delle numeriche."""

    def __init__(self, k, n_init=10, max_iter=50, seed=SEED):
        self.k, self.n_init, self.max_iter, self.seed = k, n_init, max_iter, seed

    def _dist(self, Xn, Xc, cn, cc):
        return ((Xn[:, None, :] - cn[None]) ** 2).sum(2) + self.gamma * (Xc[:, None, :] != cc[None]).sum(2)

    def fit(self, Xn, Xc):
        rng = np.random.default_rng(self.seed)
        self.gamma = float(Xn.std(axis=0).mean())
        best = None
        for _ in range(self.n_init):
            idx = rng.choice(len(Xn), self.k, replace=False)
            cn, cc = Xn[idx].copy(), Xc[idx].copy()
            for _ in range(self.max_iter):
                lab = self._dist(Xn, Xc, cn, cc).argmin(1)
                new_cn, new_cc = cn.copy(), cc.copy()
                for j in range(self.k):
                    m = lab == j
                    if m.any():
                        new_cn[j] = Xn[m].mean(0)
                        new_cc[j] = stats.mode(Xc[m], axis=0, keepdims=False).mode
                if np.allclose(new_cn, cn) and np.array_equal(new_cc, cc):
                    break
                cn, cc = new_cn, new_cc
            d = self._dist(Xn, Xc, cn, cc)
            lab = d.argmin(1)
            cost = d[np.arange(len(Xn)), lab].sum()
            if best is None or cost < best[0]:
                best = (cost, cn, cc, lab)
        self.cost_, self.cn, self.cc, self.labels_ = best
        return self

    def predict(self, Xn, Xc):
        return self._dist(Xn, Xc, self.cn, self.cc).argmin(1)

    def score(self, Xn, Xc):
        big = int(np.bincount(self.labels_, minlength=self.k).argmax())
        return np.sqrt(self._dist(Xn, Xc, self.cn[[big]], self.cc[[big]])[:, 0])


def gap_kprototypes(Xn, Xc, rng):
    """Gap statistic con riferimenti: numeriche uniformi nell'intervallo, categoriche dalle frequenze."""
    def ref():
        num = rng.uniform(Xn.min(0), Xn.max(0), size=Xn.shape)
        cat = np.column_stack([rng.choice(Xc[:, j], size=len(Xc)) for j in range(Xc.shape[1])])
        return num, cat

    def logw(a, b, k):
        if k == 1:
            m = KPrototypes(1, n_init=1).fit(a, b)
            return np.log(m.cost_)
        return np.log(KPrototypes(k).fit(a, b).cost_)

    refs = [ref() for _ in range(N_REF)]
    rows = []
    for k in KS:
        r = np.array([logw(a, b, k) for a, b in refs])
        rows.append({"k": k, "gap": r.mean() - logw(Xn, Xc, k), "s": r.std() * np.sqrt(1 + 1 / N_REF)})
    g = pd.DataFrame(rows)
    chosen = next((int(g.k[i]) for i in range(len(g) - 1) if g.gap[i] >= g.gap[i + 1] - g.s[i + 1]), int(g.k.iloc[-1]))
    return chosen, g


def fit_gmm(X, k, cov):
    return GaussianMixture(k, covariance_type=cov, reg_covar=1e-3, n_init=10, random_state=SEED).fit(X)


def gmm_score(model, X):
    """1 − probabilità a posteriori della componente più numerosa (atipicità)."""
    big = int(np.argmax(model.weights_))
    return 1 - model.predict_proba(X)[:, big]


def kmeans_score(model, X, train_labels):
    big = int(np.bincount(train_labels).argmax())
    return np.linalg.norm(X - model.cluster_centers_[big], axis=1)


def robust_cov(X):
    try:
        return MinCovDet(random_state=SEED).fit(X), "MinCovDet"
    except Exception:  # dichiarato nel piano: ripiego su Ledoit-Wolf
        return LedoitWolf().fit(X), "LedoitWolf"


def autoencoder(X):
    p = X.shape[1]
    return MLPRegressor(hidden_layer_sizes=(32, 8, 32), activation="relu", early_stopping=True,
                        validation_fraction=0.1, max_iter=500, random_state=SEED).fit(X, X), p


# ---------- valutazione ----------

def at_fraction(score, fold, y, frac):
    found = 0
    for k in np.unique(fold):
        idx = np.flatnonzero(fold == k)
        top = idx[np.argsort(-score[idx], kind="stable")[: int(round(frac * len(idx)))]]
        found += int(y[top].sum())
    return found


def segmented(x, s, w=None, grid=None):
    """s = a + b·x + c·max(0, x − τ): τ con l'errore quadratico minimo (pesi w = conteggi bootstrap)."""
    w = np.ones_like(x) if w is None else w
    H = np.maximum(0, x[:, None] - grid[None, :])
    sw, sx, sxx = w.sum(), w @ x, w @ (x * x)
    sh, shh, sxh = w @ H, w @ (H * H), (w * x) @ H
    sy, sxy, shy, syy = w @ s, (w * x) @ s, (w * s) @ H, w @ (s * s)
    T = len(grid)
    A = np.empty((T, 3, 3))
    A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = sw, sx, sh
    A[:, 1, 1], A[:, 1, 2], A[:, 2, 2] = sxx, sxh, shh
    A[:, 1, 0], A[:, 2, 0], A[:, 2, 1] = A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]
    b = np.stack([np.full(T, sy), np.full(T, sxy), shy], axis=1)
    beta = np.linalg.solve(A + 1e-9 * np.eye(3), b[..., None])[..., 0]
    sse = syy - (beta * b).sum(axis=1)
    i = int(np.argmin(sse))
    return grid[i], beta[i]


def age_sex_partition(age, female, k):
    if k <= 1:
        return np.zeros(len(age), int)
    if k % 2 == 0:
        q = pd.qcut(age, k // 2, labels=False, duplicates="drop")
        return np.asarray(q) * 2 + female.astype(int)
    return np.asarray(pd.qcut(age, k, labels=False, duplicates="drop"))


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))) if min(t.shape) > 1 else np.nan


# ---------- esecuzione ----------

def main(split):
    out = ROOT / f"analisi_impatto/04_esperimenti/non_supervisionato/parteB/{split}"
    store = prep.CACHE / f"{split}_out"
    out.mkdir(parents=True, exist_ok=True)
    store.mkdir(parents=True, exist_ok=True)
    cov = "full" if split == "G" else "diag"

    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    n = len(df)
    y = kidney.target(df).to_numpy().astype(bool)
    egfr = kidney.egfr(df).to_numpy()
    acr = df["UMAUCR"].to_numpy()
    level = kidney.kdigo_level(df).astype(str).to_numpy()
    female = kidney.is_female(df).to_numpy()
    age = df["Age"].to_numpy(dtype=float)
    dm = df["DM"].eq(1).to_numpy()
    day = pd.to_datetime(df["Data"]).rank(method="dense").astype(int).to_numpy() - 1
    numeric = prep.columns_G()[0] if split == "G" else prep.frame_T(df)[1]
    rng = np.random.default_rng(SEED)

    scores = {}      # configurazione -> punteggio OOF
    labels = {}      # configurazione -> etichetta OOF del cluster
    resid = {}       # configurazione -> punteggio OOF corretto per giornata
    fold = np.full(n, -1)
    choice, bic_rows, gap_rows, stab, timing = [], [], [], [], []
    all_assign = {}  # (algoritmo, k) -> lista di assegnazioni di tutti i 4.350 per fold

    def put(cfg, va, s_va, tr, s_tr, lab_va=None):
        scores.setdefault(cfg, np.full(n, np.nan))[va] = s_va
        if lab_va is not None:
            labels.setdefault(cfg, np.full(n, -1))[va] = lab_va
        D = np.eye(day.max() + 1)
        coef, *_ = np.linalg.lstsq(D[day[tr]], s_tr, rcond=None)
        resid.setdefault(cfg, np.full(n, np.nan))[va] = s_va - D[day[va]] @ coef

    for k_out in range(5):
        t0 = time.perf_counter()
        X_tr, X_va = prep.load(split, f"outer{k_out}")
        tr, va = X_tr.index.to_numpy(), X_va.index.to_numpy()
        fold[va] = k_out
        X_all = pd.concat([X_tr, X_va]).sort_index()
        cont = [c for c in X_tr.columns if c in numeric] if numeric else list(X_tr.columns)
        models = {}

        # misture gaussiane per sesso
        for sex in [False, True]:
            mtr, mall = female[tr] == sex, female[X_all.index] == sex
            A_tr = X_tr[cont].to_numpy()[mtr]
            fits = {}
            for k in KS:
                fits[k] = fit_gmm(A_tr, k, cov)
                bic_rows.append({"fold": k_out, "donne": sex, "k": k, "bic": fits[k].bic(A_tr)})
            kb = min(fits, key=lambda k: fits[k].bic(A_tr))
            choice.append({"fold": k_out, "algoritmo": f"gmm_{'donne' if sex else 'uomini'}", "k_scelto": kb})
            for k in KS:
                all_assign.setdefault((f"gmm_{int(sex)}", k), []).append(
                    fits[k].predict(X_all[cont].to_numpy()[mall]))
            models[f"gmm_{int(sex)}"] = {"bic": fits[kb], "k2": fits[2], "k4": fits[4]}
        for cfg in ["bic", "k2", "k4"]:
            s_va, s_tr = np.empty(len(va)), np.empty(len(tr))
            l_va = np.empty(len(va), int)
            for sex in [False, True]:
                m = models[f"gmm_{int(sex)}"][cfg]
                mv, mt = female[va] == sex, female[tr] == sex
                s_va[mv] = gmm_score(m, X_va[cont].to_numpy()[mv])
                s_tr[mt] = gmm_score(m, X_tr[cont].to_numpy()[mt])
                l_va[mv] = m.predict(X_va[cont].to_numpy()[mv]) + 10 * int(sex)
            put(f"gmm_{cfg}", va, s_va, tr, s_tr, l_va)

        # k-means con gap statistic
        kg, g = gap_statistic(X_tr.to_numpy(), rng)
        gap_rows.append(g.assign(fold=k_out))
        choice.append({"fold": k_out, "algoritmo": "kmeans", "k_scelto": kg})
        km = {k: KMeans(k, n_init=20, random_state=SEED).fit(X_tr.to_numpy()) for k in KS}
        for k in KS:
            all_assign.setdefault(("kmeans", k), []).append(km[k].predict(X_all.to_numpy()))
        for cfg, k in [("gap", kg), ("k2", 2), ("k4", 4)]:
            m = km[k]
            put(f"kmeans_{cfg}", va, kmeans_score(m, X_va.to_numpy(), m.labels_), tr,
                kmeans_score(m, X_tr.to_numpy(), m.labels_), m.predict(X_va.to_numpy()))
        models["kmeans"] = {"gap": km[kg], "k2": km[2], "k4": km[4]}

        # k-prototypes (solo T): numeriche e categoriche separate
        if split == "T":
            catc = [c for c in X_tr.columns if c not in cont]
            num_tr, cat_tr = X_tr[cont].to_numpy(), X_tr[catc].to_numpy().round(6)
            num_va, cat_va = X_va[cont].to_numpy(), X_va[catc].to_numpy().round(6)
            num_all, cat_all = X_all[cont].to_numpy(), X_all[catc].to_numpy().round(6)
            kp_k, gp = gap_kprototypes(num_tr, cat_tr, rng)
            gap_rows.append(gp.assign(fold=k_out, algoritmo="kprototypes"))
            choice.append({"fold": k_out, "algoritmo": "kprototypes", "k_scelto": kp_k})
            kp = {k: KPrototypes(k).fit(num_tr, cat_tr) for k in KS}
            for k in KS:
                all_assign.setdefault(("kprototypes", k), []).append(kp[k].predict(num_all, cat_all))
            for cfg, k in [("gap", kp_k), ("k2", 2), ("k4", 4)]:
                m = kp[k]
                put(f"kproto_{cfg}", va, m.score(num_va, cat_va), tr, m.score(num_tr, cat_tr),
                    m.predict(num_va, cat_va))
            models["kprototypes"] = {"gap": kp[kp_k], "k2": kp[2], "k4": kp[4], "categoriche": catc}

        # Mahalanobis robusta sulle continue
        mc, kind = robust_cov(X_tr[cont].to_numpy())
        choice.append({"fold": k_out, "algoritmo": "mahalanobis", "k_scelto": np.nan, "nota": kind})
        put("mahalanobis", va, mc.mahalanobis(X_va[cont].to_numpy()), tr, mc.mahalanobis(X_tr[cont].to_numpy()))
        models["mahalanobis"] = mc

        # autoencoder
        ae, _ = autoencoder(X_tr.to_numpy())
        err = lambda X: ((ae.predict(X) - X) ** 2).mean(axis=1)  # noqa: E731
        put("autoencoder", va, err(X_va.to_numpy()), tr, err(X_tr.to_numpy()))
        choice.append({"fold": k_out, "algoritmo": "autoencoder", "k_scelto": np.nan,
                       "nota": f"epoche {ae.n_iter_}"})
        models["autoencoder"] = ae

        joblib.dump({"models": models, "continuous": cont, "columns": list(X_tr.columns)},
                    store / f"modelli_outer{k_out}.joblib")
        timing.append({"fold": k_out, "secondi": round(time.perf_counter() - t0, 1)})
        print(f"{split} fold {k_out}: {timing[-1]['secondi']} s", flush=True)

    # stabilità fra fold (ARI medio fra le 10 coppie di modelli)
    for (alg, k), assigns in all_assign.items():
        if k == 1:
            continue
        aris = [adjusted_rand_score(assigns[i], assigns[j]) for i in range(5) for j in range(i + 1, 5)]
        stab.append({"algoritmo": alg, "k": k, "ari_medio": float(np.mean(aris)), "ari_min": float(np.min(aris))})

    # B5: confronto con KDIGO, solo ora
    alb = (acr >= 30) & (egfr >= 60)
    low = egfr < 60
    neg = ~y
    ev = []
    for cfg, s in scores.items():
        auc, lo, hi = delong(y, s)
        auc_r, lo_r, hi_r = delong(y, resid[cfg])
        fold_auc = [delong(y[fold == k], s[fold == k])[0] for k in range(5)]
        row = {"configurazione": cfg, "auroc": auc, "ic_basso": lo, "ic_alto": hi,
               "auroc_media_fold": float(np.mean(fold_auc)),
               "auroc_sola_albuminuria": delong(alb[alb | neg], s[alb | neg])[0],
               "auroc_egfr_sotto_60": delong(low[low | neg], s[low | neg])[0],
               "auroc_corretto_giornata": auc_r, "ic_basso_giornata": lo_r, "ic_alto_giornata": hi_r,
               "auroc_non_diabetici": delong(y[~dm], s[~dm])[0]}
        for f in FRACTIONS:
            row[f"positivi_al_{int(f * 100)}pct"] = at_fraction(s, fold, y, f)
        if cfg in labels:
            lab = labels[cfg]
            row["ari_y"] = float(np.mean([adjusted_rand_score(y[fold == k], lab[fold == k]) for k in range(5)]))
            row["nmi_y"] = float(np.mean([normalized_mutual_info_score(y[fold == k], lab[fold == k]) for k in range(5)]))
            row["nmi_kdigo"] = float(np.mean([normalized_mutual_info_score(level[fold == k], lab[fold == k]) for k in range(5)]))
            row["ari_kdigo"] = float(np.mean([adjusted_rand_score(level[fold == k], lab[fold == k]) for k in range(5)]))
            kk = len(np.unique(lab[fold == 0]))
            part = age_sex_partition(age, female, kk)
            row["ari_eta_sesso"] = float(np.mean([adjusted_rand_score(part[fold == k], lab[fold == k]) for k in range(5)]))
            row["cramer_v_giornata"] = float(np.mean([cramers_v(lab[fold == k], day[fold == k]) for k in range(5)]))
            row["kruskal_p_giornata"] = float(np.mean([
                stats.kruskal(*[day[(fold == k) & (lab == c)] for c in np.unique(lab[fold == k])]).pvalue
                if len(np.unique(lab[fold == k])) > 1 else np.nan for k in range(5)]))
        ev.append(row)
    ev = pd.DataFrame(ev)
    ev.to_csv(out / "B5_confronto_kdigo.csv", index=False)

    # prevalenza di y nei cluster (per fold)
    prev = []
    for cfg, lab in labels.items():
        for k in range(5):
            for c in np.unique(lab[fold == k]):
                m = (fold == k) & (lab == c)
                prev.append({"configurazione": cfg, "fold": k, "cluster": int(c), "n": int(m.sum()),
                             "prevalenza_y": float(y[m].mean()), "eta_mediana": float(np.median(age[m])),
                             "quota_donne": float(female[m].mean()), "quota_DM": float(dm[m].mean())})
    pd.DataFrame(prev).to_csv(out / "B5_cluster_prevalenza.csv", index=False)

    # B6: confini (punteggio standardizzato dentro il fold), IC bootstrap
    xs = {"log_acr": np.log(np.maximum(acr, 0.1)), "egfr": egfr}
    bounds = []
    bw = np.random.default_rng(SEED)
    for cfg, s in scores.items():
        z = s.copy()
        for k in range(5):
            m = fold == k
            z[m] = (s[m] - s[m].mean()) / s[m].std()
        for xname, x in xs.items():
            ok = np.isfinite(x) & np.isfinite(z)
            xx, zz = x[ok], z[ok]
            grid = np.percentile(xx, np.arange(5, 96, 1))
            tau, beta = segmented(xx, zz, grid=grid)
            boot = [segmented(xx, zz, w=bw.multinomial(len(xx), np.full(len(xx), 1 / len(xx))).astype(float),
                              grid=grid)[0] for _ in range(N_BOOT)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            conv = np.exp if xname == "log_acr" else (lambda v: v)
            ref = 30 if xname == "log_acr" else 60
            bounds.append({"configurazione": cfg, "variabile": "ACR" if xname == "log_acr" else "eGFR",
                           "punto_di_rottura": float(conv(tau)), "ic_basso": float(conv(lo)),
                           "ic_alto": float(conv(hi)), "riferimento_kdigo": ref,
                           "contiene_kdigo": bool(conv(lo) <= ref <= conv(hi)),
                           "pendenza_prima": float(beta[1]), "pendenza_dopo": float(beta[1] + beta[2])})
    pd.DataFrame(bounds).to_csv(out / "B6_confini.csv", index=False)

    pd.DataFrame(choice).to_csv(out / "B3_k_scelto.csv", index=False)
    pd.DataFrame(bic_rows).to_csv(out / "B3_bic.csv", index=False)
    pd.concat(gap_rows).to_csv(out / "B3_gap.csv", index=False)
    pd.DataFrame(stab).to_csv(out / "B3_stabilita.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "tempi.csv", index=False)
    np.savez_compressed(store / "punteggi_oof.npz", fold=fold,
                        **{f"s_{c}": v for c, v in scores.items()},
                        **{f"r_{c}": v for c, v in resid.items()},
                        **{f"l_{c}": v for c, v in labels.items()})
    print(ev[["configurazione", "auroc", "ic_basso", "ic_alto", "auroc_corretto_giornata",
              "positivi_al_10pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
