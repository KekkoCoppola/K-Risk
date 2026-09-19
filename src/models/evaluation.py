"""Valutazione della Fase A sulle previsioni out-of-fold del training (domande 1-4 dello Scope).

1. discriminazione: AUC e PR-AUC; precision, recall, specificità e quota da testare alla soglia
2. probabilità per livello KDIGO e test di tendenza di Jonckheere-Terpstra
3. sensibilità per livello KDIGO (moderato, alto, molto alto) alla soglia
4. fasce di rischio del modello contro livelli KDIGO: tabella e kappa pesato
Poi il confronto appaiato fra modelli sui 5 fold esterni (Nadeau & Bengio 2003) e l'analisi di
sensibilità no_consequence contro main.

Legge solo train.csv (target e livello KDIGO) e le previsioni out-of-fold: il test set non viene
letto. Soglia e fasce stimate qui si applicheranno una volta sola al test, alla fine.
Scelte fissate prima dei risultati: Notepad, sezione "Fase A — valutazione".
"""
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score, roc_curve

from src.config import CONFIG, resolve
from src.data.kidney import LEVELS, kdigo_level, target
from src.data.split import PROCESSED

SETTINGS = CONFIG["evaluation"]
OUTPUT = resolve(SETTINGS["output"])
PREDICTIONS = resolve(CONFIG["phase_a"]["output"]) / "oof_predictions.csv"
ALPHA = SETTINGS["alpha"]
SEED = CONFIG["seed"]
METRICS = {"auc": roc_auc_score, "pr_auc": average_precision_score}


# --- intervalli di confidenza -------------------------------------------------------------

def z_value(alpha=ALPHA):
    return stats.norm.ppf(1 - alpha / 2)


def wilson(k, n, alpha=ALPHA):
    """Proporzione k / n con intervallo di Wilson (Brown, Cai & DasGupta 2001)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    ci = stats.binomtest(int(k), int(n)).proportion_ci(1 - alpha, method="wilson")
    return k / n, ci.low, ci.high


def delong(y, p, alpha=ALPHA):
    """AUC con intervallo di DeLong et al. 1988; componenti calcolate con i ranghi medi."""
    y = np.asarray(y) == 1
    p = np.asarray(p, dtype=float)
    pos, neg = p[y], p[~y]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    # per ogni positivo: quota di negativi con p più bassa; per ogni negativo: quota di positivi
    # con p più alta (pareggi contati 1/2)
    v10 = (ranks[:m] - stats.rankdata(pos)) / n
    v01 = 1 - (ranks[m:] - stats.rankdata(neg)) / m
    auc = v10.mean()
    se = np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    z = z_value(alpha)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def pr_auc(y, p, alpha=ALPHA):
    """PR-AUC (average precision) con intervallo logit, n = numero di positivi (Boyd et al. 2013)."""
    ap = average_precision_score(y, p)
    if not 0 < ap < 1:
        return ap, np.nan, np.nan
    tau = 1 / np.sqrt(np.sum(y) * ap * (1 - ap))
    z = z_value(alpha)
    return ap, expit(logit(ap) - z * tau), expit(logit(ap) + z * tau)


def stratified_indices(strata, n_boot, seed=SEED):
    """Campioni bootstrap estratti dentro ogni livello KDIGO: le numerosità dei livelli restano
    quelle osservate, anche per i 21 "molto alto" (bootstrap stratificato, Boyd et al. 2013)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(g, len(g)) for g in groups])


def percentile_interval(values, alpha=ALPHA):
    return tuple(np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


# --- soglia, fasce, tendenza --------------------------------------------------------------

def threshold_at_sensitivity(y, p, target_sensitivity):
    """Soglia più alta con sensibilità almeno pari al valore fissato (classe 1 se p >= soglia)."""
    scores = np.sort(np.asarray(p)[np.asarray(y) == 1])[::-1]
    k = int(np.ceil(target_sensitivity * len(scores) - 1e-9))
    return scores[k - 1]


def threshold_youden(y, p):
    """Soglia che massimizza sensibilità + specificità - 1 (Youden 1950)."""
    fpr, tpr, thresholds = roc_curve(y, p)
    return thresholds[np.argmax(tpr - fpr)]


def threshold(y, p, rule=SETTINGS["threshold_rule"], target_sensitivity=SETTINGS["target_sensitivity"]):
    if rule == "sensitivity":
        return threshold_at_sensitivity(y, p, target_sensitivity)
    if rule == "youden":
        return threshold_youden(y, p)
    raise ValueError(f"regola di soglia sconosciuta: {rule}")


def operating_point(y, p, cut, alpha=ALPHA):
    """Precision, recall, specificità e quota di soggetti da testare, con intervalli di Wilson."""
    y = np.asarray(y) == 1
    flagged = np.asarray(p) >= cut
    tp, fp = np.sum(flagged & y), np.sum(flagged & ~y)
    fn, tn = np.sum(~flagged & y), np.sum(~flagged & ~y)
    counts = {"precision": (tp, tp + fp), "recall": (tp, tp + fn),
              "specificity": (tn, tn + fp), "alert_rate": (tp + fp, len(y))}
    row = {"threshold": cut}
    for name, (k, n) in counts.items():
        row[name], row[f"{name}_low"], row[f"{name}_high"] = wilson(k, n, alpha)
    return row


def band_cutpoints(p, proportions):
    """Soglie che dividono p in fasce con le proporzioni date (quantili di p)."""
    return np.quantile(p, np.cumsum(proportions)[:-1])


def assign_bands(p, cutpoints):
    """Fascia 0 = rischio più basso; p uguale a una soglia va nella fascia superiore."""
    return np.searchsorted(cutpoints, p, side="right")


def jonckheere(p, codes):
    """Test di Jonckheere-Terpstra contro l'alternativa "p cresce con il livello" (unilaterale),
    approssimazione normale con correzione per i pareggi (Hollander, Wolfe & Chicken 2014).

    concordance = JT / coppie: quota di coppie di soggetti di livelli diversi in cui p è più alta
    nel livello più grave (pareggi 1/2); 0,5 = nessuna tendenza.
    """
    p = np.asarray(p, dtype=float)
    codes = np.asarray(codes)
    groups = [p[codes == g] for g in np.unique(codes)]
    jt, pairs = 0.0, 0
    for low, high in combinations(groups, 2):
        ranks = stats.rankdata(np.concatenate([low, high]))
        # U di Mann-Whitney del livello più grave: coppie in cui la sua p è più alta
        jt += ranks[len(low):].sum() - len(high) * (len(high) + 1) / 2
        pairs += len(low) * len(high)
    N = float(len(p))
    n = np.array([len(g) for g in groups], dtype=float)
    t = np.unique(p, return_counts=True)[1].astype(float)
    var = ((N * (N - 1) * (2 * N + 5) - np.sum(n * (n - 1) * (2 * n + 5))
            - np.sum(t * (t - 1) * (2 * t + 5))) / 72
           + np.sum(n * (n - 1) * (n - 2)) * np.sum(t * (t - 1) * (t - 2)) / (36 * N * (N - 1) * (N - 2))
           + np.sum(n * (n - 1)) * np.sum(t * (t - 1)) / (8 * N * (N - 1)))
    z = (jt - pairs / 2) / np.sqrt(var) if var > 0 else np.nan
    return {"jt": jt, "z": z, "p_value": stats.norm.sf(z), "concordance": jt / pairs}


# --- confronto sui fold -------------------------------------------------------------------

def corrected_ttest(differences, n_train, n_test, alpha=ALPHA):
    """t appaiato sui fold con la varianza corretta di Nadeau & Bengio 2003: (1/k + n_test/n_train) s².
    I fold condividono gran parte del training, quindi la varianza semplice sarebbe troppo piccola."""
    d = np.asarray(differences, dtype=float)
    k = len(d)
    mean = d.mean()
    se = np.sqrt((1 / k + n_test / n_train) * d.var(ddof=1))
    half = stats.t.ppf(1 - alpha / 2, k - 1) * se
    t = mean / se if se > 0 else np.nan
    return {"difference": mean, "low": mean - half, "high": mean + half, "t": t,
            "p_value": 2 * stats.t.sf(abs(t), k - 1)}


def holm(p_values):
    """p-value corretti per confronti multipli (Holm 1979)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(p)
    out[order] = adjusted
    return out


def fold_metrics(oof, y):
    """AUC e PR-AUC di ogni fold esterno, per set e modello."""
    rows = []
    for (feature_set, model, fold), g in oof.groupby(["feature_set", "model", "fold"]):
        yy = y[g["row"].to_numpy()]
        rows.append({"feature_set": feature_set, "model": model, "fold": fold, "n": len(g),
                     "prevalence": yy.mean(),
                     **{name: metric(yy, g["probability"]) for name, metric in METRICS.items()}})
    return pd.DataFrame(rows)


def compare(folds, n_train, n_test):
    """Confronti appaiati sui fold: ogni coppia di modelli nello stesso set (p-value di Holm dentro
    set e metrica) e, per ogni modello, no_consequence meno main."""
    models, sets = [], []
    for metric in METRICS:
        wide = folds.pivot_table(index="fold", columns=["feature_set", "model"], values=metric)
        for feature_set in wide.columns.get_level_values(0).unique():
            block = []
            for a, b in combinations(wide[feature_set].columns, 2):
                test = corrected_ttest(wide[feature_set][a] - wide[feature_set][b], n_train, n_test)
                block.append({"feature_set": feature_set, "metric": metric, "model_a": a, "model_b": b, **test})
            block = pd.DataFrame(block)
            block["p_holm"] = holm(block["p_value"])
            models.append(block)
        if {"main", "no_consequence"} <= set(wide.columns.get_level_values(0)):
            for model in wide["main"].columns:
                diff = wide["no_consequence"][model] - wide["main"][model]
                sets.append({"model": model, "metric": metric, **corrected_ttest(diff, n_train, n_test)})
    return pd.concat(models, ignore_index=True), pd.DataFrame(sets)


# --- valutazione --------------------------------------------------------------------------

def is_constant(g):
    """Previsione costante dentro ogni fold (classificatore di maggioranza): soglia e fasce non
    hanno senso, il modello serve solo come controllo di coerenza delle metriche di ordinamento."""
    return g.groupby("fold")["probability"].nunique().max() == 1


def evaluate_model(p, y, codes, constant, proportions, n_boot, alpha=ALPHA, seed=SEED):
    """Tabelle delle domande 1-4 per un modello e un set di feature."""
    auc, auc_low, auc_high = delong(y, p, alpha)
    ap, ap_low, ap_high = pr_auc(y, p, alpha)
    discrimination = {"auc": auc, "auc_low": auc_low, "auc_high": auc_high,
                      "pr_auc": ap, "pr_auc_low": ap_low, "pr_auc_high": ap_high, "prevalence": y.mean()}
    trend = jonckheere(p, codes)

    # bootstrap stratificato: medie per livello, tendenza e kappa (fasce ristimate a ogni campione)
    boot = {"concordance": [], "kappa": [], **{f"mean_{i}": [] for i in range(len(LEVELS))}}
    for idx in stratified_indices(codes, n_boot, seed):
        pb, cb = p[idx], codes[idx]
        boot["concordance"].append(jonckheere(pb, cb)["concordance"])
        for i in range(len(LEVELS)):
            boot[f"mean_{i}"].append(pb[cb == i].mean())
        if not constant:
            bands = assign_bands(pb, band_cutpoints(pb, proportions))
            boot["kappa"].append(cohen_kappa_score(cb, bands, labels=range(len(LEVELS)),
                                                   weights=SETTINGS["kappa_weights"]))
    trend["concordance_low"], trend["concordance_high"] = percentile_interval(boot["concordance"], alpha)

    levels = []
    for i, level in enumerate(LEVELS):
        low, high = percentile_interval(boot[f"mean_{i}"], alpha)
        levels.append({"level": level, "n": int(np.sum(codes == i)), "mean_probability": p[codes == i].mean(),
                       "low": low, "high": high, "median_probability": np.median(p[codes == i])})
    out = {"discrimination": discrimination, "trend": trend, "levels": levels}
    if constant:
        return out

    cut = threshold(y, p)
    out["discrimination"].update(operating_point(y, p, cut, alpha))
    out["operating_points"] = [{"target_sensitivity": s, **operating_point(y, p, threshold_at_sensitivity(y, p, s), alpha)}
                               for s in SETTINGS["sensitivity_grid"]]
    out["sensitivity"] = []
    for i, level in enumerate(LEVELS[1:], start=1):
        n, k = int(np.sum(codes == i)), int(np.sum(p[codes == i] >= cut))
        value, low, high = wilson(k, n, alpha)
        out["sensitivity"].append({"level": level, "n": n, "detected": k, "missed": n - k,
                                   "sensitivity": value, "low": low, "high": high})
    cutpoints = band_cutpoints(p, proportions)
    bands = assign_bands(p, cutpoints)
    table = pd.crosstab(pd.Categorical(np.asarray(LEVELS)[codes], LEVELS),
                        pd.Categorical(bands, range(len(LEVELS))), dropna=False)
    out["bands"] = [{"level": level, "band": band + 1, "n": int(table.loc[level, band])}
                    for level in LEVELS for band in range(len(LEVELS))]
    low, high = percentile_interval(boot["kappa"], alpha)
    out["kappa"] = {"kappa": cohen_kappa_score(codes, bands, labels=range(len(LEVELS)),
                                               weights=SETTINGS["kappa_weights"]),
                    "low": low, "high": high, "observed_agreement": np.mean(codes == bands)}
    out["cutpoints"] = {"threshold": cut, **{f"band_{b + 2}_from": c for b, c in enumerate(cutpoints)}}
    return out


def evaluate(oof, y, levels, n_boot=SETTINGS["bootstrap"], alpha=ALPHA, seed=SEED):
    """Tutte le tabelle della valutazione, come dict nome -> DataFrame.

    oof: previsioni out-of-fold in formato lungo (row, fold, model, feature_set, probability),
    row = posizione del soggetto in y e levels.
    """
    if SETTINGS["bands"] != "kdigo_quantiles":
        raise ValueError(f"fasce non implementate: {SETTINGS['bands']}")
    y = np.asarray(y)
    codes = np.asarray(pd.Categorical(levels, LEVELS, ordered=True).codes)
    # fasce con le stesse proporzioni dei livelli KDIGO nel training
    proportions = np.bincount(codes, minlength=len(LEVELS)) / len(codes)
    rows = {name: [] for name in ["discrimination", "operating_points", "levels", "trend",
                                  "sensitivity", "bands", "kappa", "cutpoints"]}
    for (feature_set, model), g in oof.groupby(["feature_set", "model"], sort=False):
        p = g.set_index("row")["probability"].reindex(range(len(y))).to_numpy()
        assert not np.isnan(p).any(), f"{feature_set} {model}: soggetti senza previsione out-of-fold"
        key = {"feature_set": feature_set, "model": model}
        result = evaluate_model(p, y, codes, is_constant(g), proportions, n_boot, alpha, seed)
        for name, value in result.items():
            rows[name] += [{**key, **r} for r in (value if isinstance(value, list) else [value])]
    tables = {f"q{q}_{name}": pd.DataFrame(rows[name]) for q, name in
              [(1, "discrimination"), (1, "operating_points"), (2, "levels"), (2, "trend"),
               (3, "sensitivity"), (4, "bands"), (4, "kappa")]}
    tables["cutpoints"] = pd.DataFrame(rows["cutpoints"])

    folds = fold_metrics(oof, y)
    # metriche anche come media dei fold (Forman & Scholz 2010): la versione sulle previsioni
    # aggregate penalizza i modelli non calibrati fra un fold e l'altro
    means = folds.groupby(["feature_set", "model"], sort=False)[list(METRICS)].mean().add_suffix("_folds_mean")
    tables["q1_discrimination"] = tables["q1_discrimination"].merge(means.reset_index(), on=["feature_set", "model"])
    n_test = oof.groupby(["feature_set", "model", "fold"]).size().mean()
    tables["folds"] = folds
    tables["comparison_models"], tables["comparison_sets"] = compare(folds, len(y) - n_test, n_test)
    return tables


def load_predictions(sensitivity=None):
    """Previsioni out-of-fold primarie; con un'analisi di sensibilità della Fase A si aggiungono le
    sue, come modello "<modello>_<analisi>", per confrontarle sugli stessi fold."""
    oof = pd.read_csv(PREDICTIONS)
    if sensitivity is None:
        return oof
    extra = pd.read_csv(PREDICTIONS.parent / "sensitivity" / sensitivity / "oof_predictions.csv")
    extra["model"] = extra["model"] + "_" + sensitivity
    return pd.concat([oof, extra], ignore_index=True)


def run(sensitivity=None):
    train = pd.read_csv(PROCESSED / "train.csv")
    tables = evaluate(load_predictions(sensitivity), target(train), kdigo_level(train))
    output = OUTPUT if sensitivity is None else PREDICTIONS.parent / "sensitivity" / sensitivity / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
        print(f"{output / name}.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sensitivity", choices=list(CONFIG["phase_a"].get("sensitivity", {})),
                        help="valuta anche un'analisi di sensibilità della Fase A")
    run(parser.parse_args().sensitivity)
