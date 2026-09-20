"""Valutazione della Fase B sulle previsioni out-of-fold reali.

- esito primario (domanda 5): sensibilità sui casi gravi (alto + molto alto) alla soglia con
  sensibilità complessiva 0,90, stimata per tecnica e modello sulle probabilità grezze; confronto
  appaiato con "nessuna correzione" sugli stessi soggetti: McNemar esatto (McNemar 1947), IC della
  differenza appaiata (Newcombe 1998, metodo 10), Holm sulle tecniche principali, per modello
- secondari: domande 1-4 per tecnica (evaluation.evaluate); PR-AUC e AUC contro "nessuna
  correzione" con Nadeau-Bengio; calibrazione (intercetta, pendenza, Brier: Van Calster et al.
  2016) grezza e ricalibrata; soglia "ingenua" 0,5 sulle probabilità grezze
Legge solo train.csv e le previsioni out-of-fold: il test set non viene letto.
Protocollo: Notepad, "Fase B — protocollo fissato prima dei risultati".
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

import src.models.evaluation as ev
from src.config import CONFIG, resolve
from src.data.kidney import LEVELS, kdigo_level, target
from src.data.split import PROCESSED

SETTINGS = CONFIG["phase_b"]
EVAL = SETTINGS["evaluation"]
OUTPUT = resolve(EVAL["output"])
PREDICTIONS = resolve(SETTINGS["output"]) / "oof_predictions.csv"
SEVERE = [LEVELS.index(level) for level in EVAL["severe_levels"]]
CONFIRMATORY = EVAL["confirmatory"]
NAIVE = EVAL["naive_threshold"]
TARGET = CONFIG["evaluation"]["target_sensitivity"]
ALPHA = CONFIG["evaluation"]["alpha"]
CLIP = SETTINGS["clip"]


def pooled(g, n, column="probability", required=True):
    """Probabilità di un gruppo (tecnica, modello) nell'ordine delle righe di train.csv.
    required: errore se manca la previsione di qualche soggetto (esecuzione incompleta); i NaN sono
    ammessi solo dove sono previsti, cioè le probabilità ricalibrate dei bracci esplorativi."""
    p = g.set_index("row")[column].reindex(range(n)).to_numpy()
    if required and np.isnan(p).any():
        raise RuntimeError(f"{g['technique'].iloc[0]} {g['model'].iloc[0]}: "
                           f"{int(np.isnan(p).sum())} soggetti senza previsione out-of-fold ({column})")
    return p


# --- confronti appaiati sugli stessi soggetti ---------------------------------------------

def mcnemar_exact(a, b):
    """a, b: rilevato sì/no sugli stessi soggetti. (solo a, solo b, p bilaterale esatto)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    p = 1.0 if n == 0 else stats.binomtest(min(only_a, only_b), n, 0.5).pvalue
    return only_a, only_b, p


def newcombe_paired(a, b, alpha=ALPHA):
    """Differenza p_a - p_b fra proporzioni appaiate con IC di Newcombe 1998 (metodo 10): limiti di
    Wilson delle due proporzioni combinati con la correlazione phi fra a e b."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n = len(a)
    both, only_a, only_b, neither = (int(np.sum(m)) for m in (a & b, a & ~b, ~a & b, ~a & ~b))
    p1, p2 = (both + only_a) / n, (both + only_b) / n
    _, l1, u1 = ev.wilson(both + only_a, n, alpha)
    _, l2, u2 = ev.wilson(both + only_b, n, alpha)
    denominator = np.sqrt(float(both + only_a) * (only_b + neither) * (both + only_b) * (only_a + neither))
    phi = (both * neither - only_a * only_b) / denominator if denominator > 0 else 0.0
    delta = np.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    epsilon = np.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return p1 - p2, p1 - p2 - delta, p1 - p2 + epsilon


def severe_endpoint(oof, y, codes):
    """Esito primario: casi gravi riconosciuti alla soglia con sensibilità complessiva 0,90,
    per tecnica e modello, e confronto appaiato con "nessuna correzione"."""
    severe = np.isin(codes, SEVERE)
    rows, detected = [], {}
    for (technique, model), g in oof.groupby(["technique", "model"], sort=False):
        if ev.is_constant(g):
            continue
        p = pooled(g, len(y))
        cut = ev.threshold_at_sensitivity(y, p, TARGET)
        detected[technique, model] = p[severe] >= cut
        value, low, high = ev.wilson(int(detected[technique, model].sum()), int(severe.sum()))
        rows.append({"technique": technique, "model": model, "threshold": cut, "n": int(severe.sum()),
                     "detected": int(detected[technique, model].sum()), "sensitivity": value,
                     "low": low, "high": high, "alert_rate": float(np.mean(p >= cut))})
    comparisons = []
    for model in dict.fromkeys(model for _, model in detected):
        if ("none", model) not in detected:
            continue
        block = []
        for technique in dict.fromkeys(t for t, m in detected if m == model and t != "none"):
            gained, lost, p_value = mcnemar_exact(detected[technique, model], detected["none", model])
            difference, low, high = newcombe_paired(detected[technique, model], detected["none", model])
            block.append({"model": model, "technique": technique, "confirmatory": technique in CONFIRMATORY,
                          "gained": gained, "lost": lost, "difference": difference, "low": low, "high": high,
                          "p_value": p_value})
        block = pd.DataFrame(block)
        if len(block):
            confirmatory = block["confirmatory"]
            block.loc[confirmatory, "p_holm"] = ev.holm(block.loc[confirmatory, "p_value"])
            comparisons.append(block)
    return pd.DataFrame(rows), pd.concat(comparisons, ignore_index=True)


def compare_techniques(folds, n_train, n_test):
    """PR-AUC e AUC di ogni tecnica meno "nessuna correzione", appaiate sui 5 fold esterni
    (Nadeau & Bengio 2003); Holm sulle tecniche principali, per modello e metrica."""
    blocks = []
    for metric in ev.METRICS:
        wide = folds.pivot_table(index="fold", columns=["model", "technique"], values=metric)
        for model in wide.columns.get_level_values(0).unique():
            if "none" not in wide[model].columns:
                continue
            block = pd.DataFrame([{"model": model, "metric": metric, "technique": t,
                                   "confirmatory": t in CONFIRMATORY,
                                   **ev.corrected_ttest(wide[model][t] - wide[model]["none"], n_train, n_test)}
                                  for t in wide[model].columns if t != "none"])
            if len(block):
                confirmatory = block["confirmatory"]
                block.loc[confirmatory, "p_holm"] = ev.holm(block.loc[confirmatory, "p_value"])
                blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


# --- calibrazione e soglia ingenua ----------------------------------------------------------

def calibration(y, p, clip=CLIP):
    """Intercetta (calibrazione complessiva, pendenza fissa a 1), pendenza e Brier
    (Van Calster et al. 2016). Previsione costante: pendenza non definita."""
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), clip, 1 - clip)
    lp = logit(p)
    intercept = optimize.brentq(lambda a: np.sum(y - expit(a + lp)), -20, 20)
    slope = (LogisticRegression(C=np.inf, max_iter=1000).fit(lp[:, None], y).coef_[0][0]
             if np.ptp(lp) > 1e-12 else np.nan)
    return {"intercept": intercept, "slope": slope, "brier": float(np.mean((p - y) ** 2)),
            "mean_probability": float(p.mean()), "prevalence": float(y.mean())}


def calibration_table(oof, y):
    rows = []
    for (technique, model), g in oof.groupby(["technique", "model"], sort=False):
        for kind, column in [("grezza", "probability"), ("ricalibrata", "probability_calibrated")]:
            p = pooled(g, len(y), column, required=column == "probability")
            if not np.isnan(p).any():
                rows.append({"technique": technique, "model": model, "probabilities": kind, **calibration(y, p)})
    return pd.DataFrame(rows)


def naive_table(oof, y, codes):
    """Soglia 0,5 sulle probabilità grezze: quello che mostrerebbe una valutazione ingenua."""
    severe = np.isin(codes, SEVERE)
    rows = []
    for (technique, model), g in oof.groupby(["technique", "model"], sort=False):
        p = pooled(g, len(y))
        severe_value, _, _ = ev.wilson(int(np.sum(p[severe] >= NAIVE)), int(severe.sum()))
        rows.append({"technique": technique, "model": model, **ev.operating_point(y, p, NAIVE),
                     "severe_sensitivity": severe_value})
    return pd.DataFrame(rows)


def calibrated_levels(oof, y, codes, n_boot=CONFIG["evaluation"]["bootstrap"], seed=CONFIG["seed"]):
    """Probabilità media ricalibrata per livello KDIGO, con IC bootstrap stratificato."""
    rows = []
    samples = list(ev.stratified_indices(codes, n_boot, seed))
    for (technique, model), g in oof.groupby(["technique", "model"], sort=False):
        p = pooled(g, len(y), "probability_calibrated", required=False)
        if np.isnan(p).any():
            continue
        for i, level in enumerate(LEVELS):
            boot = [p[idx][codes[idx] == i].mean() for idx in samples]
            low, high = ev.percentile_interval(boot)
            rows.append({"technique": technique, "model": model, "level": level,
                         "mean_probability": p[codes == i].mean(), "low": low, "high": high})
    return pd.DataFrame(rows)


def evaluate(oof, y, levels, n_boot=CONFIG["evaluation"]["bootstrap"]):
    y = np.asarray(y)
    codes = np.asarray(pd.Categorical(levels, LEVELS, ordered=True).codes)
    # domande 1-4 per tecnica: la tecnica prende il posto del set di variabili
    as_sets = oof.drop(columns="feature_set").rename(columns={"technique": "feature_set"})
    tables = {name: table.rename(columns={"feature_set": "technique"})
              for name, table in ev.evaluate(as_sets, y, levels, n_boot).items()}
    tables.pop("comparison_sets", None)
    n_test = oof.groupby(["technique", "model", "fold"]).size().mean()
    tables["comparison_techniques"] = compare_techniques(tables["folds"], len(y) - n_test, n_test)
    tables["primary_endpoint"], tables["primary_comparison"] = severe_endpoint(oof, y, codes)
    tables["calibration"] = calibration_table(oof, y)
    tables["naive"] = naive_table(oof, y, codes)
    tables["q2_levels_calibrated"] = calibrated_levels(oof, y, codes, n_boot)
    return tables


def run(output=OUTPUT):
    train = pd.read_csv(PROCESSED / "train.csv")
    tables = evaluate(pd.read_csv(PREDICTIONS), target(train), kdigo_level(train))
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
        print(f"{output / name}.csv", flush=True)


if __name__ == "__main__":
    run()
