"""Qualita' e utilita' clinica (analisi post-hoc) sulle previsioni out-of-fold della Fase B.

Braccio "nessuna correzione" (per XGBoost l'analisi depth_1_12, come in Fase C). Nessun
riaddestramento; il test set non viene letto.
1-2. decision curve (Vickers & Elkin 2006) contro "testare tutti" e "non testare nessuno", su tutto
     il training e dentro diabetici e non diabetici. Nel sottogruppo il net benefit si calcola con
     numerosita' e prevalenza del gruppo: il subgroup net benefit di Benitez-Aurioles et al. 2024
     (eq. 5) e' 1 - pi_g + lambda * NB_g, quindi dentro un gruppo ordina le strategie come NB_g
3.   curva di calibrazione flessibile (spline cubica ristretta del logit di p, Van Calster et al. 2019)
4.   MCC, F1, accuratezza bilanciata: solo descrittive, improprie alla soglia clinica (Van Calster et
     al. 2025)
5.   confronto con Bragg-Gresham et al. 2024, anche su non diabetici con bersaglio sola albuminuria
6-7. punti operativi a 0,05 e 0,07 e costo per caso trovato ($49 a test ACR, Cusick et al. 2023)
8.   calibrazione dentro i sottogruppi
Nessun test formale e nessun intervallo sul net benefit (Vickers et al. 2023).
Protocollo: Notepad, "Qualita' e utilita' clinica - protocollo post-hoc".
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit

import src.models.evaluation as ev
from src.config import CONFIG, resolve
from src.data.kidney import LEVELS, kdigo_level, target
from src.data.split import PROCESSED
from src.models.evaluation_b import calibration, pooled
from src.models.phase_c import CUTPOINTS, PREDICTIONS, SEVERE, denoise_constant, groups

SETTINGS = CONFIG["quality"]
OUTPUT = resolve(SETTINGS["evaluation"]["output"])
TECHNIQUE = SETTINGS["technique"]
OPERATING = SETTINGS["operating_thresholds"]
COST = SETTINGS["test_cost"]
EXTERNAL = SETTINGS["external"]
KNOTS = SETTINGS["calibration"]["knots"]
BINS = SETTINGS["calibration"]["bins"]
CALIBRATION_B = resolve(CONFIG["phase_b"]["evaluation"]["output"]) / "calibration.csv"
ACR = CONFIG["columns"]["acr"]
ACR_THRESHOLD = CONFIG["target"]["acr_threshold"]
DIABETES = CONFIG["columns"]["diabetes"]
CLIP = CONFIG["phase_b"]["clip"]
ALPHA = CONFIG["evaluation"]["alpha"]
N_BOOT = CONFIG["evaluation"]["bootstrap"]
SEED = CONFIG["seed"]
TOTAL = "tutti"
# punti della curva flessibile, fra il percentile 1 e 99 delle probabilita' del gruppo
CURVE_POINTS = 200
HISTOGRAM_EDGES = np.linspace(0, 1, 101)
SETTINGS_EXTERNAL = {"tesi": "tutti, bersaglio composito",
                     "bragg_gresham": "non diabetici, sola albuminuria (ACR >= 30)"}


def threshold_grid(low=SETTINGS["thresholds"]["low"], high=SETTINGS["thresholds"]["high"],
                   step=SETTINGS["thresholds"]["step"]):
    """Soglie della decision curve, estremi inclusi; arrotondate per avere 0,05 e 0,07 esatti."""
    return np.round(np.arange(low, high + step / 2, step), 6)


THRESHOLDS = threshold_grid()


# --- net benefit ------------------------------------------------------------------------------

def odds(t):
    return t / (1 - t)


def confusion(y, p, t):
    """Veri e falsi positivi e negativi con positivo = p >= t, come in `evaluation.operating_point`."""
    y = np.asarray(y) == 1
    flagged = np.asarray(p) >= t
    return (int(np.sum(flagged & y)), int(np.sum(flagged & ~y)),
            int(np.sum(~flagged & y)), int(np.sum(~flagged & ~y)))


def net_benefit(y, p, t):
    """NB(t) = VP/n - FP/n * t/(1-t) (Vickers & Elkin 2006). n e' quello dei soggetti passati:
    dentro un sottogruppo entra quindi la prevalenza del sottogruppo."""
    tp, fp, _, _ = confusion(y, p, t)
    n = len(y)
    return tp / n - fp / n * odds(t)


def net_benefit_all(prevalence, t):
    """"Testare tutti": sensibilita' 1 e specificita' 0."""
    return prevalence - (1 - prevalence) * odds(t)


def net_benefit_from_rates(sensitivity, specificity, prevalence, t):
    """Seconda formula dello stesso net benefit (Matos et al. 2026), usata come controllo."""
    return sensitivity * prevalence - (1 - specificity) * (1 - prevalence) * odds(t)


def avoided_per_100(nb_model, nb_all, t):
    """Esami inutili evitati ogni 100 persone rispetto a "testare tutti" (Vickers et al. 2019)."""
    return 100 * (nb_model - nb_all) / odds(t)


def decision_curve(y, p, thresholds=THRESHOLDS):
    """Una riga per soglia: conteggi, net benefit del modello e delle due strategie di riferimento."""
    y = np.asarray(y)
    n, positives = len(y), int(np.sum(y))
    prevalence = positives / n
    rows = []
    for t in thresholds:
        tp, fp, fn, tn = confusion(y, p, t)
        nb, nb_all = net_benefit(y, p, t), net_benefit_all(prevalence, t)
        rows.append({"threshold": float(t), "n": n, "positives": positives, "prevalence": prevalence,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn, "tests": tp + fp,
                     "net_benefit": nb, "net_benefit_all": nb_all, "net_benefit_none": 0.0,
                     "avoided_per_100": avoided_per_100(nb, nb_all, t),
                     "beats_both": bool(nb > max(nb_all, 0.0))})
    return rows


def ranges(thresholds, flags):
    """Tratti contigui della griglia in cui flags e' vero, come testo "0,095-0,200"."""
    out, start, previous = [], None, None
    for t, flag in zip(thresholds, flags):
        if flag and start is None:
            start = t
        if not flag and start is not None:
            out.append((start, previous))
            start = None
        previous = t
    if start is not None:
        out.append((start, previous))
    return "; ".join(f"{a:.3f}-{b:.3f}".replace(".", ",") for a, b in out)


def decision_summary(curve):
    """Per gruppo e modello: soglie della griglia in cui il modello batte entrambe le strategie.
    Descrive la curva; non indica una soglia da usare (Vickers et al. 2019)."""
    rows = []
    for (group, model), g in curve.groupby(["group", "model"], sort=False):
        g = g.sort_values("threshold")
        useful = g["beats_both"].to_numpy()
        all_better = ((g["net_benefit_all"] >= g["net_benefit"]) & (g["net_benefit_all"] > 0)).to_numpy()
        rows.append({"group": group, "model": model, "thresholds": len(g),
                     "thresholds_beating_both": int(useful.sum()),
                     "ranges_beating_both": ranges(g["threshold"], useful),
                     "ranges_test_all_not_worse": ranges(g["threshold"], all_better),
                     "max_avoided_per_100": float(g["avoided_per_100"].max())})
    return pd.DataFrame(rows)


def check_decision_curves(curve, constant_model="dummy", tolerance=1e-12):
    """Controlli fissati nel protocollo, su tutti i gruppi e le soglie:
    - le due formule del net benefit coincidono;
    - il classificatore di maggioranza coincide con "testare tutti" sotto la sua previsione costante e
      con "non testare nessuno" sopra;
    - il net benefit e' collassabile: somma pesata dei due sottogruppi = net benefit su tutti."""
    sensitivity = curve["tp"] / curve["positives"]
    specificity = curve["tn"] / (curve["n"] - curve["positives"])
    second = net_benefit_from_rates(sensitivity, specificity, curve["prevalence"], curve["threshold"])
    bad = (second - curve["net_benefit"]).abs() > tolerance
    if bad.any():
        raise RuntimeError(f"le due formule del net benefit divergono in {int(bad.sum())} righe")

    dummy = curve[curve["model"] == constant_model]
    if dummy.empty:
        raise RuntimeError("classificatore di maggioranza assente: coerenza non verificabile")
    flags_all = dummy["tests"] == dummy["n"]
    flags_none = dummy["tests"] == 0
    if not (flags_all | flags_none).all():
        raise RuntimeError("il classificatore di maggioranza segnala solo una parte dei soggetti")
    wrong = ((flags_all & ((dummy["net_benefit"] - dummy["net_benefit_all"]).abs() > tolerance))
             | (flags_none & (dummy["net_benefit"].abs() > tolerance)))
    if wrong.any():
        raise RuntimeError("classificatore di maggioranza diverso da 'testare tutti'/'nessuno'")

    wide = curve.pivot_table(index=["model", "threshold"], columns="group",
                             values=["net_benefit", "n"], aggfunc="first")
    subgroups = [g for g in wide["n"].columns if g != TOTAL]
    weighted = sum(wide["n"][g] * wide["net_benefit"][g] for g in subgroups)
    gap = (weighted - wide["n"][TOTAL] * wide["net_benefit"][TOTAL]).abs()
    if gap.isna().any() or (gap > 1e-9).any():
        raise RuntimeError("net benefit non collassabile: sottogruppi e totale non tornano")


# --- costi --------------------------------------------------------------------------------

def cost_rows(curve_rows, cost=COST):
    """Costo per caso trovato a ogni soglia, e costo incrementale di ogni caso che "testare tutti"
    recupera rispetto al modello. Solo il costo del test ACR: non e' un'analisi costo-efficacia."""
    rows = []
    for r in curve_rows:
        n, positives, tp, tests = r["n"], r["positives"], r["tp"], r["tests"]
        extra_tests, extra_cases = n - tests, positives - tp
        row = {"threshold": r["threshold"], "n": n, "positives": positives,
               "tests_per_100": 100 * tests / n, "found_per_100": 100 * tp / n,
               "missed_per_100": 100 * (positives - tp) / n,
               "tests_per_case": tests / tp if tp else np.nan,
               "tests_per_case_all": n / positives if positives else np.nan,
               "extra_tests_per_100_all": 100 * extra_tests / n,
               "extra_cases_per_100_all": 100 * extra_cases / n}
        for name, value in cost.items():
            suffix = "" if name == "central" else f"_{name}"
            row[f"cost_per_case{suffix}"] = value * row["tests_per_case"]
            row[f"cost_per_case_all{suffix}"] = value * row["tests_per_case_all"]
            row[f"saving_per_100{suffix}"] = value * row["extra_tests_per_100_all"]
            row[f"incremental_cost_per_case{suffix}"] = (value * extra_tests / extra_cases
                                                         if extra_cases else np.nan)
        rows.append(row)
    return rows


# --- misure di classificazione (descrittive) ----------------------------------------------

def classification_row(y, p, t):
    """MCC, F1 e accuratezza bilanciata con sensibilita', specificita', VPP e VPN. Improprie alla
    soglia clinica (Van Calster et al. 2025): solo descrittive."""
    tp, fp, fn, tn = confusion(y, p, t)
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sensitivity, "specificity": specificity,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
            "mcc": (tp * tn - fp * fn) / denominator if denominator > 0 else np.nan}


# --- calibrazione -------------------------------------------------------------------------

def rcs_basis(x, knots):
    """Spline cubica ristretta (Harrell 2015): x e k - 2 termini non lineari, lineare oltre i nodi
    estremi. Normalizzata per (t_k - t_1)^2 come in Harrell, per tenere le colonne sulla stessa scala."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    k = len(t)
    if k < 3 or np.any(np.diff(t) <= 0):
        raise ValueError(f"servono almeno 3 nodi strettamente crescenti, trovati {t}")

    def cube(u):
        return np.maximum(u, 0) ** 3

    columns = [x]
    for j in range(k - 2):
        columns.append((cube(x - t[j])
                        - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                        + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
                       / (t[k - 1] - t[0]) ** 2)
    return np.column_stack(columns)


def logistic_fit(X, y, max_iter=100, tol=1e-10):
    """Regressione logistica non penalizzata con intercetta, per Newton-Raphson. Restituisce i
    coefficienti (intercetta per prima) e la matrice di covarianza (inversa dell'informazione)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X1.shape[1])
    for _ in range(max_iter):
        mu = expit(X1 @ beta)
        information = X1.T @ (X1 * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(information, X1.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError("regressione logistica della curva di calibrazione non convergente")
    mu = expit(X1 @ beta)
    information = X1.T @ (X1 * (mu * (1 - mu))[:, None])
    return beta, np.linalg.inv(information)


def flexible_curve(y, p, quantiles=KNOTS, points=CURVE_POINTS, alpha=ALPHA, clip=CLIP):
    """Curva di calibrazione flessibile: logistica di y su una spline cubica ristretta di logit(p),
    nodi ai quantili dati di logit(p); banda puntuale col metodo delta (Van Calster et al. 2019:
    "loess or spline functions"). Valutata fra il percentile 1 e 99 di p."""
    p = np.clip(np.asarray(p, dtype=float), clip, 1 - clip)
    lp = logit(p)
    knots = np.quantile(lp, quantiles)
    beta, covariance = logistic_fit(rcs_basis(lp, knots), y)
    grid = np.linspace(*np.quantile(p, [0.01, 0.99]), points)
    B = np.column_stack([np.ones(points), rcs_basis(logit(grid), knots)])
    eta = B @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", B, covariance, B))
    z = ev.z_value(alpha)
    return [{"predicted": float(g), "observed": float(expit(e)), "low": float(expit(e - z * s)),
             "high": float(expit(e + z * s))} for g, e, s in zip(grid, eta, se)]


def calibration_bins(y, p, bins=BINS, alpha=ALPHA):
    """Punti ai quantili di p (decili): probabilita' media contro proporzione osservata (Wilson)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    labels = pd.qcut(p, bins, labels=False, duplicates="drop")
    rows = []
    for b in np.unique(labels):
        here = labels == b
        observed, low, high = ev.wilson(int(y[here].sum()), int(here.sum()), alpha)
        rows.append({"bin": int(b) + 1, "n": int(here.sum()), "events": int(y[here].sum()),
                     "predicted": float(p[here].mean()), "observed": observed,
                     "low": low, "high": high})
    return rows


def simple_indices(n, n_boot, seed=SEED):
    """Campioni bootstrap semplici (non stratificati): il numero di eventi varia fra i campioni."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, n, n)


def bootstrap_intervals(y, p, samples, alpha=ALPHA):
    boot = {name: [] for name in ("intercept", "slope", "brier", "oe_ratio")}
    for idx in samples:
        c = calibration(y[idx], p[idx])
        for name in ("intercept", "slope", "brier"):
            boot[name].append(c[name])
        boot["oe_ratio"].append(y[idx].sum() / p[idx].sum())
    return {name: ev.percentile_interval(values, alpha) for name, values in boot.items()}


def calibration_row(y, p, samples, alpha=ALPHA, simple_samples=None):
    """Intercetta, pendenza e Brier (`evaluation_b.calibration`, Van Calster et al. 2016) piu' il
    rapporto O:E (Van Calster et al. 2025), con IC bootstrap percentile sui campioni dati.

    samples: bootstrap stratificato sul livello KDIGO, come fissato nel protocollo. Ogni livello
    sopra "basso" e' positivo per definizione, quindi quei campioni hanno tutti lo stesso numero di
    eventi e gli intervalli di intercetta e O:E escludono la variabilita' della prevalenza.
    simple_samples: bootstrap semplice, colonne *_simple (deviazione dichiarata nel Notepad)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    row = calibration(y, p)
    row["oe_ratio"] = float(y.sum() / p.sum())
    for name, (low, high) in bootstrap_intervals(y, p, samples, alpha).items():
        row[f"{name}_low"], row[f"{name}_high"] = low, high
    if simple_samples is not None:
        for name, (low, high) in bootstrap_intervals(y, p, simple_samples, alpha).items():
            row[f"{name}_low_simple"], row[f"{name}_high_simple"] = low, high
    row["n"], row["events"] = int(len(y)), int(y.sum())
    return row


def histogram_rows(y, p, edges=HISTOGRAM_EDGES):
    """Distribuzione delle probabilita' per esito (set minimo di Van Calster et al. 2025)."""
    rows = []
    for outcome in (0, 1):
        counts, _ = np.histogram(np.asarray(p)[np.asarray(y) == outcome], edges)
        rows += [{"outcome": outcome, "bin_from": float(a), "bin_to": float(b), "n": int(c)}
                 for a, b, c in zip(edges[:-1], edges[1:], counts)]
    return rows


def check_calibration(table, reference, tolerance=1e-9):
    """Su tutto il training intercetta, pendenza e Brier devono coincidere con la Fase B."""
    reference = reference[(reference["technique"] == TECHNIQUE)
                          & (reference["probabilities"] == "grezza")].set_index("model")
    mine = table[table["group"] == TOTAL].set_index("model")
    missing = set(mine.index) - set(reference.index)
    if missing:
        raise RuntimeError(f"modelli assenti da {CALIBRATION_B.name}: {sorted(missing)}")
    for column in ("intercept", "slope", "brier"):
        gap = (mine[column] - reference.loc[mine.index, column]).abs()
        if (gap > tolerance).any():
            raise RuntimeError(f"calibrazione diversa dalla Fase B ({column}):\n{gap.to_string()}")


# --- confronto esterno --------------------------------------------------------------------

def screened_at(y, p, cut, alpha=ALPHA):
    point = ev.operating_point(y, p, cut, alpha)
    return {"threshold": float(cut), "n": int(len(y)), "positives": int(np.sum(y)),
            "prevalence": float(np.mean(y)),
            "screened": point["alert_rate"], "screened_low": point["alert_rate_low"],
            "screened_high": point["alert_rate_high"], "detected": point["recall"],
            "detected_low": point["recall_low"], "detected_high": point["recall_high"]}


def external_rows(y, p, sensitivity=EXTERNAL["sensitivity"], thresholds=OPERATING):
    """Quota da esaminare a sensibilita' fissata e alle soglie di Bragg-Gresham et al. 2024."""
    rows = [{"rule": f"sensibilita' {sensitivity:.2f}",
             **screened_at(y, p, ev.threshold_at_sensitivity(y, p, sensitivity))}]
    rows += [{"rule": f"soglia {t:.2f}", **screened_at(y, p, t)} for t in thresholds]
    return rows


def screening_curve(y, p, grid=np.round(np.arange(0.01, 1.0001, 0.01), 2)):
    """Quota di soggetti da esaminare contro quota di positivi trovati (per la figura)."""
    rows = []
    for s in grid:
        cut = ev.threshold_at_sensitivity(y, p, s)
        tp, fp, _, _ = confusion(y, p, cut)
        rows.append({"target_sensitivity": float(s), "threshold": float(cut),
                     "screened": (tp + fp) / len(y), "detected": tp / np.sum(y)})
    return rows


def albuminuria(train):
    """Bersaglio di Bragg-Gresham et al. 2024: sola albuminuria, ACR >= 30 mg/g."""
    acr = train[ACR].to_numpy(dtype=float)
    if np.isnan(acr).any():
        raise RuntimeError(f"{ACR}: valori mancanti, bersaglio albuminuria non definito")
    return (acr >= ACR_THRESHOLD).astype(int)


# --- valutazione completa -------------------------------------------------------------------

def operating_row(y, p, codes, t, alpha=ALPHA):
    """Punto operativo a una soglia esatta: esami, casi trovati e mancati ogni 100, casi gravi."""
    tp, fp, fn, _ = confusion(y, p, t)
    n, positives = len(y), int(np.sum(y))
    severe = np.isin(codes, SEVERE)
    detected = int(np.sum(p[severe] >= t))
    severe_value, severe_low, severe_high = ev.wilson(detected, int(severe.sum()), alpha)
    recall, recall_low, recall_high = ev.wilson(tp, positives, alpha)
    nb, nb_all = net_benefit(y, p, t), net_benefit_all(positives / n, t)
    return {"threshold": float(t), "n": n, "positives": positives,
            "tests_per_100": 100 * (tp + fp) / n, "found_per_100": 100 * tp / n,
            "missed_per_100": 100 * fn / n, "recall": recall, "recall_low": recall_low,
            "recall_high": recall_high, "severe_n": int(severe.sum()), "severe_detected": detected,
            "severe_sensitivity": severe_value, "severe_low": severe_low, "severe_high": severe_high,
            "net_benefit": nb, "net_benefit_all": nb_all,
            "avoided_per_100": avoided_per_100(nb, nb_all, t)}


TABLES = ["decision_curve", "costs", "operating_points", "classification_descriptive",
          "external_comparison", "screening_curves", "calibration", "calibration_curves",
          "calibration_bins", "probability_histogram"]


def evaluate(oof, train, cutpoints, reference_calibration=None, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """Tutte le tabelle del blocco, come dict nome -> DataFrame.

    oof: previsioni out-of-fold della Fase B (row, fold, model, technique, probability), row =
    posizione del soggetto in train. cutpoints: soglia del progetto per tecnica e modello.
    reference_calibration: calibration.csv della Fase B, per il controllo di coerenza.
    """
    y_all = target(train).to_numpy()
    y_alb = albuminuria(train)
    codes_all = np.asarray(pd.Categorical(kdigo_level(train), LEVELS, ordered=True).codes)
    masks = groups(train)
    if TOTAL not in masks:
        raise RuntimeError(f"gruppo '{TOTAL}' assente da phase_c.groups")
    non_diabetic = train[DIABETES].to_numpy() == 0
    # stessi campioni bootstrap per tutti i modelli dello stesso gruppo
    samples = {name: list(ev.stratified_indices(codes_all[mask], n_boot, seed))
               for name, mask in masks.items()}
    simple = {name: list(simple_indices(int(mask.sum()), n_boot, seed)) for name, mask in masks.items()}
    cutpoints = cutpoints.set_index(["technique", "model"])
    selected = oof[oof["technique"] == TECHNIQUE]
    if selected.empty:
        raise RuntimeError(f"tecnica {TECHNIQUE} assente dalle previsioni")
    rows = {name: [] for name in TABLES}
    for model, g in selected.groupby("model", sort=False):
        p_all = denoise_constant(pooled(g, len(train)), g)
        constant = ev.is_constant(g)
        for name, mask in masks.items():
            key = {"group": name, "model": model}
            p, y, codes = p_all[mask], y_all[mask], codes_all[mask]
            curve = decision_curve(y, p)
            rows["decision_curve"] += [{**key, **r} for r in curve]
            if constant:
                # il classificatore di maggioranza serve solo al controllo della decision curve
                continue
            rows["costs"] += [{**key, **r} for r in cost_rows(curve)]
            rows["operating_points"] += [{**key, **operating_row(y, p, codes, t, alpha)}
                                         for t in OPERATING]
            rows["calibration"].append({**key, **calibration_row(y, p, samples[name], alpha,
                                                                 simple[name])})
            rows["calibration_curves"] += [{**key, **r} for r in flexible_curve(y, p, alpha=alpha)]
            rows["calibration_bins"] += [{**key, **r} for r in calibration_bins(y, p, alpha=alpha)]
            rows["probability_histogram"] += [{**key, **r} for r in histogram_rows(y, p)]
        if constant:
            continue
        # la soglia del progetto puo' mancare solo per un modello costante (come in Fase C)
        if (TECHNIQUE, model) not in cutpoints.index:
            raise RuntimeError(f"{TECHNIQUE} {model}: nessuna soglia in {CUTPOINTS.name}")
        project = float(cutpoints.loc[(TECHNIQUE, model), "threshold"])
        rules = [("soglia del progetto (sensibilita' 0,90)", project)]
        rules += [(f"soglia {t:.2f}", t) for t in OPERATING]
        for rule, t in rules:
            rows["classification_descriptive"].append(
                {"group": TOTAL, "model": model, "rule": rule, **classification_row(y_all, p_all, t)})
        for setting, y_ext, mask in [("tesi", y_all, masks[TOTAL]),
                                     ("bragg_gresham", y_alb, non_diabetic)]:
            key = {"setting": setting, "population": SETTINGS_EXTERNAL[setting], "model": model}
            rows["external_comparison"] += [{**key, **r}
                                            for r in external_rows(y_ext[mask], p_all[mask])]
            rows["screening_curves"] += [{**key, **r}
                                         for r in screening_curve(y_ext[mask], p_all[mask])]
    tables = {name: pd.DataFrame(rows[name]) for name in TABLES}
    check_decision_curves(tables["decision_curve"])
    if reference_calibration is not None:
        check_calibration(tables["calibration"], reference_calibration)
    tables["decision_summary"] = decision_summary(tables["decision_curve"])
    reference = pd.DataFrame([{"setting": "riferimento", "population": "Bragg-Gresham et al. 2024",
                               "model": "bragg_gresham_2024", "rule": f"soglia {r['threshold']:.2f}",
                               "threshold": r["threshold"], "screened": r["screened"],
                               "detected": r["detected"]} for r in EXTERNAL["reference"]])
    tables["external_comparison"] = pd.concat([tables["external_comparison"], reference],
                                              ignore_index=True)
    return tables


def check_models(oof, expected=CONFIG["phase_a"]["models"]):
    """Un modello assente dalle previsioni sparirebbe in silenzio da ogni tabella: e' un errore."""
    missing = set(expected) - set(oof.loc[oof["technique"] == TECHNIQUE, "model"])
    if missing:
        raise RuntimeError(f"modelli assenti da {PREDICTIONS.name} ({TECHNIQUE}): {sorted(missing)}")


def run(output=OUTPUT):
    train = pd.read_csv(PROCESSED / "train.csv")
    oof = pd.read_csv(PREDICTIONS)
    check_models(oof)
    tables = evaluate(oof, train, pd.read_csv(CUTPOINTS),
                      reference_calibration=pd.read_csv(CALIBRATION_B))
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
        print(f"{output / name}.csv", flush=True)


if __name__ == "__main__":
    run()
