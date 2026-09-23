"""Fase D: ricerca del tetto di prestazione (analisi post-hoc, esplorativa).

Diagnostiche, per capire dove sta il limite:
  - controllo positivo: la stessa pipeline con l'albumina urinaria fra le feature deve imparare
  - curva di apprendimento: AUROC al crescere della quota di training
  - qualità del target: AUROC per gruppi di giornate di raccolta e per componente del target
  - audit del 23/09/2026: controllo positivo a intensità nota (feature semi-sintetica), rumore
    dell'etichetta in AUROC, sensibilità alla soglia e all'unità dell'ACR (le ultime due sulle
    previsioni out-of-fold esistenti, senza addestramenti)
Candidati, confrontati con i modelli della Fase A sugli stessi fold esterni:
  ensemble, XGBoost sui NaN nativi, XGBoost con spazio allargato, scomposizione del target,
  altre famiglie di modelli (CatBoost, LightGBM, EBM, TabPFN).
Regola di decisione fissata in config (phase_d) prima di qualsiasi calcolo. Nessun modello finale
sull'intero training finché un candidato non supera la regola. Il test set non viene letto.
"""
import argparse
import json
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score

import src.models.evaluation as ev
from src.config import CONFIG, resolve
from src.data.folds import INNER, OUTER
from src.data.imputed import METHOD, OUTPUT as IMPUTED, fold_name, load_fold
from src.data.kidney import egfr, target
from src.data.preprocess import select_features
from src.data.split import PROCESSED
from src.models.interpretation import OUTPUT as INTERPRETATION, shap_values
from src.models.phase_a import fit, objective, tune
from src.models.phase_b import SPACES_B
from src.models.zoo import SEED

SETTINGS = CONFIG["phase_d"]
OUTPUT = resolve(SETTINGS["output"])
CANDIDATES = SETTINGS["candidates"]
DIAGNOSTICS = SETTINGS["diagnostics"]
REFERENCES = SETTINGS["references"]
N_TRIALS = SETTINGS["n_trials"]
COLUMNS = CONFIG["columns"]
THRESHOLDS = CONFIG["target"]
FEATURE_SET = "main"


# --- dati ---------------------------------------------------------------------------------

def components(df):
    """Componenti del target: albuminuria (ACR >= 30) ed eGFR < 60, come array 0/1."""
    albuminuria = (df[COLUMNS["acr"]] >= THRESHOLDS["acr_threshold"]).astype(int).to_numpy()
    low_egfr = (egfr(df) < THRESHOLDS["egfr_threshold"]).astype(int).to_numpy()
    return albuminuria, low_egfr


def loader(df, y, data="imputed", extra=(), directory=IMPUTED, method=METHOD):
    """Dati di un fold: (X_tr, y_tr, X_va, y_va). data = "imputed" (fold della Fase A), "raw" (stesse
    righe, feature codificate ma non scalate né imputate) oppure "imputed_scaled" (fold della Fase A
    con tutte le colonne standardizzate). extra: colonne di df aggiunte."""
    raw = select_features(df, FEATURE_SET) if data == "raw" else None
    y = np.asarray(y)

    def load(outer, inner):
        X_tr, X_va = load_fold(FEATURE_SET, outer, inner, directory, method)
        if raw is not None:
            X_tr = raw.iloc[X_tr.index.to_numpy()].set_index(X_tr.index)
            X_va = raw.iloc[X_va.index.to_numpy()].set_index(X_va.index)
        if data == "imputed_scaled":
            # tutte le colonne standardizzate con media e deviazione del training del fold
            mean, sd = X_tr.mean(), X_tr.std(ddof=0).replace(0, 1)
            X_tr, X_va = (X_tr - mean) / sd, (X_va - mean) / sd
        for column in extra:
            X_tr = X_tr.assign(**{column: df[column].to_numpy()[X_tr.index.to_numpy()]})
            X_va = X_va.assign(**{column: df[column].to_numpy()[X_va.index.to_numpy()]})
        return X_tr, y[X_tr.index.to_numpy()], X_va, y[X_va.index.to_numpy()]
    return load


# --- risultati ----------------------------------------------------------------------------

def record_path(tag, outer, output=OUTPUT):
    return output / tag / f"{fold_name(outer)}.json"


def save_record(tag, outer, record, output=OUTPUT):
    path = record_path(tag, outer, output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def reference_record(name, outer):
    """Risultato della Fase A per un modello di riferimento (XGBoost: analisi max_depth 1-12)."""
    directory = resolve(SETTINGS["reference_overrides"].get(name, CONFIG["phase_a"]["output"]))
    path = directory / FEATURE_SET / f"{name}_{fold_name(outer)}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_record(tag, outer, output=OUTPUT):
    if tag in REFERENCES:
        return reference_record(tag, outer)
    return json.loads(record_path(tag, outer, output).read_text(encoding="utf-8"))


# --- validazione annidata -----------------------------------------------------------------

def tune_without_pruner(name, data, space, n_trials):
    """Come phase_a.tune, ma ogni tentativo viene valutato su tutti i fold interni."""
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.NopPruner())
    study.optimize(lambda trial: objective(trial, name, data, space), n_trials=n_trials)
    return study


def nested(tag, name, load, space, n_trials=N_TRIALS, output=OUTPUT, pruner=True):
    """Per ogni fold esterno: Optuna sui 5 fold interni (se c'è uno spazio), riaddestramento sul
    training esterno, previsioni sul fold esterno. Si riprende da dove si è fermato."""
    for outer in range(OUTER):
        if record_path(tag, outer, output).exists():
            continue
        start = time.perf_counter()
        record = {"candidate": tag, "model": name, "fold": fold_name(outer), "tuned": bool(space)}
        params = {}
        if space:
            data = [load(outer, j) for j in range(INNER)]
            study = (tune if pruner else tune_without_pruner)(name, data, space, n_trials)
            params = study.best_params
            states = [t.state for t in study.trials]
            record.update(inner_pr_auc=study.best_value, trials=len(states),
                          pruned=states.count(optuna.trial.TrialState.PRUNED))
        X_tr, y_tr, X_va, y_va = load(outer, None)
        record.update(params=params, rows=X_va.index.tolist(),
                      probability=fit(name, params, X_tr, y_tr).predict_proba(X_va)[:, 1].tolist(),
                      seconds=round(time.perf_counter() - start, 1))
        save_record(tag, outer, record, output)
        print(f"{tag} {record['fold']}: {record['seconds']:.0f} s", flush=True)


# --- tri-ensemble sulle variabili più importanti --------------------------------------------

def consensus_top(rankings, k):
    """Prime k variabili per rango medio fra più classifiche (ognuna dalla più importante); a parità
    di rango medio vale l'ordine della prima classifica."""
    features = list(rankings[0])
    if any(sorted(r) != sorted(features) for r in rankings):
        raise ValueError("le classifiche non contengono le stesse variabili")
    positions = [{f: i for i, f in enumerate(r)} for r in rankings]
    mean_rank = {f: np.mean([p[f] for p in positions]) for f in features}
    return sorted(features, key=lambda f: (mean_rank[f], positions[0][f]))[:k]


def ranking(model, name, X, y, shap_rows):
    """Variabili dalla più importante, col criterio di interpretation.top_features: |coefficiente|
    per la logistica, SHAP medio assoluto per gli alberi (per la Random Forest su shap_rows righe
    estratte per classe, perché TreeSHAP su centinaia di alberi profondi è molto lento)."""
    if name == "lr_penalized":
        scores = np.abs(model.coef_[0])
    else:
        rows = np.arange(len(X))
        if name == "random_forest" and shap_rows and shap_rows < len(X):
            rows = stratified_subsample(y, shap_rows / len(X), np.random.default_rng(SEED))
        scores = np.abs(shap_values(model, X.iloc[rows])).mean(axis=0)
    return list(X.columns[np.argsort(-scores, kind="stable")])


def fold_rankings(X_tr, y_tr, outer, members, shap_rows):
    """Classifiche stimate sul solo training del fold esterno, con gli iperparametri della Fase A."""
    return [ranking(fit(name, reference_record(name, outer)["params"], X_tr, y_tr), name, X_tr, y_tr,
                    shap_rows) for name in members]


def full_training_rankings(members, directory=INTERPRETATION):
    """Classifiche dei modelli finali della Fase A, stimati sull'intero training (coefficienti e SHAP
    di analytics/phase_a/interpretation/): la selezione vede anche i soggetti di validazione."""
    odds = pd.read_csv(directory / "odds_ratios.csv").query("feature_set == @FEATURE_SET")
    importance = pd.read_csv(directory / "shap_importance.csv").query("feature_set == @FEATURE_SET")
    lists = []
    for name in members:
        if name == "lr_penalized":
            table = odds[odds["model"] == name]
            lists.append(list(table.iloc[np.argsort(-table["coefficient"].abs().to_numpy(),
                                                    kind="stable")]["feature"]))
        else:
            table = importance[importance["model"] == name]
            lists.append(list(table.sort_values("mean_abs_shap", ascending=False, kind="stable")["feature"]))
    return lists


def run_tri_ensemble(tag, spec, df, output=OUTPUT, members=REFERENCES):
    """Media delle probabilità dei riferimenti addestrati sulle prime spec["top"] variabili, con gli
    iperparametri della Fase A del fold. selection: "nested" (classifica sul training del fold) o
    "full_training" (classifica sull'intero training, solo come diagnostica)."""
    if spec["selection"] not in ("nested", "full_training"):
        raise ValueError(f"{tag}: selezione sconosciuta {spec['selection']}")
    load = loader(df, target(df).to_numpy(), spec["data"])
    for outer in range(OUTER):
        if record_path(tag, outer, output).exists():
            continue
        start = time.perf_counter()
        X_tr, y_tr, X_va, _ = load(outer, None)
        rankings = (fold_rankings(X_tr, y_tr, outer, members, spec.get("shap_rows"))
                    if spec["selection"] == "nested" else full_training_rankings(members))
        features = consensus_top(rankings, spec["top"])
        p = np.mean([fit(name, reference_record(name, outer)["params"], X_tr[features], y_tr)
                     .predict_proba(X_va[features])[:, 1] for name in members], axis=0)
        record = {"candidate": tag, "members": list(members), "selection": spec["selection"],
                  "fold": fold_name(outer), "features": features, "rows": X_va.index.tolist(),
                  "probability": p.tolist(), "seconds": round(time.perf_counter() - start, 1)}
        save_record(tag, outer, record, output)
        print(f"{tag} {record['fold']}: {record['seconds']:.0f} s", flush=True)


# --- candidati ----------------------------------------------------------------------------

def run_candidate(tag, df, output=OUTPUT):
    spec = CANDIDATES[tag]
    kind = spec["kind"]
    y = target(df).to_numpy()
    if kind == "ensemble":
        for outer in range(OUTER):
            records = [reference_record(name, outer) for name in REFERENCES]
            rows = records[0]["rows"]
            assert all(r["rows"] == rows for r in records), "righe diverse fra i riferimenti"
            p = np.mean([r["probability"] for r in records], axis=0)
            save_record(tag, outer, {"candidate": tag, "members": REFERENCES, "fold": fold_name(outer),
                                     "rows": rows, "probability": p.tolist()}, output)
        return
    if kind == "tri_ensemble":
        run_tri_ensemble(tag, spec, df, output)
        return
    if kind == "xgboost":
        space = {**SPACES_B["xgboost"], **spec.get("space", {})}
        nested(tag, "xgboost", loader(df, y, spec["data"]), space, spec.get("n_trials", N_TRIALS), output,
               pruner=spec.get("pruner") != "none")
        return
    if kind == "decomposition":
        parts = dict(zip(["albuminuria", "egfr"], components(df)))
        for part, y_part in parts.items():
            nested(f"{tag}/{part}", "xgboost", loader(df, y_part, spec["data"]), SPACES_B["xgboost"],
                   N_TRIALS, output)
        for outer in range(OUTER):
            a, g = (read_record(f"{tag}/{part}", outer, output) for part in parts)
            assert a["rows"] == g["rows"]
            p = 1 - (1 - np.asarray(a["probability"])) * (1 - np.asarray(g["probability"]))
            save_record(tag, outer, {"candidate": tag, "fold": fold_name(outer), "rows": a["rows"],
                                     "probability": p.tolist()}, output)
        return
    space = SETTINGS["spaces"].get(kind, SPACES_B.get(kind))
    nested(tag, kind, loader(df, y, spec["data"]), space, spec.get("n_trials", N_TRIALS), output)


# --- diagnostiche -------------------------------------------------------------------------

def positive_control(df, output=OUTPUT):
    """XGBoost con gli iperparametri della Fase A e le colonne extra (albumina urinaria)."""
    spec = DIAGNOSTICS["positive_control"]
    load = loader(df, target(df).to_numpy(), "imputed", spec["extra"])
    for outer in range(OUTER):
        params = reference_record(spec["model"], outer)["params"]
        X_tr, y_tr, X_va, _ = load(outer, None)
        p = fit(spec["model"], params, X_tr, y_tr).predict_proba(X_va)[:, 1]
        save_record("positive_control", outer, {"candidate": "positive_control", "extra": spec["extra"],
                                                "fold": fold_name(outer), "params": params,
                                                "rows": X_va.index.tolist(), "probability": p.tolist()},
                    output)


def stratified_subsample(y, fraction, rng):
    """Posizioni di una quota di righe, estratta dentro ogni classe."""
    keep = [rng.choice(np.flatnonzero(y == c), int(round(fraction * np.sum(y == c))), replace=False)
            for c in (0, 1)]
    return np.sort(np.concatenate(keep))


def learning_curve(df, output=OUTPUT):
    spec = DIAGNOSTICS["learning_curve"]
    load = loader(df, target(df).to_numpy())
    rows = []
    for name in spec["models"]:
        for outer in range(OUTER):
            params = reference_record(name, outer)["params"]
            X_tr, y_tr, X_va, y_va = load(outer, None)
            for fraction in spec["fractions"]:
                for draw in range(spec["draws"] if fraction < 1 else 1):
                    keep = stratified_subsample(y_tr, fraction, np.random.default_rng(SEED + draw))
                    p = fit(name, params, X_tr.iloc[keep], y_tr[keep]).predict_proba(X_va)[:, 1]
                    rows.append({"model": name, "fold": outer, "fraction": fraction, "draw": draw,
                                 "n_train": len(keep), "positives": int(y_tr[keep].sum()),
                                 "auc": roc_auc_score(y_va, p), "pr_auc": average_precision_score(y_va, p)})
            print(f"curva di apprendimento {name} fold {outer}", flush=True)
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "learning_curve.csv", index=False)
    return table


def oof(tag, n, output=OUTPUT):
    """Probabilità out-of-fold di un modello nell'ordine delle righe di train.csv, e fold di ogni riga."""
    p = np.full(n, np.nan)
    folds = np.full(n, -1)
    for outer in range(OUTER):
        record = read_record(tag, outer, output)
        p[record["rows"]] = record["probability"]
        folds[record["rows"]] = outer
    assert not np.isnan(p).any(), f"{tag}: soggetti senza previsione"
    return p, folds


def label_quality(df, tags, output=OUTPUT):
    """AUROC nelle giornate con urine più diluite (mediana di UCRE nel quartile basso delle giornate)
    contro le altre, e AUROC per componente del target (ogni componente contro i negativi)."""
    spec = DIAGNOSTICS["label_quality"]
    y = target(df).to_numpy()
    albuminuria, low_egfr = components(df)
    day_median = df.groupby(spec["day"])[spec["urine_creatinine"]].median()
    low_days = day_median[day_median <= day_median.quantile(spec["low_quantile"])].index
    low = df[spec["day"]].isin(low_days).to_numpy()
    rows = []
    for tag in tags:
        p, _ = oof(tag, len(df), output)
        groups = {"tutte le giornate": np.ones(len(df), bool), "giornate UCRE basso": low,
                  "altre giornate": ~low}
        for group, mask in groups.items():
            rows.append({"model": tag, "analysis": group, "n": int(mask.sum()),
                         "positives": int(y[mask].sum()), "prevalence": y[mask].mean(),
                         "auc": roc_auc_score(y[mask], p[mask])})
        negatives = y == 0
        for name, positive in [("solo albuminuria", (albuminuria == 1) & (low_egfr == 0)),
                               ("eGFR < 60", low_egfr == 1)]:
            mask = negatives | positive
            rows.append({"model": tag, "analysis": f"componente: {name}", "n": int(mask.sum()),
                         "positives": int(positive.sum()), "prevalence": positive[mask].mean(),
                         "auc": roc_auc_score(positive[mask], p[mask])})
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "label_quality.csv", index=False)
    return table, sorted(map(str, low_days))


# --- audit del tetto dei dati (23/09/2026) ---------------------------------------------------

SEMI_SYNTHETIC_COLUMN = "z_semisintetica"


def noise_for_auroc(signal, y, noise, target_auroc, iterations=60):
    """DS del rumore che, sommato al segnale, porta l'AUROC univariata a target_auroc (bisezione; il
    rumore è un'estrazione fissa, quindi il risultato è deterministico)."""
    def auc(sd):
        return roc_auc_score(y, signal + sd * noise)
    if auc(0.0) < target_auroc:
        raise ValueError(f"il segnale da solo ha AUROC {auc(0.0):.3f} < {target_auroc}")
    low, high = 0.0, 1.0
    while auc(high) > target_auroc:
        high *= 2
        if high > 1e6:
            raise ValueError("AUROC chiesta non raggiungibile con questo rumore")
    for _ in range(iterations):
        middle = (low + high) / 2
        low, high = (middle, high) if auc(middle) > target_auroc else (low, middle)
    return (low + high) / 2


def semi_synthetic_control(df, spec=None, output=OUTPUT):
    """Controllo positivo a intensità nota: z = log(max(ACR, 1)) + rumore gaussiano, con la DS del
    rumore tarata sull'intero training perché l'AUROC univariata di z sia quella fissata; XGBoost con
    gli iperparametri della Fase A del fold e z fra le feature. Passa se l'AUROC out-of-fold è almeno
    l'AUROC univariata di z meno la tolleranza. Mai un modello della tesi. reference_auc (AUROC dello
    stesso modello senza z) è solo contesto e non entra nel criterio."""
    spec = spec or DIAGNOSTICS["semi_synthetic_control"]
    y = target(df).to_numpy()
    signal = np.log(np.maximum(df[COLUMNS["acr"]].to_numpy(dtype=float), 1.0))
    noise = np.random.default_rng(SEED).standard_normal(len(df))
    reference = ev.delong(y, oof(spec["model"], len(y), output)[0])[0]
    rows = []
    for goal in spec["target_auroc"]:
        tag = f"semi_synthetic_control/z{round(goal * 100)}"
        sd = noise_for_auroc(signal, y, noise, goal)
        z = signal + sd * noise
        load = loader(df.assign(**{SEMI_SYNTHETIC_COLUMN: z}), y, "imputed", [SEMI_SYNTHETIC_COLUMN])
        for outer in range(OUTER):
            params = reference_record(spec["model"], outer)["params"]
            X_tr, y_tr, X_va, _ = load(outer, None)
            p = fit(spec["model"], params, X_tr, y_tr).predict_proba(X_va)[:, 1]
            save_record(tag, outer, {"candidate": tag, "target_auroc": goal, "noise_sd": sd,
                                     "fold": fold_name(outer), "params": params,
                                     "rows": X_va.index.tolist(), "probability": p.tolist()}, output)
        p, _ = oof(tag, len(y), output)
        auc, auc_low, auc_high = ev.delong(y, p)
        z_auc = roc_auc_score(y, z)
        rows.append({"target_auroc": goal, "noise_sd": sd, "noise_sd_over_signal_sd": sd / signal.std(),
                     "z_auc": z_auc, "oof_auc": auc, "oof_auc_low": auc_low, "oof_auc_high": auc_high,
                     "reference_auc": reference, "passes": bool(auc >= z_auc - spec["tolerance"])})
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "semi_synthetic_control.csv", index=False)
    return table


def acr_labels(acr, low_egfr, threshold, grey=None):
    """Etichetta con un'altra soglia di ACR (eGFR < 60 resta positivo) e righe da tenere: con
    grey = (lo, hi) si escludono i soggetti con ACR nella zona grigia, salvo quelli con eGFR < 60."""
    acr = np.asarray(acr, dtype=float)
    low_egfr = np.asarray(low_egfr).astype(bool)
    y = ((acr >= threshold) | low_egfr).astype(int)
    keep = np.ones(len(acr), bool)
    if grey is not None:
        keep = ~((acr >= grey[0]) & (acr <= grey[1]) & ~low_egfr)
    return y, keep


def same_day_auc(y, p, day):
    """AUROC sulle sole coppie (positivo, negativo) della stessa giornata: media delle AUROC di
    giornata pesata per il numero di coppie; le giornate con una sola classe non contano."""
    y, p, day = np.asarray(y), np.asarray(p, dtype=float), np.asarray(day)
    concordant, pairs = 0.0, 0
    for d in np.unique(day):
        mask = day == d
        n_pos = int(y[mask].sum())
        n_neg = int(mask.sum()) - n_pos
        if n_pos and n_neg:
            concordant += roc_auc_score(y[mask], p[mask]) * n_pos * n_neg
            pairs += n_pos * n_neg
    return concordant / pairs if pairs else np.nan


def auc_row(tag, analysis, y, p, **extra):
    auc, low, high = ev.delong(y, p)
    return {"model": tag, "analysis": analysis, **extra, "n": len(y), "positives": int(np.sum(y)),
            "prevalence": float(np.mean(y)), "auc": auc, "auc_low": low, "auc_high": high}


def label_noise_auroc(df, spec=None, output=OUTPUT):
    """Rumore dell'etichetta in AUROC, sulle previsioni out-of-fold esistenti: per fascia di ACR dei
    positivi (contro tutti i negativi), sui casi netti (zona grigia esclusa) e sulle sole coppie della
    stessa giornata. Togliere i casi vicino alla soglia alza l'AUROC anche con un'etichetta perfetta
    (effetto spettro): i guadagni sono limiti superiori del costo del rumore."""
    spec = spec or DIAGNOSTICS["label_noise_auroc"]
    y = target(df).to_numpy()
    acr = df[COLUMNS["acr"]].to_numpy(dtype=float)
    _, low_egfr = components(df)
    grey = tuple(spec["grey_zone"])
    _, clean = acr_labels(acr, low_egfr, THRESHOLDS["acr_threshold"], grey)
    masks = {"tutti": np.ones(len(y), bool)}
    for low, high in spec["bands"]:
        masks[f"fascia ACR {low:g}-{high:g}"] = (y == 0) | ((y == 1) & (acr >= low) & (acr < high))
    masks[f"casi netti (zona grigia {grey[0]:g}-{grey[1]:g} esclusa)"] = clean
    rows = []
    for tag in spec["models"]:
        p, _ = oof(tag, len(y), output)
        rows += [auc_row(tag, analysis, y[mask], p[mask]) for analysis, mask in masks.items()]
        rows.append({"model": tag, "analysis": "coppie della stessa giornata", "n": len(y),
                     "positives": int(y.sum()), "prevalence": float(y.mean()),
                     "auc": same_day_auc(y, p, df[spec["day"]].to_numpy()), "auc_low": np.nan,
                     "auc_high": np.nan})
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "label_noise_auroc.csv", index=False)
    return table


def acr_label_sensitivity(df, spec=None, output=OUTPUT):
    """AUROC delle previsioni out-of-fold esistenti (modelli addestrati sulla soglia 30) contro
    etichette con altre soglie di UMAUCR e con la scala alternativa dell'ACR: con fattore 100 invece
    di 176,8 la soglia 30 vera equivale a UMAUCR >= 30 x 176,8 / 100. Le zone grigie sono in unità
    vere (mg/g) e vengono portate sulla scala di UMAUCR. Misura quanto l'ordinamento si trasferisce,
    non un modello riaddestrato. La riga "alternativa" senza zona grigia coincide con la riga
    "osservata" a 53,04: è voluto, così ogni scala ha la sua riga di partenza."""
    spec = spec or DIAGNOSTICS["acr_label_sensitivity"]
    acr = df[COLUMNS["acr"]].to_numpy(dtype=float)
    _, low_egfr = components(df)
    observed, threshold = spec["factor"]["observed"], THRESHOLDS["acr_threshold"]
    settings = [("osservata", t, None) for t in spec["thresholds"]]
    for scale, factor in [("osservata", observed), ("alternativa", spec["factor"]["alternative"])]:
        to_umaucr = observed / factor
        cut = round(threshold * to_umaucr, 6)
        if scale == "alternativa":
            settings.append((scale, cut, None))
        settings += [(scale, cut, (low, high, low * to_umaucr, high * to_umaucr))
                     for low, high in spec["grey_zones"]]
    rows = []
    for tag in spec["models"]:
        p, _ = oof(tag, len(acr), output)
        for scale, cut, grey in settings:
            y, keep = acr_labels(acr, low_egfr, cut, grey[2:] if grey else None)
            rows.append(auc_row(tag, "sensibilità della soglia ACR", y[keep], p[keep], scale=scale,
                                threshold=cut, grey_zone=f"{grey[0]:g}-{grey[1]:g}" if grey else ""))
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "acr_label_sensitivity.csv", index=False)
    return table


# --- valutazione --------------------------------------------------------------------------

def metrics_row(tag, y, output=OUTPUT):
    """Metriche principali di un modello sulle previsioni out-of-fold, e AUROC di ogni fold."""
    p, fold = oof(tag, len(y), output)
    auc, auc_low, auc_high = ev.delong(y, p)
    ap, ap_low, ap_high = ev.pr_auc(y, p)
    point = ev.operating_point(y, p, ev.threshold_at_sensitivity(y, p, SETTINGS["target_sensitivity"]))
    per_fold = [roc_auc_score(y[fold == k], p[fold == k]) for k in range(OUTER)]
    row = {"model": tag, "auc": auc, "auc_low": auc_low, "auc_high": auc_high, "pr_auc": ap,
           "pr_auc_low": ap_low, "pr_auc_high": ap_high, "specificity": point["specificity"],
           "specificity_low": point["specificity_low"], "specificity_high": point["specificity_high"],
           "alert_rate": point["alert_rate"], "auc_folds_mean": np.mean(per_fold)}
    return row, pd.Series(per_fold, name=tag)


def evaluate(df, tags, diagnostics=(), output=OUTPUT):
    """Metriche principali per riferimenti, candidati e diagnostiche; confronto di ogni candidato con
    il riferimento con AUROC media sui fold più alta (t corretto di Nadeau-Bengio, Holm fra i
    candidati). Le diagnostiche hanno solo le metriche, fuori dalla famiglia di Holm."""
    y = target(df).to_numpy()
    n = len(y)
    rows, folds = [], []
    for role, group in [("riferimento", REFERENCES), ("candidato", tags), ("diagnostica", diagnostics)]:
        for tag in group:
            row, per_fold = metrics_row(tag, y, output)
            rows.append({"role": role, **row})
            folds.append(per_fold)
    metrics = pd.DataFrame(rows)
    folds = pd.concat(folds, axis=1)
    references = metrics[metrics["role"] == "riferimento"]
    best = references.loc[references["auc_folds_mean"].idxmax(), "model"]
    n_test = n / OUTER
    comparison = pd.DataFrame([{"candidate": tag, "reference": best,
                                **ev.corrected_ttest(folds[tag] - folds[best], n - n_test, n_test)}
                               for tag in tags])
    if len(comparison):
        comparison["p_holm"] = ev.holm(comparison["p_value"])
        comparison["improves"] = ((comparison["difference"] >= SETTINGS["min_difference"])
                                  & (comparison["low"] > 0) & (comparison["p_holm"] < 0.05))
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "discrimination.csv", index=False)
    folds.to_csv(output / "folds_auc.csv", index_label="fold")
    comparison.to_csv(output / "comparison.csv", index=False)
    return metrics, comparison


def completed(tags, output=OUTPUT):
    return [t for t in tags if record_path(t, OUTER - 1, output).exists()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", nargs="+", default=[], choices=list(CANDIDATES))
    parser.add_argument("--diagnostics", nargs="+", default=[], choices=list(DIAGNOSTICS))
    parser.add_argument("--evaluate", action="store_true", help="valuta tutto ciò che è già calcolato")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    train = pd.read_csv(PROCESSED / "train.csv")
    for tag in args.candidates:
        run_candidate(tag, train)
    # diagnostiche con le metriche in discrimination.csv, fuori dalla famiglia di Holm
    scored_diagnostics = ["positive_control"] + [name for name, spec in DIAGNOSTICS.items()
                                                 if spec.get("kind") == "tri_ensemble"]
    for name in args.diagnostics:
        if DIAGNOSTICS[name].get("kind") == "tri_ensemble":
            run_tri_ensemble(name, DIAGNOSTICS[name], train)
        elif name == "positive_control":
            positive_control(train)
        elif name == "learning_curve":
            print(learning_curve(train).groupby(["model", "fraction"])[["auc", "pr_auc"]].mean().round(3))
        elif name == "label_quality":
            table, low_days = label_quality(train, [*REFERENCES, *completed(CANDIDATES)])
            print("giornate UCRE basso:", low_days)
            print(table.round(3).to_string(index=False))
        elif name == "semi_synthetic_control":
            print(semi_synthetic_control(train).round(3).to_string(index=False))
        elif name == "label_noise_auroc":
            print(label_noise_auroc(train).round(3).to_string(index=False))
        elif name == "acr_label_sensitivity":
            print(acr_label_sensitivity(train).round(3).to_string(index=False))
    if args.evaluate:
        metrics, comparison = evaluate(train, completed(CANDIDATES), completed(scored_diagnostics))
        print(metrics.round(3).to_string(index=False))
        print(comparison.round(4).to_string(index=False))
