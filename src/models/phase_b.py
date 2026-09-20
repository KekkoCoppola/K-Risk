"""Fase B: i cinque modelli della Fase A con ogni tecnica di bilanciamento.

Stesso protocollo della Fase A (CV annidata, Optuna TPE + MedianPruner, PR-AUC sulla validazione
reale) con 30 tentativi per fold: il primo è l'ottimo della Fase A per lo stesso fold (avvio
caldo). Il bilanciamento tocca solo la parte di training (src/data/augmented.py). Le tecniche
esplorative (CTGAN) non vengono ottimizzate: riusano gli iperparametri del braccio indicato in
config (params_from) e non vengono ricalibrate.
Ricalibrazione (Platt annidato): per ogni fold esterno il modello con i parametri scelti viene
riaddestrato sui 5 fold interni bilanciati e predice le righe interne reali; la regressione di
Platt su logit(p) si applica alle previsioni del fold esterno.
"nessuna correzione" = risultati della Fase A (per XGBoost l'analisi di sensibilità depth_1_12).
Il test set non viene letto. Protocollo: Notepad, "Fase B — protocollo fissato prima dei risultati".
"""
import argparse
import json
import warnings

import numpy as np
import optuna
import pandas as pd
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.config import CONFIG, resolve
from src.data.augmented import OUTPUT as AUGMENTED, TECHNIQUES, labels, training_set
from src.data.imputed import fold_name, load_fold
from src.data.split import PROCESSED
from src.models.phase_a import FOLDS, fit, record_path, run_fold, tuning_splits
from src.models.zoo import MODELS, SPACES, columns

SETTINGS = CONFIG["phase_b"]
OUTPUT = resolve(SETTINGS["output"])
MODELS_DIR = resolve(SETTINGS["models_dir"])
FEATURE_SET = SETTINGS["feature_sets"][0]
N_TRIALS = SETTINGS["n_trials"]
CLIP = SETTINGS["clip"]
SPACES_B = {name: {**space, **SETTINGS["space_overrides"].get(name, {})} for name, space in SPACES.items()}
CONFIRMATORY = SETTINGS["evaluation"]["confirmatory"]
EXPLORATORY = [t for t, spec in TECHNIQUES.items() if spec.get("exploratory")]


def baseline_dir(name):
    """Cartella dei risultati della Fase A usati come "nessuna correzione" per un modello."""
    baseline = SETTINGS["baseline"]
    return resolve(baseline["overrides"].get(name, baseline["records"]))


def read_record(technique, name, outer, feature_set=FEATURE_SET, output=OUTPUT):
    directory = baseline_dir(name) if technique == "none" else output / technique
    return json.loads(record_path(directory, name, feature_set, outer).read_text(encoding="utf-8"))


def loader(name, technique, y_all, codes_all, feature_set=FEATURE_SET, directory=AUGMENTED, imputed=None):
    """Dati di un fold: training bilanciato con la tecnica, validazione sempre reale."""
    imputed_kwargs = {} if imputed is None else {"directory": imputed[0], "method": imputed[1]}

    def load(outer, inner):
        X_tr, y_tr, weight = training_set(technique, feature_set, y_all, codes_all, outer, inner,
                                          directory, imputed)
        _, X_va = load_fold(feature_set, outer, inner, **imputed_kwargs)
        use = columns(name, X_va.columns)
        data = (X_tr[use], y_tr, X_va[use], np.asarray(y_all)[X_va.index.to_numpy()])
        return data if weight is None else data + (weight,)
    return load


def run_technique(technique, y_all, codes_all, models=MODELS, n_trials=N_TRIALS, output=OUTPUT,
                  models_dir=MODELS_DIR, directory=AUGMENTED, imputed=None):
    spec = TECHNIQUES[technique]
    for name in models:
        for outer in FOLDS:
            if record_path(output / technique, name, FEATURE_SET, outer).exists():
                continue
            params, warm_start = None, None
            if spec.get("exploratory"):
                params = read_record(spec["params_from"], name, outer, output=output)["params"]
            elif name in SPACES_B:
                warm_start = read_record("none", name, outer)["params"]
            record = run_fold(name, FEATURE_SET, outer, y_all, n_trials, params, SPACES_B.get(name),
                              output=output / technique, models_dir=models_dir / technique,
                              loader=loader(name, technique, y_all, codes_all, directory=directory, imputed=imputed),
                              prefix=f"phase_b_{technique}", meta={"technique": technique},
                              warm_start=warm_start)
            pruned = f", {record['pruned']}/{record['trials']} tentativi interrotti" if record["tuned"] else ""
            print(f"{technique} {name} {record['fold']}: {record['seconds']:.0f} s{pruned}", flush=True)


# --- ricalibrazione -----------------------------------------------------------------------

def platt_fit(p, y, clip=CLIP):
    """Regressione logistica di y su logit(p): (intercetta, pendenza). Previsione costante:
    pendenza 0 e intercetta = logit della prevalenza."""
    z = logit(np.clip(np.asarray(p, dtype=float), clip, 1 - clip))
    if np.ptp(z) < 1e-12:
        return float(logit(np.clip(np.mean(y), clip, 1 - clip))), 0.0
    model = LogisticRegression(C=np.inf, max_iter=1000).fit(z[:, None], y)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def platt_apply(p, a, b, clip=CLIP):
    return expit(a + b * logit(np.clip(np.asarray(p, dtype=float), clip, 1 - clip)))


def inner_predictions(name, technique, params, outer, y_all, codes_all, directory=AUGMENTED, imputed=None):
    """Previsioni reali dei fold interni (o dei fold esterni, per il training intero) con i
    parametri scelti: ogni soggetto del training del fold una volta sola."""
    load = loader(name, technique, y_all, codes_all, directory=directory, imputed=imputed)
    rows, probs = [], []
    for o, i in tuning_splits(outer):
        X_tr, y_tr, X_va, _, *weight = load(o, i)
        probs.append(fit(name, params, X_tr, y_tr, *weight).predict_proba(X_va)[:, 1])
        rows.append(X_va.index.to_numpy())
    return np.concatenate(rows), np.concatenate(probs)


def calibration_path(technique, name, outer, output=OUTPUT):
    return output / technique / "calibration" / FEATURE_SET / f"{name}_{fold_name(outer)}.json"


def calibrate(technique, y_all, codes_all, models=MODELS, output=OUTPUT, directory=AUGMENTED, imputed=None):
    """Platt annidato per ogni modello e fold ("none" compreso); si riprende da dove si è fermato."""
    for name in models:
        for outer in FOLDS:
            path = calibration_path(technique, name, outer, output)
            if path.exists():
                continue
            record = read_record(technique, name, outer, output=output)
            rows, p_inner = inner_predictions(name, technique, record["params"], outer, y_all, codes_all,
                                              directory, imputed)
            a, b = platt_fit(p_inner, np.asarray(y_all)[rows])
            result = {"model": name, "technique": technique, "fold": fold_name(outer), "a": a, "b": b}
            if outer is not None:
                result["probability_calibrated"] = platt_apply(record["probability"], a, b).tolist()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=1), encoding="utf-8")
            print(f"calibrazione {technique} {name} {result['fold']}: a = {a:.3f}, b = {b:.3f}", flush=True)


def collect(techniques, models=MODELS, output=OUTPUT):
    """Previsioni out-of-fold di "nessuna correzione" e delle tecniche, in formato lungo."""
    frames = []
    for technique in ["none", *techniques]:
        for name in models:
            for outer in range(len(FOLDS) - 1):
                try:
                    record = read_record(technique, name, outer, output=output)
                except FileNotFoundError:
                    continue
                path = calibration_path(technique, name, outer, output)
                calibrated = (json.loads(path.read_text(encoding="utf-8"))["probability_calibrated"]
                              if path.exists() else np.nan)
                frames.append(pd.DataFrame({"row": record["rows"], "fold": fold_name(outer), "model": name,
                                            "feature_set": FEATURE_SET, "technique": technique,
                                            "probability": record["probability"],
                                            "probability_calibrated": calibrated}))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--techniques", nargs="+", default=[], choices=list(TECHNIQUES))
    parser.add_argument("--calibrate", nargs="+", default=[], choices=["none", *CONFIRMATORY])
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--output", default=None, help="cartella alternativa (prove di tempo)")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output = OUTPUT if args.output is None else resolve(args.output)
    y_all, codes_all = labels(pd.read_csv(PROCESSED / "train.csv"))
    for technique in args.techniques:
        run_technique(technique, y_all, codes_all, args.models, args.trials, output,
                      models_dir=MODELS_DIR if args.output is None else output / "models")
    for technique in args.calibrate:
        calibrate(technique, y_all, codes_all, args.models, output)
    if args.collect:
        oof = collect([*CONFIRMATORY, *EXPLORATORY], output=output)
        oof.to_csv(output / "oof_predictions.csv", index=False)
        print(f"{output / 'oof_predictions.csv'}: {len(oof)} righe", flush=True)
