"""Fase A: i cinque modelli sui dati originali, con cross-validation annidata.

Per ogni modello e fold esterno k:
  1. iperparametri scelti con Optuna sui 5 fold interni di k (TPE + MedianPruner, PR-AUC media)
  2. riaddestramento sul training esterno di k e previsioni sul fold esterno k (out-of-fold)
Modello finale: iperparametri scelti con Optuna usando i fold esterni come CV dell'intero
training, riaddestramento sull'intero training (il test si valuta una volta sola, alla fine).
Set no_consequence: stessi iperparametri del set main, fold per fold (opzione A, Notepad).
Ogni risultato è salvato appena pronto: l'esecuzione si può interrompere e riprendere.
Nessun bilanciamento e nessun peso di classe (Fase B). Il test set non viene letto.
"""
import argparse
import json
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

from src.config import resolve
from src.data.folds import INNER, OUTER
from src.data.imputed import METHOD, OUTPUT as IMPUTED, fold_name, load_fold
from src.data.kidney import target
from src.data.preprocess import FEATURE_SETS
from src.data.split import PROCESSED
from src.models.zoo import MODELS, SEED, SETTINGS, SPACES, build, columns, suggest

OUTPUT = resolve(SETTINGS["output"])
MODELS_DIR = resolve(SETTINGS["models_dir"])
N_TRIALS = SETTINGS["n_trials"]
TUNING_SET = SETTINGS["tuning_set"]
# analisi di sensibilità decise dopo i risultati (Notepad): nome -> modello e intervalli modificati
SENSITIVITY = SETTINGS.get("sensitivity", {})
FOLDS = [*range(OUTER), None]


def xy(name, feature_set, y, outer, inner=None, directory=IMPUTED, method=METHOD):
    """Fold imputato con le sole colonne del modello, e il target allineato per riga."""
    X_train, X_valid = load_fold(feature_set, outer, inner, directory, method)
    use = columns(name, X_train.columns)
    return X_train[use], y.iloc[X_train.index], X_valid[use], y.iloc[X_valid.index]


def tuning_splits(outer):
    """Fold su cui si valuta ogni tentativo: gli interni del fold esterno, oppure i fold
    esterni stessi quando si sceglie il modello finale sull'intero training."""
    if outer is None:
        return [(k, None) for k in range(OUTER)]
    return [(outer, j) for j in range(INNER)]


def objective(trial, name, data, space):
    """PR-AUC media sui fold; il pruner interrompe i tentativi sotto la mediana."""
    params = suggest(trial, space)
    scores = []
    for step, (X_tr, y_tr, X_va, y_va) in enumerate(data):
        p = build(name, params).fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
        scores.append(average_precision_score(y_va, p))
        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


def tune(name, data, space, n_trials):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(lambda trial: objective(trial, name, data, space), n_trials=n_trials)
    return study


def record_path(output, name, feature_set, outer):
    return output / feature_set / f"{name}_{fold_name(outer)}.json"


def run_fold(name, feature_set, outer, y, n_trials=N_TRIALS, params=None, space=None,
             directory=IMPUTED, method=METHOD, output=OUTPUT, models_dir=MODELS_DIR):
    """Un modello su un fold esterno (outer = None: modello finale sull'intero training).

    params = None e modello con spazio di ricerca: iperparametri scelti con Optuna.
    params dato: iperparametri riusati (set no_consequence), nessuna ottimizzazione.
    """
    start = time.perf_counter()
    space = SPACES.get(name) if space is None else space
    record = {"model": name, "feature_set": feature_set, "fold": fold_name(outer),
              "imputer": method, "tuned": params is None and space is not None}
    if record["tuned"]:
        data = [xy(name, feature_set, y, o, i, directory, method) for o, i in tuning_splits(outer)]
        study = tune(name, data, space, n_trials)
        params = study.best_params
        states = [t.state for t in study.trials]
        record.update(inner_pr_auc=study.best_value, trials=len(states),
                      pruned=states.count(optuna.trial.TrialState.PRUNED))
        trials_dir = output / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(trials_dir / f"{name}_{feature_set}_{fold_name(outer)}.csv",
                                        index=False)
    record["params"] = params or {}

    X_tr, y_tr, X_va, y_va = xy(name, feature_set, y, outer, None, directory, method)
    model = build(name, params).fit(X_tr, y_tr)
    if hasattr(model, "n_iter_"):
        record["converged"] = bool(np.all(model.n_iter_ < model.max_iter))
    if outer is None:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / f"phase_a_{name}_{feature_set}.joblib")
    else:
        record["rows"] = X_va.index.tolist()
        record["probability"] = model.predict_proba(X_va)[:, 1].tolist()
    record["seconds"] = round(time.perf_counter() - start, 1)

    path = record_path(output, name, feature_set, outer)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def collect(output=OUTPUT):
    """Previsioni out-of-fold di tutti i modelli in formato lungo: riga, fold, modello, set, p."""
    frames = []
    for path in sorted(output.glob("*/*_outer*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        frames.append(pd.DataFrame({"row": record["rows"], "fold": record["fold"],
                                    "model": record["model"], "feature_set": record["feature_set"],
                                    "probability": record["probability"]}))
    return pd.concat(frames, ignore_index=True)


def run(models=MODELS, feature_sets=tuple(FEATURE_SETS), n_trials=N_TRIALS, output=OUTPUT,
        spaces=SPACES, models_dir=MODELS_DIR):
    train = pd.read_csv(PROCESSED / "train.csv")
    y = target(train)
    # prima il set di ottimizzazione: gli altri set ne riusano gli iperparametri
    for feature_set in sorted(feature_sets, key=lambda s: s != TUNING_SET):
        for name in models:
            for outer in FOLDS:
                if record_path(output, name, feature_set, outer).exists():
                    continue
                params = None
                if feature_set != TUNING_SET and name in spaces:
                    tuned = record_path(output, name, TUNING_SET, outer)
                    params = json.loads(tuned.read_text(encoding="utf-8"))["params"]
                record = run_fold(name, feature_set, outer, y, n_trials, params, spaces.get(name),
                                  output=output, models_dir=models_dir)
                pruned = f", {record['pruned']}/{record['trials']} tentativi interrotti" if record["tuned"] else ""
                print(f"{feature_set} {name} {record['fold']}: {record['seconds']:.0f} s{pruned}",
                      flush=True)
    collect(output).to_csv(output / "oof_predictions.csv", index=False)


def sensitivity_spaces(name, sensitivity=SENSITIVITY, spaces=SPACES):
    """Spazi di ricerca di un'analisi di sensibilità: quelli primari, con gli intervalli modificati
    solo per il modello indicato."""
    spec = sensitivity[name]
    return {**spaces, spec["model"]: {**spaces[spec["model"]], **spec["space"]}}


def run_sensitivity(name, feature_sets=tuple(FEATURE_SETS), n_trials=N_TRIALS):
    """Stesso protocollo, un solo modello con lo spazio modificato. Risultati e modelli finali in
    cartelle separate: i risultati primari non vengono toccati."""
    run([SENSITIVITY[name]["model"]], feature_sets, n_trials, output=OUTPUT / "sensitivity" / name,
        spaces=sensitivity_spaces(name), models_dir=MODELS_DIR / "sensitivity" / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--sensitivity", choices=list(SENSITIVITY),
                        help="analisi di sensibilità della config (ignora --models)")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # convergenza della logistica registrata nel campo "converged" di ogni risultato
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if args.sensitivity:
        run_sensitivity(args.sensitivity, args.feature_sets, args.trials)
    else:
        run(args.models, args.feature_sets, args.trials)
