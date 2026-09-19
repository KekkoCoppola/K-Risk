"""I cinque modelli della Fase A (gli stessi in Fase B) e i loro spazi di ricerca.

Motivazioni e fonti nel Notepad, sezione "Fase A — modelli e protocollo". Nessun peso di
classe: è una tecnica di bilanciamento, appartiene alla Fase B.
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import CONFIG

SEED = CONFIG["seed"]
SETTINGS = CONFIG["phase_a"]
MODELS = SETTINGS["models"]
SPACES = SETTINGS["spaces"]
SCORED = SETTINGS["scored_predictors"]
# modelli con iperparametri da ottimizzare
TUNED = [name for name in MODELS if name in SPACES]


def build(name, params=None):
    params = params or {}
    if name == "dummy":
        # prevede la prevalenza del training: PR-AUC attesa = prevalenza
        return DummyClassifier(strategy="prior")
    if name == "lr_scored":
        # pochi predittori (circa 85 eventi per variabile): nessuna penalizzazione
        return LogisticRegression(C=np.inf, max_iter=5000)
    if name == "lr_penalized":
        return LogisticRegression(solver="saga", max_iter=5000, random_state=SEED, **params)
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=SEED, **params)
    if name == "xgboost":
        return XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1,
                             random_state=SEED, **params)
    raise ValueError(f"modello sconosciuto: {name}")


def columns(name, available):
    """Colonne usate dal modello: la logistica SCORED usa solo i predittori di SCORED."""
    if name == "lr_scored":
        return [c for c in SCORED if c in available]
    return list(available)


def suggest(trial, space):
    """Iperparametri proposti da Optuna secondo uno spazio della config."""
    params = {}
    for name, spec in space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                               log=spec.get("log", False))
        else:
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params
