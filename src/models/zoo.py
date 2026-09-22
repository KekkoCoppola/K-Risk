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
    # famiglie della Fase D: pacchetti importati solo quando servono
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=SEED, verbose=False, allow_writing_files=False, **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        # subsample_freq = 1: senza, LightGBM ignora subsample
        return LGBMClassifier(random_state=SEED, subsample_freq=1, verbose=-1, **params)
    if name == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(random_state=SEED, **params)
    if name == "tabpfn":
        import os

        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        # pesi v2 (licenza Prior Labs: Apache 2.0 con attribuzione, senza login); su CPU il pacchetto
        # rifiuta più di 1.000 righe se non lo si autorizza esplicitamente
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, random_state=SEED, device="cpu", ignore_pretraining_limits=True, **params)
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
