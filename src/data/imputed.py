"""Fold imputati salvati su disco, per stimare MissForest una volta sola.

Per ogni set di feature si salvano i 5 fold esterni, i 5 × 5 fold interni e il training intero
(per il modello finale). Scaling e imputer sono stimati solo sulla parte di training di ciascun
fold e non vedono mai y: riusare i fold imputati per tutti i modelli e tutte le tecniche di
bilanciamento non introduce leakage. Il test set non viene letto.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import CONFIG, resolve
from src.data.folds import inner_folds, outer_folds
from src.data.imputation import chosen_imputer
from src.data.preprocess import FEATURE_SETS, build_preprocessor, feature_types, select_features
from src.data.split import PROCESSED

OUTPUT = resolve(CONFIG["cv"]["imputed"])
METHOD = CONFIG["imputation"]["method"]
NO_ROWS = np.array([], dtype=int)


def fold_name(outer=None, inner=None):
    """full = training intero; outerK = fold esterno; outerK_innerJ = fold interno."""
    if outer is None:
        return "full"
    return f"outer{outer}" if inner is None else f"outer{outer}_inner{inner}"


def impute_fold(X, train_idx, valid_idx, imputer):
    """Stima scaling e imputer sulle righe di training e trasforma training e validazione."""
    preprocessor = build_preprocessor(clone(imputer), X.columns).fit(X.iloc[train_idx])
    X_valid = preprocessor.transform(X.iloc[valid_idx]) if len(valid_idx) else np.empty((0, X.shape[1]))
    return preprocessor.transform(X.iloc[train_idx]), X_valid


def save_fold(path, X, train_idx, valid_idx, imputer, method=METHOD):
    X_train, X_valid = impute_fold(X, train_idx, valid_idx, imputer)
    numeric, categorical = feature_types(X.columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X_train=X_train, X_valid=X_valid, train_idx=train_idx,
                        valid_idx=valid_idx, columns=np.array(numeric + categorical),
                        imputer=np.array(method))


def load_fold(feature_set, outer=None, inner=None, directory=OUTPUT, method=METHOD):
    """(X_train, X_valid) come DataFrame indicizzati dalla posizione di riga in train.csv."""
    path = directory / feature_set / f"{fold_name(outer, inner)}.npz"
    with np.load(path, allow_pickle=False) as data:
        assert str(data["imputer"]) == method, f"{path}: imputato con {data['imputer']}, config: {method}"
        columns = list(data["columns"])
        X_train = pd.DataFrame(data["X_train"], columns=columns, index=data["train_idx"])
        X_valid = pd.DataFrame(data["X_valid"], columns=columns, index=data["valid_idx"])
    return X_train, X_valid


def all_folds(df):
    """Tutti i fold da imputare: (outer, inner, train_idx, valid_idx)."""
    folds = [(None, None, np.arange(len(df)), NO_ROWS)]
    for k, (train, valid) in enumerate(outer_folds(df)):
        folds.append((k, None, train, valid))
        for j, (inner_train, inner_valid) in enumerate(inner_folds(df, train)):
            folds.append((k, j, inner_train, inner_valid))
    return folds


def build_cache(df, feature_set, imputer=None, directory=OUTPUT, method=METHOD):
    """Imputa e salva i fold che mancano (si può interrompere e riprendere)."""
    imputer = chosen_imputer() if imputer is None else imputer
    X = select_features(df, feature_set)
    for outer, inner, train_idx, valid_idx in all_folds(df):
        path = directory / feature_set / f"{fold_name(outer, inner)}.npz"
        if path.exists():
            continue
        start = time.perf_counter()
        save_fold(path, X, train_idx, valid_idx, imputer, method)
        print(f"{feature_set}/{path.stem}: {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    train = pd.read_csv(PROCESSED / "train.csv")
    for feature_set in FEATURE_SETS:
        build_cache(train, feature_set)
