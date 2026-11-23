"""Preprocessing dentro il fold per gli split G e T (piano preregistrato, B1).

Stesso schema del progetto (`build_preprocessor` in `src/data/preprocess.py`): numeriche
standardizzate, categoriche imputate con la moda, poi MissForest (`chosen_imputer()`) su
tutte le colonne. In più, come da piano: `log1p` sulle continue con asimmetria > 1 nel
training del fold, standardizzazione finale di tutte le colonne, colonne a varianza nulla
tolte. Tutto è stimato solo sulle righe di training del fold, senza y.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.folds import inner_folds, outer_folds  # noqa: E402
from src.data.imputation import chosen_imputer  # noqa: E402
from src.data.preprocess import CATEGORICAL, NUMERIC, encode  # noqa: E402

CACHE = ROOT / "data/processed/non_supervisionato"
RENAL = ["SCRE", "UmALB", "UCRE", "BUN"]


def columns_G():
    """Split G: le 74 colonne del progetto + 4 misure renali grezze (numeriche)."""
    return NUMERIC + RENAL, list(CATEGORICAL)


def transform_fold(X, train_idx, valid_idx, numeric, categorical, imputer=None):
    """Restituisce (X_train, X_valid, colonne, info) già standardizzati e imputati."""
    imputer = chosen_imputer() if imputer is None else imputer
    X = X[numeric + categorical].astype(float).copy()
    train = X.iloc[train_idx]
    skew = train[numeric].skew()
    logcols = skew[skew > 1].index.tolist()
    for c in logcols:
        shift = -train[c].min() if train[c].min() < 0 else 0.0
        X[c] = np.log1p(np.clip(X[c] + shift, 0, None))
    prepare = ColumnTransformer(
        [("numeric", StandardScaler(), numeric),
         ("categorical", SimpleImputer(strategy="most_frequent"), categorical)],
        verbose_feature_names_out=False,
    )
    pipe = make_pipeline(prepare, clone(imputer)).fit(X.iloc[train_idx])
    X_tr = pipe.transform(X.iloc[train_idx])
    X_va = pipe.transform(X.iloc[valid_idx])
    cols = numeric + categorical
    keep = X_tr.std(axis=0) > 0
    scaler = StandardScaler().fit(X_tr[:, keep])
    X_tr, X_va = scaler.transform(X_tr[:, keep]), scaler.transform(X_va[:, keep])
    cols = [c for c, k in zip(cols, keep) if k]
    info = {"log1p": logcols, "tolte_varianza_nulla": [c for c, k in zip(numeric + categorical, keep) if not k]}
    return X_tr, X_va, cols, info


def all_folds(df, inner=True):
    folds = []
    for k, (tr, va) in enumerate(outer_folds(df)):
        folds.append((f"outer{k}", tr, va))
        if inner:
            for j, (itr, iva) in enumerate(inner_folds(df, tr)):
                folds.append((f"outer{k}_inner{j}", itr, iva))
    return folds


def build_cache(split, X, numeric, categorical, df, inner=True):
    out = CACHE / split
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for name, tr, va in all_folds(df, inner):
        path = out / f"{name}.npz"
        if path.exists():
            continue
        start = time.perf_counter()
        X_tr, X_va, cols, info = transform_fold(X, tr, va, numeric, categorical)
        np.savez_compressed(path, X_train=X_tr, X_valid=X_va, train_idx=tr, valid_idx=va,
                            columns=np.array(cols), log1p=np.array(info["log1p"]),
                            dropped=np.array(info["tolte_varianza_nulla"]))
        sec = time.perf_counter() - start
        log.append({"fold": name, "secondi": round(sec, 1), "colonne": len(cols),
                    "log1p": len(info["log1p"]), "tolte": len(info["tolte_varianza_nulla"])})
        print(f"{split}/{name}: {sec:.0f} s, {len(cols)} colonne", flush=True)
    return pd.DataFrame(log)


def load(split, name):
    with np.load(CACHE / split / f"{name}.npz", allow_pickle=False) as d:
        cols = list(d["columns"])
        X_tr = pd.DataFrame(d["X_train"], columns=cols, index=d["train_idx"])
        X_va = pd.DataFrame(d["X_valid"], columns=cols, index=d["valid_idx"])
    return X_tr, X_va


def frame_G(df):
    numeric, categorical = columns_G()
    return encode(df)[numeric + categorical], numeric, categorical


def frame_T(df, max_levels=10):
    """Split T: tutte le 190 colonne. Adattamenti tecnici (piano, B1 e Deviazioni):
    codifiche di `encode()`; 9 → mancante nelle colonne con 9 = sconosciuto; colonne con solo
    1 o vuoto → vuoto = 0 (stessa regola di HypertenHis); `Data` → numero della giornata;
    altre colonne non numeriche → codici interi; colonne interamente vuote tolte.
    Categoriche = al massimo `max_levels` valori distinti; le altre numeriche."""
    from src.config import CONFIG
    X = encode(df)
    for c in CONFIG["features"].get("unknown_columns", []):
        if c in X:
            X[c] = X[c].where(X[c] != 9)
    info = {"vuoto_uguale_0": [], "codificate": [], "tolte_vuote": []}
    X["Data"] = pd.to_datetime(X["Data"]).rank(method="dense") - 1
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.Series(pd.factorize(X[c])[0], index=X.index).replace(-1, np.nan)
            info["codificate"].append(c)
    for c in X.columns:
        vals = set(X[c].dropna().unique())
        if vals == {1} and X[c].isna().any():
            X[c] = X[c].fillna(0)
            info["vuoto_uguale_0"].append(c)
    empty = [c for c in X.columns if X[c].isna().all()]
    info["tolte_vuote"] = empty
    X = X.drop(columns=empty)
    nu = X.nunique()
    categorical = [c for c in X.columns if nu[c] <= max_levels]
    numeric = [c for c in X.columns if c not in categorical]
    return X[numeric + categorical], numeric, categorical, info
