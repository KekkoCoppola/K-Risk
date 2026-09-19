import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from src.config import CONFIG
from src.data.folds import INNER, OUTER, inner_folds, outer_folds
from src.data.imputed import all_folds, load_fold, save_fold
from src.data.kidney import kdigo_level
from src.data.preprocess import select_features
from src.data.split import PROCESSED


@pytest.fixture(scope="module")
def train():
    return pd.read_csv(PROCESSED / "train.csv")


def test_outer_folds_unchanged(train):
    # gli stessi fold usati nel confronto delle imputazioni (Passi 4-10 del Notepad)
    cv = StratifiedKFold(OUTER, shuffle=True, random_state=CONFIG["seed"])
    expected = list(cv.split(train, kdigo_level(train)))
    for (tr, va), (tr_exp, va_exp) in zip(outer_folds(train), expected):
        assert np.array_equal(tr, tr_exp) and np.array_equal(va, va_exp)


def test_inner_folds_are_nested(train):
    for outer_train, outer_valid in outer_folds(train):
        folds = inner_folds(train, outer_train)
        assert len(folds) == INNER
        valid_union = np.concatenate([va for _, va in folds])
        # ogni riga del training esterno è in validazione interna una sola volta
        assert np.array_equal(np.sort(valid_union), np.sort(outer_train))
        for tr, va in folds:
            assert not set(tr) & set(va)
            # la validazione esterna non entra mai nei fold interni
            assert not (set(tr) | set(va)) & set(outer_valid)


def test_all_folds_count(train):
    assert len(all_folds(train)) == 1 + OUTER + OUTER * INNER


def test_fold_roundtrip(train, tmp_path):
    X = select_features(train)
    train_idx, valid_idx = outer_folds(train)[0]
    path = tmp_path / "main" / "outer0.npz"
    save_fold(path, X, train_idx, valid_idx, SimpleImputer(strategy="median"), method="mediana")
    X_train, X_valid = load_fold("main", outer=0, directory=tmp_path, method="mediana")
    assert X_train.shape == (len(train_idx), X.shape[1])
    assert X_valid.shape == (len(valid_idx), X.shape[1])
    assert list(X_train.index) == list(train_idx) and list(X_valid.index) == list(valid_idx)
    assert set(X_train.columns) == set(X.columns)
    assert not X_train.isna().any().any() and not X_valid.isna().any().any()
    # un fold imputato con un altro metodo non si carica per errore
    with pytest.raises(AssertionError):
        load_fold("main", outer=0, directory=tmp_path, method="MissForest")
