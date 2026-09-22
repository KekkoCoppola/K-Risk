import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.data.folds import OUTER
from src.data.imputed import build_cache, load_fold
from src.data.kidney import target
from src.data.split import PROCESSED
from src.models.phase_d import (CANDIDATES, REFERENCES, SETTINGS, components, evaluate, loader,
                                reference_record, save_record, stratified_subsample)

METHOD = "mediana"


@pytest.fixture(scope="module")
def train():
    return pd.read_csv(PROCESSED / "train.csv")


@pytest.fixture(scope="module")
def cache(train, tmp_path_factory):
    """Fold imputati con la mediana (veloce) in una cartella temporanea, solo set main."""
    directory = tmp_path_factory.mktemp("imputed")
    build_cache(train, "main", SimpleImputer(strategy="median"), directory, METHOD)
    return directory


def test_rule_fixed_in_config():
    assert SETTINGS["min_difference"] == 0.01
    assert SETTINGS["target_sensitivity"] == 0.90
    assert REFERENCES == ["lr_penalized", "random_forest", "xgboost"]
    assert {"ensemble_mean", "catboost", "lightgbm", "ebm", "tabpfn"} <= set(CANDIDATES)


def test_components_make_the_target(train):
    albuminuria, low_egfr = components(train)
    assert np.array_equal(albuminuria | low_egfr, target(train).to_numpy())


def test_stratified_subsample_keeps_both_classes():
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    keep = stratified_subsample(y, 0.4, np.random.default_rng(0))
    assert len(keep) == 400 and y[keep].sum() == 40 and len(np.unique(keep)) == 400


def test_raw_loader_uses_the_same_rows(train, cache):
    y = target(train).to_numpy()
    imputed = loader(train, y, "imputed", directory=cache, method=METHOD)(0, 1)
    raw = loader(train, y, "raw", directory=cache, method=METHOD)(0, 1)
    for a, b in [(imputed[0], raw[0]), (imputed[2], raw[2])]:
        assert a.index.equals(b.index) and list(a.columns) == list(b.columns)
    assert np.array_equal(imputed[1], raw[1]) and np.array_equal(imputed[3], raw[3])
    assert raw[0].isna().any().any() and not imputed[0].isna().any().any()


def test_scaled_loader_uses_training_statistics(train, cache):
    X_tr, _, X_va, _ = loader(train, target(train), "imputed_scaled", directory=cache, method=METHOD)(0, None)
    sd = X_tr.std(ddof=0)
    assert np.allclose(X_tr.mean(), 0, atol=1e-9)
    assert np.allclose(sd[sd > 0], 1)
    assert not np.allclose(X_va.mean(), 0, atol=1e-9)


def test_positive_control_adds_the_extra_column(train, cache):
    X_tr, _, X_va, _ = loader(train, target(train), "imputed", ["UmALB"], cache, METHOD)(2, None)
    assert np.array_equal(X_va["UmALB"].to_numpy(), train["UmALB"].to_numpy()[X_va.index.to_numpy()])
    X_plain, _ = load_fold("main", 2, None, cache, METHOD)
    assert list(X_tr.columns) == [*X_plain.columns, "UmALB"]


def test_candidate_equal_to_reference_does_not_improve(train, tmp_path):
    """Un candidato identico al miglior riferimento ha differenza 0 e non supera la regola."""
    for outer in range(OUTER):
        save_record("ensemble_mean", outer, reference_record("random_forest", outer), tmp_path)
    metrics, comparison = evaluate(train, ["ensemble_mean"], output=tmp_path)
    row = comparison.iloc[0]
    assert row["reference"] == "random_forest"
    assert row["difference"] == pytest.approx(0) and not row["improves"]
    assert json.loads((tmp_path / "ensemble_mean" / "outer0.json").read_text())["rows"]
    assert set(metrics["role"]) == {"riferimento", "candidato"}
