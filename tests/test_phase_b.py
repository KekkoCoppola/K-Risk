"""Test della pipeline della Fase B su fold imputati con la mediana (veloci) e dati sintetici."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

import src.models.phase_b as pb
from src.data.augmented import build_cache as build_balanced, labels
from src.data.imputed import build_cache as build_imputed
from src.data.split import PROCESSED
from src.models.phase_a import fit, record_path, tune
from src.models.zoo import SPACES

METHOD = "mediana"
TINY_SPACE = {"n_estimators": {"type": "int", "low": 10, "high": 20},
              "max_depth": {"type": "int", "low": 2, "high": 3}}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    train = pd.read_csv(PROCESSED / "train.csv")
    imputed = tmp_path_factory.mktemp("imputed")
    build_imputed(train, "main", SimpleImputer(strategy="median"), imputed, METHOD)
    balanced = tmp_path_factory.mktemp("augmented")
    build_balanced(train, "oversampling", "main", balanced, imputed=(imputed, METHOD), log=None)
    y_all, codes_all = labels(train)
    return train, (imputed, METHOD), balanced, y_all, codes_all


def test_weights_of_one_equal_no_weights():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)))
    y = (X[0] + rng.normal(size=300) > 1).astype(int).to_numpy()
    s4 = tune("lr_penalized", [(X[:200], y[:200], X[200:], y[200:])], SPACES["lr_penalized"], 3)
    s5 = tune("lr_penalized", [(X[:200], y[:200], X[200:], y[200:], np.ones(200))], SPACES["lr_penalized"], 3)
    assert s4.best_value == pytest.approx(s5.best_value, abs=1e-6)
    a = fit("lr_scored", {}, X[:200], y[:200]).predict_proba(X[200:])[:, 1]
    b = fit("lr_scored", {}, X[:200], y[:200], np.ones(200)).predict_proba(X[200:])[:, 1]
    assert np.allclose(a, b, atol=1e-6)


def test_warm_start_is_first_trial():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 3)))
    y = (X[0] > 0.5).astype(int).to_numpy()
    warm = {"C": 0.5, "l1_ratio": 0.25}
    study = tune("lr_penalized", [(X[:150], y[:150], X[150:], y[150:])], SPACES["lr_penalized"], 2, warm)
    assert study.trials[0].params == warm


def test_spaces_b_widen_only_xgboost():
    assert pb.SPACES_B["xgboost"]["max_depth"] == {"type": "int", "low": 1, "high": 12}
    assert pb.SPACES_B["random_forest"] == SPACES["random_forest"]
    assert SPACES["xgboost"]["max_depth"]["low"] == 3


def test_baseline_dirs():
    assert pb.baseline_dir("xgboost").as_posix().endswith("analytics/phase_a/sensitivity/depth_1_12")
    assert pb.baseline_dir("random_forest").as_posix().endswith("analytics/phase_a")


def test_loader_trains_balanced_validates_real(setup):
    _, imputed, balanced, y_all, codes_all = setup
    X_tr, y_tr, X_va, y_va = pb.loader("random_forest", "oversampling", y_all, codes_all,
                                       directory=balanced, imputed=imputed)(0, None)
    assert np.sum(y_tr == 1) == np.sum(y_tr == 0)
    # validazione: solo righe reali del fold esterno 0, con le loro etichette
    assert X_va.index.min() >= 0 and np.array_equal(y_va, y_all[X_va.index.to_numpy()])
    X_tr, y_tr, X_va, _, w = pb.loader("lr_scored", "level_weight", y_all, codes_all, imputed=imputed)(0, 1)
    assert list(X_tr.columns) == list(X_va.columns) and len(w) == len(y_tr)


def test_run_technique_and_inner_predictions(setup, tmp_path, monkeypatch):
    _, imputed, balanced, y_all, codes_all = setup
    monkeypatch.setattr(pb, "SPACES_B", {**pb.SPACES_B, "random_forest": TINY_SPACE})
    monkeypatch.setattr(pb, "read_record", lambda *a, **k: {"params": {"n_estimators": 10, "max_depth": 2}})
    pb.run_technique("oversampling", y_all, codes_all, ["random_forest"], 2, tmp_path, tmp_path / "models",
                     balanced, imputed)
    record = json.loads(record_path(tmp_path / "oversampling", "random_forest", "main", 0).read_text())
    assert record["technique"] == "oversampling" and record["tuned"] and record["trials"] == 2
    # previsioni solo sulle righe reali del fold esterno 0
    assert min(record["rows"]) >= 0 and len(record["rows"]) == len(set(record["rows"]))
    assert (tmp_path / "models" / "oversampling" / "phase_b_oversampling_random_forest_main.joblib").exists()
    # ricalibrazione: previsioni interne reali che coprono il training del fold una volta sola
    rows, _ = pb.inner_predictions("random_forest", "oversampling", record["params"], 0, y_all, codes_all,
                                   balanced, imputed)
    assert len(rows) == len(set(rows)) and not set(rows) & set(record["rows"])


def test_platt_recovers_known_parameters():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 2, 20000)
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.5 + 0.7 * z)))).astype(int)
    a, b = pb.platt_fit(p, y)
    assert a == pytest.approx(0.5, abs=0.06) and b == pytest.approx(0.7, abs=0.05)
    # previsione costante: calibrata sulla prevalenza
    a, b = pb.platt_fit(np.full(100, 0.5), np.r_[np.ones(10), np.zeros(90)].astype(int))
    assert b == 0 and pb.platt_apply([0.5], a, b)[0] == pytest.approx(0.1)


def test_does_not_read_test_set():
    source = inspect.getsource(pb)
    assert "test.csv" not in source and "load_split" not in source
