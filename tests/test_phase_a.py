import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.data.folds import OUTER, outer_folds
from src.data.imputed import build_cache, load_fold
from src.data.kidney import target
from src.data.preprocess import FEATURE_SETS
from src.data.split import PROCESSED
from src.models.phase_a import collect, run_fold, sensitivity_spaces
from src.models.zoo import MODELS, SCORED, SPACES, TUNED, build, columns

METHOD = "mediana"
# spazio ridotto: verifica la meccanica dell'ottimizzazione, non le prestazioni
TINY_SPACE = {"n_estimators": {"type": "int", "low": 10, "high": 20},
              "max_depth": {"type": "int", "low": 2, "high": 3}}


@pytest.fixture(scope="module")
def train():
    return pd.read_csv(PROCESSED / "train.csv")


@pytest.fixture(scope="module")
def cache(train, tmp_path_factory):
    """Fold imputati con la mediana (veloce) in una cartella temporanea."""
    directory = tmp_path_factory.mktemp("imputed")
    for feature_set in FEATURE_SETS:
        build_cache(train, feature_set, SimpleImputer(strategy="median"), directory, METHOD)
    return directory


def lowest(space):
    """Parametri al limite inferiore di ogni intervallo (o prima scelta)."""
    return {name: spec["choices"][0] if spec["type"] == "categorical" else spec["low"]
            for name, spec in space.items()}


def test_models_match_config():
    assert TUNED == ["lr_penalized", "random_forest", "xgboost"]
    assert set(MODELS) - set(TUNED) == {"dummy", "lr_scored"}


def test_scored_columns():
    assert columns("lr_scored", FEATURE_SETS["main"]) == SCORED
    # nel set di sensibilità HGB (variabile-conseguenza) esce
    assert columns("lr_scored", FEATURE_SETS["no_consequence"]) == [c for c in SCORED if c != "HGB"]
    assert columns("xgboost", FEATURE_SETS["main"]) == FEATURE_SETS["main"]


@pytest.mark.parametrize("name", MODELS)
def test_every_model_fits(train, cache, name):
    X_tr, X_va = load_fold("main", outer=0, directory=cache, method=METHOD)
    y = target(train)
    params = lowest(SPACES[name]) if name in SPACES else None
    use = columns(name, X_tr.columns)
    p = build(name, params).fit(X_tr[use], y.iloc[X_tr.index]).predict_proba(X_va[use])[:, 1]
    assert len(p) == len(X_va) and np.all((p >= 0) & (p <= 1))


def test_dummy_predicts_prevalence(train, cache):
    X_tr, X_va = load_fold("main", outer=0, directory=cache, method=METHOD)
    y = target(train)
    p = build("dummy").fit(X_tr, y.iloc[X_tr.index]).predict_proba(X_va)[:, 1]
    assert np.allclose(p, y.iloc[X_tr.index].mean())


def test_tuned_fold_and_reuse(train, cache, tmp_path):
    y = target(train)
    paths = dict(directory=cache, method=METHOD, output=tmp_path, models_dir=tmp_path / "models")
    tuned = run_fold("random_forest", "main", 0, y, n_trials=2, space=TINY_SPACE, **paths)
    assert tuned["tuned"] and tuned["trials"] == 2
    # previsioni solo sul fold esterno 0, mai sulle righe usate per addestrare
    assert sorted(tuned["rows"]) == sorted(outer_folds(train)[0][1].tolist())
    assert (tmp_path / "trials" / "random_forest_main_outer0.csv").exists()
    # set di sensibilità: iperparametri riusati, nessuna ottimizzazione
    reused = run_fold("random_forest", "no_consequence", 0, y, params=tuned["params"], **paths)
    assert not reused["tuned"] and reused["params"] == tuned["params"]
    saved = json.loads((tmp_path / "no_consequence" / "random_forest_outer0.json").read_text())
    assert saved["params"] == tuned["params"]


def test_final_model_and_collect(train, cache, tmp_path):
    y = target(train)
    paths = dict(directory=cache, method=METHOD, output=tmp_path, models_dir=tmp_path / "models")
    for outer in range(OUTER):
        run_fold("lr_scored", "main", outer, y, **paths)
    final = run_fold("lr_scored", "main", None, y, **paths)
    assert "rows" not in final and final["converged"]
    assert (tmp_path / "models" / "phase_a_lr_scored_main.joblib").exists()
    oof = collect(tmp_path)
    # ogni soggetto del training ha esattamente una previsione out-of-fold
    assert sorted(oof["row"]) == list(range(len(train)))


def test_sensitivity_changes_only_its_model():
    spaces = sensitivity_spaces("depth_1_12")
    assert spaces["xgboost"]["max_depth"] == {"type": "int", "low": 1, "high": 12}
    # gli altri iperparametri di XGBoost e gli altri modelli restano quelli primari
    assert {k: v for k, v in spaces["xgboost"].items() if k != "max_depth"} == \
        {k: v for k, v in SPACES["xgboost"].items() if k != "max_depth"}
    assert spaces["random_forest"] == SPACES["random_forest"]
    # lo spazio primario (protocollo ufficiale) non viene modificato
    assert SPACES["xgboost"]["max_depth"]["low"] == 3
