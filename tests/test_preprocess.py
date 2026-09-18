from collections import Counter

import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from src.config import CONFIG, resolve
from src.data.imputation import chosen_imputer, one_se_choice, two_stage_choice
from src.data.kidney import target
from src.data.preprocess import (
    CONSEQUENCE,
    FEATURE_SETS,
    FORBIDDEN,
    build_preprocessor,
    select_features,
)
from src.data.split import PROCESSED

FEATURES = CONFIG["features"]


@pytest.fixture(scope="module")
def train():
    return pd.read_csv(PROCESSED / "train.csv")


def assigned_columns():
    excluded = [c for group in FEATURES["exclude"].values() for c in group]
    return excluded + FEATURES["numeric"] + FEATURES["categorical"]


def test_every_column_has_exactly_one_group():
    columns = assigned_columns()
    assert not [c for c, n in Counter(columns).items() if n > 1]
    header = pd.read_excel(resolve(CONFIG["data"]["raw"]), nrows=0)
    assert set(columns) == set(header.columns)


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_no_forbidden_columns(train, feature_set):
    X = select_features(train, feature_set)
    assert not set(X.columns) & set(FORBIDDEN)


def test_feature_sets(train):
    main = select_features(train, "main")
    sensitivity = select_features(train, "no_consequence")
    assert main.shape[1] == 74
    assert set(main.columns) - set(sensitivity.columns) == set(CONSEQUENCE)


def test_missing_threshold(train):
    limit = FEATURES["max_missing"]
    assert select_features(train).isna().mean().max() <= limit
    assert (train[FEATURES["exclude"]["high_missing"]].isna().mean() > limit).all()


def test_encoding(train):
    X = select_features(train)
    assert set(X["Gender"]) == {0, 1}
    for column in FEATURES["ordinal"]:
        assert set(X[column].dropna()) == {0, 1, 2}
    assert X["HypertenHis"].notna().all()


def test_no_hidden_leakage(train):
    X, y = select_features(train), target(train)
    for column in X.columns:
        observed = X[column].notna()
        auc = roc_auc_score(y[observed], X.loc[observed, column])
        assert max(auc, 1 - auc) < 0.75, column


def test_preprocessor_fills_everything(train):
    X = select_features(train)
    fit_part, new_part = X.iloc[:3000], X.iloc[3000:]
    preprocessor = build_preprocessor(SimpleImputer(strategy="median"), X.columns)
    out = preprocessor.fit(fit_part).transform(new_part)
    assert out.shape == (len(new_part), X.shape[1])
    # l'imputer numerico vede anche le categoriche come predittori
    assert preprocessor[-1].n_features_in_ == X.shape[1]
    assert not pd.isna(out).any()


def test_one_se_rule_prefers_simplest():
    table = pd.DataFrame(
        {"PR-AUC": [0.30, 0.31, 0.33], "PR-AUC SE": [0.02, 0.02, 0.02]},
        index=["mediana", "KNN", "MissForest"],
    )
    assert one_se_choice(table) == "KNN"


def test_two_stage_rule_uses_reconstruction():
    # numeri del passo 6 (75 feature, prima dell'esclusione di Homaβ)
    table = pd.DataFrame(
        {"PR-AUC": [0.249, 0.253, 0.254, 0.253], "PR-AUC SE": [0.027, 0.029, 0.031, 0.028],
         "RMSE": [1.031, 0.774, 0.870, 0.724], "RMSE SE": [0.058, 0.053, 0.168, 0.065]},
        index=["mediana", "KNN", "MICE", "MissForest"],
    )
    assert one_se_choice(table) == "mediana"
    assert two_stage_choice(table) == "KNN"


def test_two_stage_rule_on_final_features():
    # numeri del passo 8 (74 feature, set main): KNN supera la soglia RMSE di 0,0013
    table = pd.DataFrame(
        {"PR-AUC": [0.2518, 0.2554, 0.2571, 0.2557], "PR-AUC SE": [0.0299, 0.0320, 0.0328, 0.0312],
         "RMSE": [1.0384, 0.7926, 1.2529, 0.7250], "RMSE SE": [0.0471, 0.0614, 0.1360, 0.0663]},
        index=["mediana", "KNN", "MICE", "MissForest"],
    )
    assert two_stage_choice(table) == "MissForest"


def test_two_stage_rule_at_the_margin():
    # numeri del passo 10 (categoriche come predittori): KNN rientra nella soglia RMSE sul set
    # main e ne esce su no_consequence -> la regola non discrimina, decide il confronto appaiato
    main = pd.DataFrame(
        {"PR-AUC": [0.2518, 0.2586, 0.2573, 0.2572], "PR-AUC SE": [0.0299, 0.0322, 0.0323, 0.0309],
         "RMSE": [1.0384, 0.7932, 1.2518, 0.7337], "RMSE SE": [0.0471, 0.0566, 0.1351, 0.0659]},
        index=["mediana", "KNN", "MICE", "MissForest"],
    )
    no_consequence = pd.DataFrame(
        {"PR-AUC": [0.2446, 0.2481, 0.2453, 0.2463], "PR-AUC SE": [0.0281, 0.0292, 0.0292, 0.0285],
         "RMSE": [1.0376, 0.7875, 0.9497, 0.7292], "RMSE SE": [0.0457, 0.0482, 0.0954, 0.0534]},
        index=["mediana", "KNN", "MICE", "MissForest"],
    )
    assert two_stage_choice(main) == "KNN"
    assert two_stage_choice(no_consequence) == "MissForest"


def test_chosen_imputer_is_configured():
    imputer = chosen_imputer()
    assert type(imputer).__name__ == "IterativeImputer"
    assert type(imputer.estimator).__name__ == "ExtraTreesRegressor"
