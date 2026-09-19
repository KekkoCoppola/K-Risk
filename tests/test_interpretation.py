"""Test dell'interpretazione su dati sintetici: nessun modello reale viene letto."""
import inspect

import numpy as np
import pandas as pd
import pytest
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import src.models.interpretation as it


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"Age": rng.normal(size=n), "Bpsys": rng.normal(size=n),
                      "HGB": rng.normal(size=n), "Gender": rng.integers(1, 3, n), "DM": rng.integers(0, 2, n)})
    logit = -2 + 0.8 * X["Age"] + 0.4 * X["Bpsys"] + 0.6 * X["DM"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_wald_matches_hessian(data):
    """Errori standard confrontati con l'inversa dell'Hessiana della log-verosimiglianza calcolata a mano."""
    X, y = data
    model = LogisticRegression(C=np.inf, max_iter=5000).fit(X, y)
    beta = np.r_[model.intercept_, model.coef_[0]]
    design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    p = 1 / (1 + np.exp(-design @ beta))
    hessian = sum(pi * (1 - pi) * np.outer(xi, xi) for pi, xi in zip(p, design))
    expected = np.sqrt(np.diag(np.linalg.inv(hessian)))[1:]
    assert np.allclose(it.wald_se(model, X), expected, rtol=1e-6)


def test_odds_ratios_table(data):
    X, y = data
    model = LogisticRegression(C=np.inf, max_iter=5000).fit(X, y)
    table = it.odds_ratios(model, X, "lr_scored", "main")
    row = table.set_index("feature").loc["Age"]
    assert row["odds_ratio"] == pytest.approx(np.exp(model.coef_[0][0]))
    assert row["low"] < row["odds_ratio"] < row["high"] and row["per"] == "1 DS"
    assert table.set_index("feature").loc["DM", "per"] == "unità"
    # ordinata per valore assoluto del coefficiente
    assert list(np.abs(table["coefficient"])) == sorted(np.abs(table["coefficient"]), reverse=True)
    # modello penalizzato: nessun intervallo
    penalized = it.odds_ratios(LogisticRegression(C=0.1).fit(X, y), X, "lr_penalized", "main")
    assert "low" not in penalized.columns


@pytest.mark.parametrize("model", [RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0),
                                   XGBClassifier(n_estimators=30, max_depth=2, random_state=0)])
def test_shap_values_add_up(data, model):
    """I valori SHAP sommati al valore atteso ricostruiscono la previsione (probabilità per la
    Random Forest, logit per XGBoost)."""
    X, y = data
    model.fit(X, y)
    values = it.shap_values(model, X)
    assert values.shape == X.shape
    if isinstance(model, RandomForestClassifier):
        target = model.predict_proba(X)[:, 1]
        base = np.ravel(shap.TreeExplainer(model).expected_value)[-1]
    else:
        target = model.predict(X, output_margin=True)
        base = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, -1]
        # stessi contributi del pacchetto shap (che con XGBoost 3 sbaglia solo il valore base)
        assert np.allclose(values, shap.TreeExplainer(model).shap_values(X), atol=1e-6)
    assert np.allclose(values.sum(axis=1) + base, target, atol=1e-4)
    # la variabile che genera il segnale più forte è la più importante
    assert it.importance(values, X, "m", "main")["feature"].iloc[0] == "Age"


def test_does_not_read_test_set():
    source = inspect.getsource(it)
    assert "test.csv" not in source and "load_split" not in source
