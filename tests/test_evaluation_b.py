"""Test della valutazione della Fase B su dati sintetici: nessuna previsione reale viene letta."""
import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit, logit

import src.models.evaluation as ev
import src.models.evaluation_b as eb
from src.data.kidney import LEVELS


def test_mcnemar_matches_binomial():
    a = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1], dtype=bool)
    b = np.array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
    gained, lost, p = eb.mcnemar_exact(a, b)
    assert (gained, lost) == (4, 1)
    assert p == pytest.approx(stats.binomtest(1, 5, 0.5).pvalue)
    assert eb.mcnemar_exact(a, a)[2] == 1.0


def test_newcombe_properties():
    rng = np.random.default_rng(0)
    a = rng.random(71) < 0.9
    # stessi rilevamenti: differenza 0 dentro l'intervallo
    d, low, high = eb.newcombe_paired(a, a)
    assert d == 0 and low <= 0 <= high
    # phi = 0 (prodotti incrociati uguali): somma in quadratura dei limiti di Wilson
    x = np.r_[np.ones(20, bool), np.ones(20, bool), np.zeros(20, bool), np.zeros(20, bool)]
    y = np.r_[np.ones(20, bool), np.zeros(20, bool), np.ones(20, bool), np.zeros(20, bool)]
    d, low, high = eb.newcombe_paired(x, y)
    _, l1, u1 = ev.wilson(40, 80)
    _, l2, u2 = ev.wilson(40, 80)
    assert low == pytest.approx(-np.sqrt((0.5 - l1) ** 2 + (u2 - 0.5) ** 2))
    b = rng.random(71) < 0.8
    d, low, high = eb.newcombe_paired(a, b)
    assert -1 <= low < d < high <= 1


def test_calibration_slope():
    rng = np.random.default_rng(1)
    p = expit(rng.normal(-2, 1.5, 50000))
    y = (rng.random(50000) < p).astype(int)
    good = eb.calibration(y, p)
    assert good["slope"] == pytest.approx(1, abs=0.05) and good["intercept"] == pytest.approx(0, abs=0.05)
    # probabilità troppo estreme: pendenza circa 0,5
    assert eb.calibration(y, expit(2 * logit(p)))["slope"] == pytest.approx(0.5, abs=0.05)
    assert np.isnan(eb.calibration(y, np.full(len(y), 0.1))["slope"])


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(2)
    n = 2000
    codes = rng.choice(4, n, p=[0.88, 0.08, 0.025, 0.015])
    y = (codes > 0).astype(int)
    folds = rng.permutation(np.arange(n) % 5)
    frames = []
    for technique, shift in [("none", 0.0), ("class_weight", 0.3), ("ctgan", 0.1)]:
        noisy = expit(codes + shift + rng.normal(0, 1.2, n) - 2.5)
        prior = np.array([y[folds != k].mean() for k in range(5)])[folds]
        for model, p in [("dummy", prior), ("good", noisy)]:
            frames.append(pd.DataFrame({"row": np.arange(n), "fold": [f"outer{k}" for k in folds],
                                        "model": model, "feature_set": "main", "technique": technique,
                                        "probability": p,
                                        "probability_calibrated": np.nan if technique == "ctgan" else p}))
    return pd.concat(frames, ignore_index=True), y, pd.Series(np.asarray(LEVELS)[codes])


def test_evaluate_synthetic(synthetic):
    oof, y, levels = synthetic
    tables = eb.evaluate(oof, y, levels, n_boot=20)
    assert set(tables["primary_endpoint"]["model"]) == {"good"}  # maggioranza esclusa
    comparison = tables["primary_comparison"].set_index("technique")
    assert comparison.loc["class_weight", "confirmatory"] and not comparison.loc["ctgan", "confirmatory"]
    assert np.isnan(comparison.loc["ctgan", "p_holm"])
    assert set(tables["comparison_techniques"]["technique"]) == {"class_weight", "ctgan"}
    assert set(tables["q1_discrimination"]["technique"]) == {"none", "class_weight", "ctgan"}
    # calibrazione ricalibrata solo dove esiste; soglia ingenua 0,5
    assert set(tables["calibration"].query("probabilities == 'ricalibrata'")["technique"]) == {"none", "class_weight"}
    assert np.allclose(tables["naive"]["threshold"], 0.5)
    assert set(tables["q2_levels_calibrated"]["technique"]) == {"none", "class_weight"}


def test_does_not_read_test_set():
    import src.analytics.phase_b_report as report
    for module in (eb, report):
        source = inspect.getsource(module)
        assert "test.csv" not in source and "load_split" not in source
