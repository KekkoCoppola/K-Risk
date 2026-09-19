"""Test della valutazione della Fase A su dati sintetici: nessuna metrica reale viene calcolata."""
import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import src.models.evaluation as ev
from src.data.kidney import LEVELS


@pytest.fixture(scope="module")
def synthetic():
    """Livelli con proporzioni simili a KDIGO, p crescente col livello, 5 fold, 2 set, 2 modelli."""
    rng = np.random.default_rng(0)
    n = 3000
    codes = rng.choice(4, size=n, p=[0.88, 0.08, 0.025, 0.015])
    y = (codes > 0).astype(int)
    folds = rng.permutation(np.arange(n) % 5)
    frames = []
    for feature_set, shift in [("main", 0.0), ("no_consequence", -0.2)]:
        noisy = 1 / (1 + np.exp(-(codes + shift + rng.normal(0, 1.2, n) - 2.5)))
        # classificatore di maggioranza: prevalenza del training di ogni fold
        prior = np.array([y[folds != k].mean() for k in range(5)])[folds]
        for model, p in [("dummy", prior), ("good", noisy)]:
            frames.append(pd.DataFrame({"row": np.arange(n), "fold": [f"outer{k}" for k in folds],
                                        "model": model, "feature_set": feature_set, "probability": p}))
    return pd.concat(frames, ignore_index=True), y, pd.Series(np.asarray(LEVELS)[codes])


def test_wilson_reference():
    # nessun successo su 21: limite superiore z²/(n + z²)
    value, low, high = ev.wilson(0, 21)
    z = stats.norm.ppf(0.975)
    assert value == 0 and low == pytest.approx(0, abs=1e-12)
    assert high == pytest.approx(z**2 / (21 + z**2))


def test_delong_matches_bruteforce():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60)
    p = rng.integers(0, 8, 60) / 8  # molti pareggi
    auc, low, high = ev.delong(y, p)
    assert auc == pytest.approx(roc_auc_score(y, p))
    pos, neg = p[y == 1], p[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    se = np.sqrt(psi.mean(1).var(ddof=1) / len(pos) + psi.mean(0).var(ddof=1) / len(neg))
    assert (high - low) / 2 == pytest.approx(stats.norm.ppf(0.975) * se)


def test_pr_auc_interval():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    p = y * 0.3 + rng.random(200)
    ap, low, high = ev.pr_auc(y, p)
    assert ap == pytest.approx(average_precision_score(y, p))
    assert 0 < low < ap < high < 1
    # formula di Boyd et al. 2013 con n = numero di positivi (non di soggetti)
    tau = 1 / np.sqrt(y.sum() * ap * (1 - ap))
    logit = np.log(ap / (1 - ap))
    z = stats.norm.ppf(0.975)
    assert low == pytest.approx(1 / (1 + np.exp(-(logit - z * tau))))
    assert high == pytest.approx(1 / (1 + np.exp(-(logit + z * tau))))


def test_degenerate_inputs_give_nan():
    assert np.isnan(ev.delong([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3])[1])
    assert np.isnan(ev.pr_auc([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2])[1])  # separazione perfetta: AP = 1


def test_threshold_at_sensitivity():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    for target in [0.8, 0.9, 0.95]:
        cut = ev.threshold_at_sensitivity(y, p, target)
        assert np.mean(p[y == 1] >= cut) >= target
        # soglia più alta possibile: il positivo successivo farebbe scendere sotto il valore fissato
        above = np.sort(p[(y == 1) & (p > cut)])
        assert len(above) == 0 or np.mean(p[y == 1] >= above[0]) < target
    # 0,8 x 425 non deve arrotondare a 341 per errori di virgola mobile
    y = np.r_[np.ones(425), np.zeros(10)].astype(int)
    p = np.r_[np.arange(425, 0, -1), np.zeros(10)]
    assert np.sum(p[y == 1] >= ev.threshold_at_sensitivity(y, p, 0.8)) == 340


def test_threshold_youden():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    p = y * 0.2 + rng.random(300)
    fpr, tpr, thresholds = roc_curve(y, p)
    assert ev.threshold_youden(y, p) == thresholds[np.argmax(tpr - fpr)]


def test_operating_point_counts():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    row = ev.operating_point(y, p, 0.5)
    assert row["precision"] == pytest.approx(2 / 3) and row["recall"] == pytest.approx(2 / 3)
    assert row["specificity"] == pytest.approx(4 / 5) and row["alert_rate"] == pytest.approx(3 / 8)


def test_jonckheere_matches_kendall():
    rng = np.random.default_rng(5)
    codes = rng.choice(4, 400, p=[0.7, 0.2, 0.07, 0.03])
    p = np.round(codes * 0.3 + rng.normal(0, 1, 400), 1)  # pareggi nella risposta
    result = ev.jonckheere(p, codes)
    # JT e tau di Kendall fra livello e p sono lo stesso test: stesso z, p unilaterale = metà
    tau = stats.kendalltau(codes, p, method="asymptotic")
    assert 2 * min(result["p_value"], 1 - result["p_value"]) == pytest.approx(tau.pvalue)
    # JT = coppie ordinate come i livelli, pareggi 1/2 (conteggio diretto)
    i, j = np.triu_indices(len(p), 1)
    lower = codes[i] < codes[j]
    upper = codes[i] > codes[j]
    brute = (np.sum((p[j] > p[i]) & lower) + np.sum((p[i] > p[j]) & upper)
             + 0.5 * np.sum((p[i] == p[j]) & (lower | upper)))
    assert result["jt"] == pytest.approx(brute)


def test_jonckheere_constant_has_no_trend():
    result = ev.jonckheere(np.full(50, 0.1), np.repeat([0, 1], 25))
    assert result["concordance"] == 0.5 and np.isnan(result["z"])


def test_bands_follow_proportions():
    p = np.random.default_rng(6).random(10000)
    proportions = np.array([0.9, 0.08, 0.015, 0.005])
    bands = ev.assign_bands(p, ev.band_cutpoints(p, proportions))
    assert np.allclose(np.bincount(bands, minlength=4) / len(p), proportions, atol=1e-3)


def test_corrected_ttest():
    d = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    result = ev.corrected_ttest(d, n_train=800, n_test=200)
    se = np.sqrt((1 / 5 + 200 / 800) * d.var(ddof=1))
    assert result["t"] == pytest.approx(d.mean() / se)
    assert result["high"] - result["difference"] == pytest.approx(stats.t.ppf(0.975, 4) * se)


def test_holm():
    assert np.allclose(ev.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_stratified_bootstrap_keeps_levels():
    codes = np.repeat([0, 1, 2, 3], [100, 20, 5, 2])
    for idx in ev.stratified_indices(codes, 20):
        assert np.array_equal(np.bincount(codes[idx]), [100, 20, 5, 2])


def test_evaluate_synthetic(synthetic):
    oof, y, levels = synthetic
    tables = ev.evaluate(oof, y, levels, n_boot=30)
    q1 = tables["q1_discrimination"].set_index(["feature_set", "model"])
    # classificatore di maggioranza: AUC 0,5 e PR-AUC = prevalenza in ogni fold, nessuna soglia
    folds = tables["folds"].query("model == 'dummy'")
    assert np.allclose(folds["auc"], 0.5) and np.allclose(folds["pr_auc"], folds["prevalence"])
    assert np.isnan(q1.loc[("main", "dummy"), "threshold"])
    assert set(tables["q3_sensitivity"]["model"]) == {"good"}
    # soglia a sensibilità fissata: recall almeno pari al valore della config
    assert q1.loc[("main", "good"), "recall"] >= ev.SETTINGS["target_sensitivity"]
    trend = tables["q2_trend"].set_index(["feature_set", "model"])
    assert trend.loc[("main", "good"), "concordance"] > 0.5
    # una riga per livello e per fascia, fasce con le proporzioni dei livelli
    bands = tables["q4_bands"].query("feature_set == 'main' and model == 'good'")
    assert len(bands) == 16
    # (a meno di un soggetto: i quantili sono interpolati)
    assert np.allclose(bands.groupby("band")["n"].sum().to_numpy(),
                       bands.groupby("level", sort=False)["n"].sum().to_numpy(), atol=1)
    # 1 coppia di modelli per set e metrica; no_consequence contro main per ogni modello
    assert len(tables["comparison_models"]) == 2 * 2
    assert set(tables["comparison_sets"]["model"]) == {"dummy", "good"}


def test_does_not_read_test_set():
    import src.analytics.phase_a_report as report
    for module in (ev, report):
        source = inspect.getsource(module)
        assert "test.csv" not in source and "load_split" not in source
