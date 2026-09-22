"""Test del blocco qualita' e utilita' clinica su dati sintetici: nessuna previsione reale viene letta.

I punti delicati: il net benefit dentro un sottogruppo deve usare numerosita' e prevalenza del
sottogruppo; la soglia e' inclusiva (p >= t) come nel resto del progetto; i controlli di coerenza
devono fallire davvero quando i numeri non tornano.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

import src.models.clinical_utility as cu
import src.models.evaluation as ev
from src.config import CONFIG
from src.data.kidney import LEVELS, egfr, kdigo_level, target
from src.models.evaluation_b import calibration

COLUMNS = CONFIG["columns"]
MODELS = ["dummy", "good", "weak"]


def creatinine_for(egfr_target, age=50, gender=1):
    """Creatinina (umol/l) che da' l'eGFR voluto con CKD-EPI 2021 (come in test_phase_c)."""
    def value(scre):
        row = pd.DataFrame({COLUMNS["creatinine"]: [scre], COLUMNS["age"]: [age],
                            COLUMNS["gender"]: [gender]})
        return float(egfr(row).iloc[0])

    low, high = 1.0, 3000.0
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if value(middle) > egfr_target else (low, middle)
    return (low + high) / 2


@pytest.fixture(scope="module")
def synthetic():
    """800 soggetti su 4 livelli KDIGO, diabetici arricchiti di casi gravi, 3 modelli su 5 fold.
    Il livello "molto alto" ha eGFR < 60 con ACR alto; 30 soggetti hanno eGFR < 60 e ACR normale
    (positivi per il bersaglio della tesi, negativi per la sola albuminuria)."""
    rng = np.random.default_rng(0)
    n = 800
    codes = rng.choice(4, size=n, p=[0.80, 0.12, 0.05, 0.03])
    scre_normal, scre_low = creatinine_for(100.0), creatinine_for(50.0)
    scre = np.where(codes == 3, scre_low, scre_normal)
    acr = np.array([10.0, 100.0, 400.0, 400.0])[codes]
    # G3a/A1 e' "moderato": positivo per eGFR < 60, negativo per la sola albuminuria
    low_egfr_only = np.flatnonzero(codes == 1)[:30]
    scre[low_egfr_only], acr[low_egfr_only] = scre_low, 10.0
    diabetic = rng.random(n) < np.where(codes >= 2, 0.45, 0.15)
    train = pd.DataFrame({COLUMNS["creatinine"]: scre, COLUMNS["acr"]: acr,
                          COLUMNS["age"]: 50, COLUMNS["gender"]: 1,
                          COLUMNS["diabetes"]: diabetic.astype(int)})
    assert list(pd.Categorical(kdigo_level(train), LEVELS, ordered=True).codes) == list(codes)

    y = target(train).to_numpy()
    folds = rng.permutation(np.arange(n) % 5)
    frames, cut_rows = [], []
    for model in MODELS:
        if model == "dummy":
            # costante dentro la griglia 0,02-0,20 (la prevalenza sintetica, ~0,2, ne resta fuori)
            p = np.full(n, 0.1)
        else:
            noise = 1.2 if model == "good" else 3.0
            p = expit(codes + rng.normal(0, noise, n) - 2.0)
        frames.append(pd.DataFrame({"row": np.arange(n), "fold": [f"outer{k}" for k in folds],
                                    "model": model, "feature_set": "main", "technique": "none",
                                    "probability": p}))
        if model != "dummy":
            cut_rows.append({"technique": "none", "model": model,
                             "threshold": ev.threshold_at_sensitivity(y, p, 0.90)})
    return pd.concat(frames, ignore_index=True), train, pd.DataFrame(cut_rows)


@pytest.fixture(scope="module")
def tables(synthetic):
    oof, train, cutpoints = synthetic
    return cu.evaluate(oof, train, cutpoints, n_boot=40)


# --- net benefit ------------------------------------------------------------------------------

def test_net_benefit_by_hand():
    """5 soggetti, soglia 0,15: segnalati i primi tre (2 VP, 1 FP).
    NB = 2/5 - 1/5 * 0,15/0,85 = 0,4 - 0,0352941 = 0,3647059."""
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.05])
    assert cu.net_benefit(y, p, 0.15) == pytest.approx(0.4 - 0.2 * 0.15 / 0.85)
    assert cu.net_benefit(y, p, 0.15) == pytest.approx(0.3647059, abs=1e-7)


def test_threshold_is_inclusive():
    """p uguale alla soglia e' positivo, come in evaluation.operating_point."""
    y, p = np.array([1, 0]), np.array([0.05, 0.05])
    assert cu.confusion(y, p, 0.05) == (1, 1, 0, 0)
    assert ev.operating_point(y, p, 0.05)["alert_rate"] == 1.0


def test_test_all_and_two_formulas():
    """"Testare tutti" = net benefit con tutti segnalati; le due formule coincidono."""
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.12).astype(int)
    p = rng.random(500)
    for t in cu.THRESHOLDS:
        assert cu.net_benefit(y, np.ones(500), t) == pytest.approx(cu.net_benefit_all(y.mean(), t))
        tp, fp, fn, tn = cu.confusion(y, p, t)
        second = cu.net_benefit_from_rates(tp / (tp + fn), tn / (tn + fp), y.mean(), t)
        assert cu.net_benefit(y, p, t) == pytest.approx(second, abs=1e-12)


def test_avoided_per_100_by_hand():
    """Net benefit 0,01 sopra "testare tutti" a t = 0,20 (cambio 0,25): 4 esami evitati ogni 100."""
    assert cu.avoided_per_100(0.06, 0.05, 0.20) == pytest.approx(4.0)


def test_grid_is_the_declared_one():
    """0,020-0,200 passo 0,005: 37 soglie, 0,05 e 0,07 presenti esattamente."""
    grid = cu.THRESHOLDS
    assert len(grid) == 37
    assert grid[0] == 0.02 and grid[-1] == 0.2
    assert 0.05 in grid and 0.07 in grid
    assert np.allclose(np.diff(grid), 0.005)


def test_ranges_text():
    grid = [0.02, 0.025, 0.03, 0.035, 0.04]
    assert cu.ranges(grid, [True, True, False, True, False]) == "0,020-0,025; 0,035-0,035"
    assert cu.ranges(grid, [False] * 5) == ""
    assert cu.ranges(grid, [True] * 5) == "0,020-0,040"


# --- sottogruppi e controlli di coerenza -------------------------------------------------------

def test_subgroup_uses_subgroup_size_and_prevalence(synthetic, tables):
    """Fallirebbe se il net benefit del sottogruppo fosse diviso per n totale o se "testare tutti"
    usasse la prevalenza complessiva."""
    oof, train, _ = synthetic
    y = target(train).to_numpy()
    diabetic = train[COLUMNS["diabetes"]].to_numpy() == 1
    p = oof[oof["model"] == "good"].sort_values("row")["probability"].to_numpy()
    row = tables["decision_curve"].query("group == 'diabetici' and model == 'good' and threshold == 0.05")
    assert len(row) == 1
    row = row.iloc[0]
    tp = int(np.sum((p[diabetic] >= 0.05) & (y[diabetic] == 1)))
    fp = int(np.sum((p[diabetic] >= 0.05) & (y[diabetic] == 0)))
    n_group = int(diabetic.sum())
    assert row["n"] == n_group
    assert row["prevalence"] == pytest.approx(y[diabetic].mean())
    assert row["net_benefit"] == pytest.approx(tp / n_group - fp / n_group * 0.05 / 0.95)
    assert row["net_benefit_all"] == pytest.approx(cu.net_benefit_all(y[diabetic].mean(), 0.05))
    assert y[diabetic].mean() != pytest.approx(y.mean(), abs=0.01)  # il test discrimina davvero


def test_dummy_is_test_all_below_its_constant_and_none_above(tables):
    curve = tables["decision_curve"].query("model == 'dummy' and group == 'tutti'")
    constant = 0.1
    below, above = curve[curve["threshold"] <= constant], curve[curve["threshold"] > constant]
    assert len(below) and len(above)
    assert np.allclose(below["net_benefit"], below["net_benefit_all"])
    assert np.allclose(above["net_benefit"], 0.0)


def test_collapsibility_holds_and_check_catches_errors(tables):
    curve = tables["decision_curve"]
    cu.check_decision_curves(curve)  # passa sui dati corretti
    broken = curve.copy()
    i = broken.query("group == 'diabetici' and model == 'good'").index[3]
    broken.loc[i, "net_benefit"] += 1e-3
    with pytest.raises(RuntimeError):
        cu.check_decision_curves(broken)


def test_check_catches_a_wrong_dummy(tables):
    broken = tables["decision_curve"].copy()
    i = broken.query("model == 'dummy'").index[0]
    broken.loc[i, "tests"] = broken.loc[i, "n"] - 1
    with pytest.raises(RuntimeError):
        cu.check_decision_curves(broken)


def test_tables_are_complete(tables):
    """Nessuna riga persa: 3 modelli x 3 gruppi x 37 soglie nella decision curve (dummy compreso),
    i 2 modelli reali in tutte le altre tabelle, nessun NaN nelle colonne principali."""
    k = len(cu.THRESHOLDS)
    assert len(tables["decision_curve"]) == 3 * 3 * k
    assert len(tables["costs"]) == 2 * 3 * k
    assert len(tables["operating_points"]) == 2 * 3 * len(cu.OPERATING)
    assert len(tables["calibration"]) == 2 * 3
    assert len(tables["classification_descriptive"]) == 2 * (1 + len(cu.OPERATING))
    assert len(tables["external_comparison"]) == (2 * 2 * (1 + len(cu.OPERATING))
                                                  + len(cu.EXTERNAL["reference"]))
    main = ["net_benefit", "net_benefit_all", "avoided_per_100"]
    assert not tables["decision_curve"][main].isna().any().any()
    columns = ["intercept", "slope", "oe_ratio", "slope_low", "slope_high"]
    assert not tables["calibration"][columns].isna().any().any()
    assert set(tables["decision_summary"]["model"]) == set(MODELS)


def test_missing_model_is_an_error(synthetic):
    oof, _, _ = synthetic
    cu.check_models(oof, expected=MODELS)
    with pytest.raises(RuntimeError):
        cu.check_models(oof[oof["model"] != "weak"], expected=MODELS)


def test_missing_cutpoints_for_a_real_model_is_an_error(synthetic):
    oof, train, cutpoints = synthetic
    with pytest.raises(RuntimeError):
        cu.evaluate(oof, train, cutpoints[cutpoints["model"] != "good"], n_boot=5)


# --- costi e misure descrittive ----------------------------------------------------------------

def test_cost_rows_by_hand():
    """100 soggetti, 10 positivi; il modello esamina 40 persone e trova 8 casi.
    Esami per caso 5 -> $245; "testare tutti" 10 esami per caso -> $490; i 60 esami in piu'
    recuperano 2 casi -> $1.470 per caso aggiuntivo; risparmio $2.940 ogni 100."""
    curve = [{"threshold": 0.05, "n": 100, "positives": 10, "tp": 8, "tests": 40}]
    row = cu.cost_rows(curve, {"central": 49, "low": 36, "high": 64})[0]
    assert row["tests_per_case"] == pytest.approx(5.0)
    assert row["cost_per_case"] == pytest.approx(245.0)
    assert row["cost_per_case_low"] == pytest.approx(180.0)
    assert row["cost_per_case_all"] == pytest.approx(490.0)
    assert row["incremental_cost_per_case"] == pytest.approx(49 * 60 / 2)
    assert row["saving_per_100"] == pytest.approx(49 * 60)
    assert row["missed_per_100"] == pytest.approx(2.0)


def test_cost_rows_without_cases_are_nan():
    row = cu.cost_rows([{"threshold": 0.2, "n": 10, "positives": 2, "tp": 0, "tests": 0}])[0]
    assert np.isnan(row["tests_per_case"])
    full = cu.cost_rows([{"threshold": 0.02, "n": 10, "positives": 2, "tp": 2, "tests": 10}])[0]
    assert np.isnan(full["incremental_cost_per_case"])


def test_classification_row_matches_sklearn():
    rng = np.random.default_rng(2)
    y = (rng.random(400) < 0.15).astype(int)
    p = np.clip(y * 0.2 + rng.random(400) * 0.6, 0, 1)
    row = cu.classification_row(y, p, 0.3)
    flagged = (p >= 0.3).astype(int)
    assert row["mcc"] == pytest.approx(matthews_corrcoef(y, flagged))
    assert row["f1"] == pytest.approx(f1_score(y, flagged))
    assert row["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, flagged))


# --- calibrazione -----------------------------------------------------------------------------

def test_rcs_basis_is_linear_outside_the_boundary_knots():
    knots = [-3.0, -2.0, -1.0, 0.5]
    x = np.array([1.0, 2.0, 3.0, 4.0])  # oltre l'ultimo nodo
    basis = cu.rcs_basis(x, knots)
    assert np.allclose(np.diff(basis, 2, axis=0), 0.0)
    left = cu.rcs_basis(np.array([-6.0, -5.0, -4.0]), knots)  # prima del primo nodo
    assert np.allclose(left[:, 1:], 0.0)
    with pytest.raises(ValueError):
        cu.rcs_basis(x, [0.0, 0.0, 1.0])


def test_logistic_fit_matches_sklearn():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 3))
    y = (rng.random(2000) < expit(0.5 + X @ [1.0, -0.5, 0.2])).astype(int)
    beta, covariance = cu.logistic_fit(X, y)
    reference = LogisticRegression(C=np.inf, max_iter=1000, tol=1e-10).fit(X, y)
    assert beta[0] == pytest.approx(reference.intercept_[0], abs=1e-4)
    assert np.allclose(beta[1:], reference.coef_[0], atol=1e-4)
    assert np.all(np.diag(covariance) > 0)


def test_flexible_curve_follows_the_diagonal_only_when_calibrated():
    """Probabilita' vere: curva sulla diagonale. Probabilita' troppo estreme (logit raddoppiato):
    curva lontana dalla diagonale. Fallirebbe se la curva non dipendesse dagli esiti."""
    rng = np.random.default_rng(4)
    p = expit(rng.normal(-2.2, 1.0, 20000))
    y = (rng.random(20000) < p).astype(int)
    good = pd.DataFrame(cu.flexible_curve(y, p))
    assert (good["observed"] - good["predicted"]).abs().max() < 0.03
    assert ((good["low"] <= good["observed"]) & (good["observed"] <= good["high"])).all()
    extreme = expit(2 * logit(p))
    bad = pd.DataFrame(cu.flexible_curve(y, extreme))
    assert (bad["observed"] - bad["predicted"]).abs().max() > 0.1


def test_calibration_row_adds_oe_and_intervals():
    rng = np.random.default_rng(5)
    p = expit(rng.normal(-2.0, 0.8, 1500))
    y = (rng.random(1500) < p).astype(int)
    codes = np.zeros(1500, dtype=int)
    samples = list(ev.stratified_indices(codes, 50, 0))
    row = cu.calibration_row(y, p, samples)
    assert row["oe_ratio"] == pytest.approx(y.sum() / p.sum())
    assert row["slope"] == pytest.approx(calibration(y, p)["slope"])
    for name in ("intercept", "slope", "brier", "oe_ratio"):
        assert row[f"{name}_low"] <= row[name] <= row[f"{name}_high"]


def test_level_stratified_bootstrap_fixes_the_number_of_events(synthetic):
    """Ogni livello sopra "basso" e' positivo per definizione: stratificando sul livello il numero
    di eventi non varia mai, col bootstrap semplice si'. Per questo l'intervallo dell'O:E
    stratificato e' piu' stretto: e' il motivo delle colonne *_simple."""
    _, train, _ = synthetic
    y = target(train).to_numpy()
    codes = np.asarray(pd.Categorical(kdigo_level(train), LEVELS, ordered=True).codes)
    assert np.array_equal(y, (codes > 0).astype(int))
    stratified = list(ev.stratified_indices(codes, 200, 0))
    simple = list(cu.simple_indices(len(y), 200, 0))
    assert len({int(y[idx].sum()) for idx in stratified}) == 1
    assert len({int(y[idx].sum()) for idx in simple}) > 10
    p = np.clip(y * 0.3 + 0.1, 0, 1)
    row = cu.calibration_row(y, p, stratified, simple_samples=simple)
    width = row["oe_ratio_high"] - row["oe_ratio_low"]
    assert row["oe_ratio_high_simple"] - row["oe_ratio_low_simple"] > 2 * width


def test_calibration_bins_use_deciles():
    rng = np.random.default_rng(6)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    bins = pd.DataFrame(cu.calibration_bins(y, p, bins=10))
    assert len(bins) == 10 and bins["n"].sum() == 1000
    assert bins["predicted"].is_monotonic_increasing


def test_check_calibration_against_phase_b(tables):
    mine = tables["calibration"]
    reference = mine[mine["group"] == "tutti"][["model", "intercept", "slope", "brier"]].copy()
    reference["technique"], reference["probabilities"] = "none", "grezza"
    cu.check_calibration(mine, reference)
    reference.loc[reference.index[0], "slope"] += 0.01
    with pytest.raises(RuntimeError):
        cu.check_calibration(mine, reference)


# --- confronto esterno ------------------------------------------------------------------------

def test_albuminuria_ignores_egfr(synthetic):
    """Bersaglio di Bragg-Gresham: i 30 soggetti con eGFR < 60 e ACR normale sono positivi per la
    tesi e negativi per la sola albuminuria."""
    _, train, _ = synthetic
    composite, albumin = target(train).to_numpy(), cu.albuminuria(train)
    assert np.all(albumin <= composite)
    assert int(np.sum(composite - albumin)) == 30


def test_external_rows_at_fixed_sensitivity():
    rng = np.random.default_rng(7)
    y = (rng.random(600) < 0.1).astype(int)
    p = np.clip(y * 0.15 + rng.random(600) * 0.5, 0, 1)
    rows = cu.external_rows(y, p, sensitivity=0.85, thresholds=[0.05, 0.07])
    assert [r["rule"] for r in rows] == ["sensibilita' 0.85", "soglia 0.05", "soglia 0.07"]
    assert rows[0]["threshold"] == ev.threshold_at_sensitivity(y, p, 0.85)
    assert rows[0]["detected"] >= 0.85
    assert rows[1]["screened"] == pytest.approx(np.mean(p >= 0.05))


def test_external_comparison_uses_non_diabetics_for_bragg_gresham(synthetic, tables):
    _, train, _ = synthetic
    rows = tables["external_comparison"].query("setting == 'bragg_gresham'")
    non_diabetic = train[COLUMNS["diabetes"]].to_numpy() == 0
    assert (rows["n"] == int(non_diabetic.sum())).all()
    assert (rows["positives"] == int(cu.albuminuria(train)[non_diabetic].sum())).all()


def test_operating_row_counts_severe_cases():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.06, 0.01, 0.5, 0.02, 0.07])
    codes = np.array([3, 2, 1, 0, 0, 0])  # molto alto, alto, moderato, bassi
    row = cu.operating_row(y, p, codes, 0.05)
    assert row["severe_n"] == 2 and row["severe_detected"] == 2
    assert row["tests_per_100"] == pytest.approx(100 * 4 / 6)
    assert row["missed_per_100"] == pytest.approx(100 / 6)


def test_test_set_is_never_read():
    source = Path(cu.__file__).read_text(encoding="utf-8")
    assert "test.csv" not in source
    assert "train.csv" in source
