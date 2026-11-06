import json
import math

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

import src.models.phase_e as phase_e
from src.data.folds import INNER, OUTER
from src.data.imputed import build_cache, load_fold
from src.data.kidney import egfr, target
from src.data.split import PROCESSED
from src.models.phase_d import components, loader
from src.models.phase_e import (ARMS, CONTINUOUS, REFERENCES, REUSED, RULES, SETTINGS, SPACES, ZOO, build,
                                classification, creatinine_cut, cross_fitted, evaluate, exceedance,
                                expected_calibration_error, joint_probability, joint_scores, log_targets,
                                margin_target, net_benefit, operating_row, partial_auc_high_sensitivity,
                                read_record, run_combined, run_continuous, run_reused, run_zoo, score,
                                source_path, subgroup_masks, top_rows)
from src.models.zoo import SEED, suggest

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


@pytest.fixture
def one_fold(monkeypatch):
    """Solo il primo fold esterno: i test restano veloci, la logica è la stessa."""
    monkeypatch.setattr(phase_e, "OUTER", 1)


@pytest.fixture
def two_folds(monkeypatch):
    """Due fold esterni: il minimo per il cross-fitting di stacking e valutazione."""
    monkeypatch.setattr(phase_e, "OUTER", 2)


def test_protocol_registered():
    assert SETTINGS["n_trials"] == 30 and SETTINGS["evaluation"]["min_difference"] == 0.01
    assert len(ZOO) == 11 and all(name in SPACES for name in ZOO)
    assert set(REUSED) == {"lr_penalized", "random_forest", "xgboost", "catboost", "lightgbm", "ebm", "tabpfn"}
    assert set(ARMS) == {"combinato_lgbm", "combinato_lgbm_etero", "combinato_mlp", "margine_kdigo"}
    units = CONTINUOUS["margin_units"]
    assert math.isclose(units["acr"], math.log(10), abs_tol=1e-8)
    assert math.isclose(units["creatinine"], math.log(60 / 45) / 1.2, abs_tol=1e-8)
    for name in REUSED:
        for outer in range(OUTER):
            assert source_path(name, outer).exists()


def test_creatinine_cut_gives_egfr_60(train):
    cut = creatinine_cut(train)
    at_cut = train.assign(SCRE=np.exp(cut))
    assert np.allclose(egfr(at_cut), 60)
    _, low_egfr = components(train)
    assert np.array_equal(np.log(train["SCRE"].to_numpy()) > cut, low_egfr == 1)


def test_margin_target_matches_composite(train):
    assert np.array_equal((margin_target(train) >= 0).astype(int), target(train).to_numpy())


def test_joint_probability_by_hand():
    # coppie (e_a, e_c): (-1, 0) sotto entrambi i limiti 0,5; (0, 1) no; (1, -1) no -> p = 1 - 1/3
    p = joint_probability([0.5], [0.5], [-1, 0, 1], [0, 1, -1])
    assert np.isclose(p[0], 2 / 3)
    # limiti altissimi: tutte le coppie sotto, probabilità 0
    assert joint_probability([10], [10], [-1, 0, 1], [0, 1, -1])[0] == 0


def test_exceedance_by_hand():
    assert np.isclose(exceedance(np.array([0.5]), [-1, 0, 1])[0], 2 / 3)
    assert exceedance(np.array([-5.0]), [-1, 0, 1])[0] == 0


def test_inner_probabilities_use_only_other_inner_folds():
    """La probabilità di una riga interna non deve dipendere dai residui del suo fold interno."""
    rng = np.random.default_rng(0)
    n = 50
    mu_in, t_in = rng.normal(size=(n, 2)), rng.normal(size=(n, 2))
    fold_in, cut_in = np.repeat(np.arange(5), 10), rng.normal(size=n)
    mu_out, cut_out = rng.normal(size=(7, 2)), rng.normal(size=7)
    p_in, p_out = joint_scores(mu_in, t_in, cut_in, fold_in, mu_out, cut_out)
    changed = t_in.copy()
    changed[fold_in == 0] += 100
    q_in, q_out = joint_scores(mu_in, changed, cut_in, fold_in, mu_out, cut_out)
    assert np.array_equal(p_in[fold_in == 0], q_in[fold_in == 0])
    assert not np.array_equal(p_in[fold_in != 0], q_in[fold_in != 0])
    assert not np.array_equal(p_out, q_out)


def test_classification_by_hand():
    row = classification([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert (row["tp"], row["fn"], row["fp"], row["tn"]) == (1, 1, 1, 2)
    assert row["sensitivity"] == 0.5 and np.isclose(row["specificity"], 2 / 3)
    assert row["ppv"] == 0.5 and np.isclose(row["npv"], 2 / 3) and row["alert_rate"] == 0.4


def test_every_new_family_fits_and_scores(train, cache):
    X_tr, X_va = load_fold("main", 0, 0, cache, METHOD)
    y = target(train).to_numpy()
    X_tr, y_tr = X_tr.iloc[:800], y[X_tr.index[:800]]
    for name in ZOO:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=SEED))
        params = suggest(study.ask(), SPACES[name])
        s = score(name, build(name, params).fit(X_tr, y_tr), X_va.iloc[:100])
        assert s.shape == (100,) and np.isfinite(s).all(), name


def test_zoo_keeps_inner_predictions_of_the_best_trial(train, cache, tmp_path, one_fold):
    y = target(train).to_numpy()
    load = loader(train, y, "imputed", directory=cache, method=METHOD)
    run_zoo("lda", train, output=tmp_path, n_trials=3, load=load)
    record = read_record("lda", 0, tmp_path)
    X_tr, _, X_va, _ = load(0, None)
    assert sorted(record["inner_rows"]) == sorted(X_tr.index) and record["rows"] == X_va.index.tolist()
    assert set(record["inner_rows"]).isdisjoint(record["rows"])
    # le previsioni interne conservate coincidono con quelle di un riaddestramento con i parametri scelti
    refit = np.concatenate([build("lda", record["params"]).fit(a, b).predict_proba(c)[:, 1]
                            for a, b, c, _ in (load(0, j) for j in range(INNER))])
    assert np.allclose(record["inner_score"], refit)


def test_reused_family_copies_outer_predictions(train, cache, tmp_path, one_fold):
    load = loader(train, target(train).to_numpy(), "imputed", directory=cache, method=METHOD)
    run_reused("xgboost", train, output=tmp_path, load=load)
    record = read_record("xgboost", 0, tmp_path)
    source = json.loads(source_path("xgboost", 0).read_text(encoding="utf-8"))
    assert record["score"] == source["probability"] and record["params"] == source["params"]
    assert record["rows"] == source["rows"]


def test_continuous_arms_and_combination(train, cache, tmp_path, two_folds):
    t = log_targets(train)
    load_t = loader(train, t, "imputed", directory=cache, method=METHOD)
    run_continuous("combinato_lgbm", train, output=tmp_path, n_trials=2, load=load_t)
    run_continuous("combinato_lgbm_etero", train, output=tmp_path, load=load_t)
    run_continuous("combinato_mlp", train, output=tmp_path, n_trials=1, load=load_t)
    load_m = loader(train, margin_target(train), "imputed", directory=cache, method=METHOD)
    run_continuous("margine_kdigo", train, output=tmp_path, n_trials=2, load=load_m)
    for arm in ARMS:
        for outer in range(2):
            record = read_record(arm, outer, tmp_path)
            p = np.asarray(record["score"])
            assert len(p) == len(record["rows"]) and ((p >= 0) & (p <= 1)).all(), arm
            assert set(record["inner_rows"]).isdisjoint(record["rows"]), arm
    assert np.asarray(read_record("combinato_lgbm", 0, tmp_path)["mu"]).shape == (len(p), 2)
    # stacking e media dei ranghi su due classificatori e sul braccio continuo, sui due fold calcolati
    # (solo le loro righe: il cross-fitting usa l'altro fold)
    y = target(train).to_numpy()
    load = loader(train, y, "imputed", directory=cache, method=METHOD)
    for name in ("lda", "naive_bayes"):
        run_zoo(name, train, output=tmp_path, n_trials=2, load=load)
    rows = sorted(read_record("lda", 0, tmp_path)["rows"] + read_record("lda", 1, tmp_path)["rows"])
    subset = train.iloc[rows].reset_index(drop=True)
    for name in ("lda", "naive_bayes", "combinato_lgbm"):
        for outer in range(2):
            record = read_record(name, outer, tmp_path)
            record["rows"] = [rows.index(r) for r in record["rows"]]
            phase_e.save_record(name, outer, record, tmp_path / "subset")
    run_combined(subset, output=tmp_path / "subset", bases=["lda", "naive_bayes", "combinato_lgbm"])
    stacking = read_record("stacking", 0, tmp_path / "subset")
    assert set(stacking["coefficients"]) == {"lda", "naive_bayes", "combinato_lgbm",
                                             "combinato_lgbm: margine ACR", "combinato_lgbm: margine eGFR"}
    assert stacking["rows"] == sorted(read_record("lda", 0, tmp_path / "subset")["rows"])
    ranks = np.asarray(read_record("rank_mean", 1, tmp_path / "subset")["score"])
    assert ((ranks >= 0) & (ranks <= 1)).all()


def test_cross_fitting_ignores_labels_of_the_evaluated_fold():
    """Ricalibrazione e soglie del fold k stimate sugli altri fold: cambiare le etichette del fold k
    non cambia né le probabilità né le allerte del fold k."""
    rng = np.random.default_rng(1)
    n = 1000
    fold = np.repeat(np.arange(OUTER), n // OUTER)
    y = rng.integers(0, 2, n)
    s = rng.uniform(size=n) + 0.3 * y
    q, flags, _ = cross_fitted("lda", s, fold, y, RULES)
    flipped = y.copy()
    flipped[fold == 0] = 1 - flipped[fold == 0]
    r, other, _ = cross_fitted("lda", s, fold, flipped, RULES)
    assert np.array_equal(q[fold == 0], r[fold == 0])
    for rule in RULES:
        assert np.array_equal(flags[rule][fold == 0], other[rule][fold == 0])
    assert not np.array_equal(q[fold == 1], r[fold == 1])


def test_cross_fitting_ignores_the_scale_of_each_fold():
    """Modelli diversi per fold possono dare punteggi su scale diverse: una trasformazione monotona
    dei punteggi di un solo fold non deve cambiare probabilità né allerte."""
    rng = np.random.default_rng(2)
    n = 1000
    fold = np.repeat(np.arange(OUTER), n // OUTER)
    y = rng.integers(0, 2, n)
    s = rng.uniform(size=n) + 0.3 * y
    q, flags, _ = cross_fitted("lda", s, fold, y, RULES)
    rescaled = s.copy()
    rescaled[fold == 2] = 100 * rescaled[fold == 2] ** 3 + 7
    r, other, _ = cross_fitted("lda", rescaled, fold, y, RULES)
    assert np.allclose(q, r)
    for rule in RULES:
        assert np.array_equal(flags[rule], other[rule])


def test_operating_row_by_hand():
    # livelli: 0 basso, 1 moderato, 2 alto, 3 molto alto; gravi = alto e molto alto
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    codes = np.array([1, 2, 3, 1, 0, 0, 0, 0, 0, 0])
    flagged = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], bool)
    row = operating_row(y, flagged, codes)
    assert (row["tp"], row["fn"], row["fp"], row["tn"]) == (3, 1, 1, 5)
    assert row["severe_sensitivity"] == 1.0 and row["sensitivity_moderato"] == 0.5
    assert np.isclose(row["npv"], 5 / 6) and row["no_information_rate"] == 0.6 and row["accuracy"] == 0.8
    assert np.isclose(row["lr_positive"], 0.75 / (1 / 6)) and np.isclose(row["lr_negative"], 0.25 / (5 / 6))
    assert np.isclose(row["f2"], 5 * 3 / (5 * 3 + 4 * 1 + 1))
    # kappa: accordo osservato 0,8, atteso (4*4 + 6*6) / 100 = 0,52
    assert np.isclose(row["kappa"], (0.8 - 0.52) / 0.48)
    assert row["tests_avoided_per_100"] == 60 and row["missed_per_1000"] == 100


def test_top_rows_by_hand():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    q = np.linspace(1, 0, 10)
    codes = np.array([2, 1, 0, 0, 0, 0, 0, 0, 0, 3])
    row = top_rows(y, q, codes, fractions=[0.2])[0]
    assert row["n_top"] == 2 and row["precision"] == 1.0 and np.isclose(row["lift"], 1 / 0.3)
    assert np.isclose(row["recall"], 2 / 3) and row["severe_recall"] == 0.5


def test_net_benefit_by_hand():
    y = np.array([1, 0, 0, 0])
    q = np.array([0.9, 0.6, 0.1, 0.1])
    model, everyone = net_benefit(y, q, 0.5)
    assert np.isclose(model, 1 / 4 - 1 / 4) and np.isclose(everyone, 1 / 4 - 3 / 4)


def test_partial_auc_and_calibration_error():
    y = np.array([0, 1] * 50)
    assert np.isclose(partial_auc_high_sensitivity(y, y.astype(float)), 1.0)
    # ogni decile di q contiene 5 positivi su 10 e probabilità 0,5: errore nullo
    assert np.isclose(expected_calibration_error(y, np.full(100, 0.5) + np.linspace(-1e-6, 1e-6, 100)), 0.0,
                      atol=1e-5)
    # probabilità costante 0,2 con prevalenza 0,5: errore 0,3
    assert np.isclose(expected_calibration_error(y, np.full(100, 0.2) + np.linspace(-1e-6, 1e-6, 100)), 0.3,
                      atol=1e-5)


def test_subgroups_partition_the_training(train):
    masks = subgroup_masks(train)
    for a, b in [("diabetici", "non diabetici"), ("donne", "uomini"), ("ipertensione nota", "ipertensione non nota")]:
        assert np.array_equal(masks[a], ~masks[b])
    ages = masks["età < 40"].astype(int) + masks["età 40-59"] + masks["età >= 60"]
    assert (ages == 1).all()


@pytest.mark.skipif(not phase_e.done("random_forest"), reason="record della fase E non ancora calcolati")
def test_evaluate_on_the_references(train, tmp_path):
    """Pannello completo sui tre riferimenti (record veri copiati in una cartella temporanea)."""
    for name in REFERENCES:
        for outer in range(OUTER):
            phase_e.save_record(name, outer, read_record(name, outer), tmp_path)
    tables = evaluate(train, REFERENCES, output=tmp_path)
    metrics = tables["metrics"].set_index("model")
    assert list(metrics.index) == REFERENCES
    assert ((metrics["auc"] > 0.5) & (metrics["auc"] < 1)).all()
    assert (tables["operating_points"].groupby("model")["rule"].nunique() == len(RULES)).all()
    assert len(tables["subgroups"]) == 9 * len(REFERENCES)
    # a sensibilità fissata dagli altri fold la sensibilità ottenuta resta vicina all'obiettivo
    point = tables["operating_points"].query("rule == 'sens90'")
    assert ((point["sensitivity"] > 0.8) & (point["sensitivity"] < 0.97)).all()


@pytest.mark.skipif(not phase_e.done("random_forest"), reason="record della fase E non ancora calcolati")
def test_reused_raw_auc_reproduces_phase_d(train):
    """Controllo di coerenza sui record veri: l'AUROC grezza dei riferimenti riusati è quella della
    Fase D (analytics/phase_d/discrimination.csv)."""
    reference = pd.read_csv(phase_e.resolve("analytics/phase_d/discrimination.csv")).set_index("model")["auc"]
    y = target(train).to_numpy()
    for name in ("lr_penalized", "random_forest", "xgboost"):
        p = np.full(len(y), np.nan)
        for outer in range(OUTER):
            record = read_record(name, outer)
            p[record["rows"]] = record["score"]
        assert np.isclose(roc_auc_score(y, p), reference[name], atol=1e-12), name
