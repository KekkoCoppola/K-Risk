"""Test delle tecniche di bilanciamento della Fase B (dati sintetici e fold imputati con la mediana)."""
import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

import src.data.augmented as aug
from src.data.imputed import build_cache as build_imputed, load_fold
from src.data.preprocess import FEATURE_SETS, feature_types
from src.data.split import PROCESSED

METHOD = "mediana"


@pytest.fixture(scope="module")
def data():
    """Training sintetico: 2 numeriche, 1 binaria, 1 ordinale; positivi in 3 livelli."""
    rng = np.random.default_rng(0)
    n = 600
    codes = rng.choice(4, n, p=[0.85, 0.1, 0.035, 0.015])
    X = pd.DataFrame({"Age": rng.normal(size=n) + codes, "Bpsys": rng.normal(size=n),
                      "DM": rng.integers(0, 2, n), "Smoking": rng.integers(0, 3, n)},
                     index=np.arange(1000, 1000 + n))
    return X, (codes > 0).astype(int), codes


@pytest.fixture(scope="module")
def imputed(tmp_path_factory):
    train = pd.read_csv(PROCESSED / "train.csv")
    directory = tmp_path_factory.mktemp("imputed")
    build_imputed(train, "main", SimpleImputer(strategy="median"), directory, METHOD)
    return train, (directory, METHOD)


def test_class_weights_balance_classes(data):
    _, y, _ = data
    w = aug.class_weights(y)
    assert w.mean() == pytest.approx(1)
    assert w[y == 1].sum() == pytest.approx(w[y == 0].sum())


def test_level_weights_follow_costs(data):
    _, y, codes = data
    w = aug.level_weights(y, codes, {"moderato": 1, "alto": 2, "molto alto": 3})
    assert w.mean() == pytest.approx(1)
    assert w[y == 1].sum() == pytest.approx(w[y == 0].sum())
    per_level = [w[codes == i][0] for i in (1, 2, 3)]
    assert np.allclose(np.array(per_level) / per_level[0], [1, 2, 3])


@pytest.mark.parametrize("technique", ["undersampling", "oversampling", "smote_nc", "smote_nc_level"])
def test_sampling_reaches_ratio_and_keeps_real_rows(data, technique):
    X, y, codes = data
    X_bal, y_bal, source = aug.balance(technique, X, y, codes, seed=0)
    assert np.sum(y_bal == 1) == np.sum(y_bal == 0)
    real = source != aug.SYNTHETIC
    # righe reali identiche agli originali, con la loro etichetta
    assert np.allclose(X_bal.to_numpy()[real], X.loc[source[real]].to_numpy())
    assert np.array_equal(y_bal[real], y[np.searchsorted(X.index, source[real])])
    # categoriche sintetiche solo fra i valori osservati
    for column in ["DM", "Smoking"]:
        assert set(X_bal[column]) <= set(X[column])
    assert list(X_bal.columns) == list(X.columns)


def test_smote_level_quotas_follow_levels(data):
    X, y, codes = data
    X_bal, y_bal, source = aug.smote_nc_level(X, y, codes, seed=0)
    counts = np.array([np.sum(codes == i) for i in (1, 2, 3)])
    quotas = aug.level_quotas(counts, int(np.sum(y == 0) - y.sum()))
    # tutte le righe mancanti generate, con le proporzioni dei livelli (a meno di un soggetto)
    assert quotas.sum() == np.sum(source == aug.SYNTHETIC)
    assert np.all(np.abs(quotas - quotas.sum() * counts / counts.sum()) <= 1)


def test_smote_level_interpolates_within_level():
    """Livelli ben separati in Age: ogni riga sintetica cade nell'intervallo di un solo livello."""
    rng = np.random.default_rng(1)
    codes = np.r_[np.zeros(200, int), np.ones(30, int), np.full(10, 3)]
    X = pd.DataFrame({"Age": np.r_[rng.normal(0, 1, 200), rng.normal(10, 1, 30), rng.normal(50, 1, 10)],
                      "DM": rng.integers(0, 2, 240)})
    y = (codes > 0).astype(int)
    X_bal, _, source = aug.smote_nc_level(X, y, codes, seed=0)
    age = X_bal.loc[source == aug.SYNTHETIC, "Age"]
    inside = [(age >= X.Age[codes == i].min()) & (age <= X.Age[codes == i].max()) for i in (1, 3)]
    assert np.all(inside[0] | inside[1])


def test_level_quotas_exact_total():
    assert aug.level_quotas([283, 40, 17], 2800).sum() == 2800
    assert sorted(aug.level_quotas([1, 1, 1], 4)) == [1, 1, 2]


def test_tiny_level_is_duplicated():
    X = pd.DataFrame({"Age": np.arange(12.0), "DM": [0, 1] * 6}, index=range(12))
    y = np.r_[np.zeros(9, int), np.ones(3, int)]
    codes = np.r_[np.zeros(9, int), 1, 1, 3]  # un solo "molto alto"
    X_bal, y_bal, source = aug.smote_nc_level(X, y, codes, seed=0)
    assert np.sum(y_bal == 1) == 9 and np.sum(source == aug.SYNTHETIC) == 6


def test_cache_is_repeatable_and_validation_real(imputed, tmp_path):
    train, imp = imputed
    for directory in (tmp_path / "a", tmp_path / "b"):
        aug.build_cache(train, "smote_nc", "main", directory, imputed=imp, log=None, outer_only=True)
    X_a, y_a = aug.load_balanced("smote_nc", "main", 0, None, tmp_path / "a")
    X_b, _ = aug.load_balanced("smote_nc", "main", 0, None, tmp_path / "b")
    assert np.array_equal(X_a.to_numpy(), X_b.to_numpy())
    X_real, _ = load_fold("main", 0, directory=imp[0], method=METHOD)
    assert np.sum(y_a == 1) == np.sum(y_a == 0)
    assert list(X_a.columns) == list(X_real.columns)
    # nessuna colonna di livello o di classe fra le variabili; con outer_only niente fold interni
    assert "condition" not in X_a.columns and "kdigo_level" not in X_a.columns
    assert not aug.cache_path("smote_nc", "main", 0, 0, tmp_path / "a").exists()
    # pesi: stesso X del fold imputato
    y_all, codes_all = aug.labels(train)
    X_w, _, w = aug.training_set("level_weight", "main", y_all, codes_all, 0, None, imputed=imp)
    assert X_w.equals(X_real) and w.mean() == pytest.approx(1)


def test_ctgan_smoke(data):
    pytest.importorskip("ctgan")
    X, y, codes = data
    spec = {"epochs": 1, "batch_size": 100, "max_rounds": 200}
    X_bal, y_bal, _ = aug.ctgan_sample(X, y, 0, spec, codes)
    assert np.sum(y_bal == 1) == np.sum(y_bal == 0)
    assert list(X_bal.columns) == list(X.columns)
    assert set(X_bal["DM"]) <= set(X["DM"])


def test_does_not_read_test_set():
    source = inspect.getsource(aug)
    assert "test.csv" not in source and "load_split" not in source


def test_categorical_indices_match_config():
    columns = FEATURE_SETS["main"]
    _, categorical = feature_types(columns)
    assert [columns[i] for i in aug.categorical_indices(columns)] == categorical
