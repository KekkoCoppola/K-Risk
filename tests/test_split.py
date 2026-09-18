import pytest

from src.data.kidney import egfr, kdigo_level, target
from src.data.load import DIABETES, ID, load_eligible
from src.data.split import split_dataset, strata


@pytest.fixture(scope="module")
def df():
    return load_eligible()


def test_eligible_rows(df):
    assert len(df) == 5801
    assert target(df).sum() == 567


def test_egfr_is_plausible(df):
    values = egfr(df)
    assert values.between(1, 200).all()


def test_kdigo_levels(df):
    counts = kdigo_level(df).value_counts()
    assert counts["basso"] == 5234
    assert counts["moderato"] == 473
    assert counts["alto"] == 67
    assert counts["molto alto"] == 27


def test_split_is_reproducible(df):
    train_a, test_a = split_dataset(df)
    train_b, test_b = split_dataset(df)
    assert train_a[ID].tolist() == train_b[ID].tolist()
    assert test_a[ID].tolist() == test_b[ID].tolist()


def test_split_is_disjoint_and_complete(df):
    train, test = split_dataset(df)
    assert set(train[ID]).isdisjoint(test[ID])
    assert len(train) + len(test) == len(df)


def test_strata_proportions(df):
    train, test = split_dataset(df)
    for label in strata(df).unique():
        p_train = (strata(train) == label).mean()
        p_test = (strata(test) == label).mean()
        assert abs(p_train - p_test) < 0.005


def test_severe_cases_are_split_proportionally(df):
    train, test = split_dataset(df)
    assert (kdigo_level(train) == "molto alto").sum() == 21
    assert (kdigo_level(test) == "molto alto").sum() == 6


def test_diabetic_positives_in_test(df):
    _, test = split_dataset(df)
    assert ((target(test) == 1) & (test[DIABETES] == 1)).sum() == 23
