import pytest

from src.data.load import DIABETES, ID, binary_target, load_labeled
from src.data.split import split_dataset, strata


@pytest.fixture(scope="module")
def df():
    return load_labeled()


def test_target_rows(df):
    assert len(df) == 5802
    assert binary_target(df).sum() == 456


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


def test_diabetic_positives_in_test(df):
    _, test = split_dataset(df)
    assert ((binary_target(test) == 1) & (test[DIABETES] == 1)).sum() == 20
