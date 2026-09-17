import pandas as pd

from src.config import CONFIG, resolve

TARGET = CONFIG["columns"]["target"]
DIABETES = CONFIG["columns"]["diabetes"]
ID = CONFIG["columns"]["id"]


def load_raw():
    return pd.read_excel(resolve(CONFIG["data"]["raw"]))


def load_labeled():
    """Dataset con target valorizzato (righe con DN mancante escluse)."""
    df = load_raw()
    df = df[df[TARGET].notna()].reset_index(drop=True)
    assert df[ID].is_unique
    return df


def binary_target(df):
    return (df[TARGET] > 0).astype(int)
