import pandas as pd

from src.config import CONFIG, resolve

COLUMNS = CONFIG["columns"]
ID = COLUMNS["id"]
DIABETES = COLUMNS["diabetes"]
REQUIRED = [COLUMNS["creatinine"], COLUMNS["acr"], COLUMNS["age"], COLUMNS["gender"]]


def load_raw():
    return pd.read_excel(resolve(CONFIG["data"]["raw"]))


def load_eligible():
    """Soggetti per cui il target KDIGO è calcolabile (creatinina e ACR presenti)."""
    df = load_raw()
    df = df.dropna(subset=REQUIRED).reset_index(drop=True)
    assert df[ID].is_unique
    return df
