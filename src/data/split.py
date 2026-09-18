from functools import reduce

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CONFIG, resolve
from src.data.kidney import kdigo_level, target
from src.data.load import DIABETES, load_eligible

PROCESSED = resolve(CONFIG["data"]["processed"])
TEST_SIZE = CONFIG["split"]["test_size"]
STRATIFY = CONFIG["split"]["stratify"]
SEED = CONFIG["seed"]

DERIVED = {"kdigo_level": kdigo_level, "target": target}


def column(df, name):
    """Colonna del dataset oppure variabile calcolata (kdigo_level, target)."""
    if name in DERIVED:
        return DERIVED[name](df)
    return df[name]


def strata(df, columns=STRATIFY):
    """Etichetta combinata delle colonne di stratificazione."""
    parts = (column(df, c).astype(str) for c in columns)
    return reduce(lambda a, b: a + "_" + b, parts)


def split_dataset(df, test_size=TEST_SIZE, seed=SEED, stratify=STRATIFY):
    """stratify: lista di colonne (da config) oppure None per split non stratificato."""
    labels = strata(df, stratify) if stratify else None
    return train_test_split(
        df, test_size=test_size, stratify=labels, random_state=seed
    )


def save_split(train, test):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)


def load_split():
    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    return train, test


def summary(train, test):
    rows = []
    for name, d in [("train", train), ("test", test)]:
        y = target(d) == 1
        dm = d[DIABETES] == 1
        rows.append({
            "set": name,
            "n": len(d),
            "positivi": int(y.sum()),
            "positivi %": round(100 * y.mean(), 2),
            "DM": int(dm.sum()),
            "DM positivi": int((dm & y).sum()),
        })
    return pd.DataFrame(rows)


def kdigo_summary(train, test):
    counts = pd.DataFrame({
        "train": kdigo_level(train).value_counts(),
        "test": kdigo_level(test).value_counts(),
    })
    counts["% nel test"] = (100 * counts["test"] / counts.sum(axis=1)).round(1)
    return counts


def smd(a, b):
    """Differenza media standardizzata fra due serie."""
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    sd = ((a.var() + b.var()) / 2) ** 0.5
    return abs(a.mean() - b.mean()) / sd if sd > 0 else 0.0


def balance_table(train, test, columns):
    rows = {"target": smd(target(train), target(test))}
    rows[DIABETES] = smd(train[DIABETES], test[DIABETES])
    for c in columns:
        rows[c] = smd(train[c], test[c])
    return pd.Series(rows, name="SMD").round(3)


if __name__ == "__main__":
    train, test = split_dataset(load_eligible())
    save_split(train, test)
    print(summary(train, test).to_string(index=False))
    print()
    print(kdigo_summary(train, test).to_string())
    print()
    print(balance_table(train, test, CONFIG["analytics"]["balance_columns"]).to_string())
