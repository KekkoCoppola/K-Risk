from functools import reduce

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CONFIG, resolve
from src.data.load import DIABETES, TARGET, binary_target, load_labeled

PROCESSED = resolve(CONFIG["data"]["processed"])
TEST_SIZE = CONFIG["split"]["test_size"]
STRATIFY = CONFIG["split"]["stratify"]
SEED = CONFIG["seed"]


def strata(df, columns=STRATIFY):
    """Etichetta combinata delle colonne di stratificazione (target binarizzato)."""
    parts = [binary_target(df) if c == TARGET else df[c] for c in columns]
    return reduce(lambda a, b: a + "_" + b, (p.astype(str) for p in parts))


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
        dn = binary_target(d) == 1
        dm = d[DIABETES] == 1
        rows.append({
            "set": name,
            "n": len(d),
            "DN+": int(dn.sum()),
            "DN+ %": round(100 * dn.mean(), 2),
            "DM": int(dm.sum()),
            "DM & DN+": int((dm & dn).sum()),
        })
    return pd.DataFrame(rows)


def smd(a, b):
    """Differenza media standardizzata fra due serie."""
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    sd = ((a.var() + b.var()) / 2) ** 0.5
    return abs(a.mean() - b.mean()) / sd if sd > 0 else 0.0


def balance_table(train, test, columns):
    rows = {"DN > 0": smd(binary_target(train), binary_target(test))}
    rows[DIABETES] = smd(train[DIABETES], test[DIABETES])
    for c in columns:
        rows[c] = smd(train[c], test[c])
    return pd.Series(rows, name="SMD").round(3)


if __name__ == "__main__":
    train, test = split_dataset(load_labeled())
    save_split(train, test)
    print(summary(train, test).to_string(index=False))
    print()
    cols = CONFIG["analytics"]["balance_columns"]
    print(balance_table(train, test, cols).to_string())
