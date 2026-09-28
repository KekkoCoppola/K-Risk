"""Controllo dei numeri di valorizzazione_tesi.md (audit del 22-23/09/2026): ogni numero dichiarato
in 05_numeri.csv viene ricalcolato dal CSV del progetto e confrontato con il valore scritto nel
documento (stesso arrotondamento).
Controlla anche che il valore compaia davvero nel testo di valorizzazione_tesi.md.

Uso, dalla radice del repository:
    python docs/audit/check_numbers.py . docs/audit/05_numeri.csv valorizzazione_tesi.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse(value):
    """Numero scritto all'italiana: punto per le migliaia, virgola per i decimali."""
    import re
    text = str(value).strip().replace("−", "-").replace("+", "").replace("%", "").replace(" ", "")
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")
    return float(text)


def load(repo, name):
    table = pd.read_csv(repo / name)
    if "grey_zone" in table.columns:
        table["grey_zone"] = table["grey_zone"].fillna("")
    return table


def compute(repo, row):
    query = str(row.get("filtro", "") or "").strip()
    column = row["colonna"]
    how = str(row["aggregazione"]).strip()
    if how == "diff":
        first, second = query.split(";;")
        a = load(repo, row["file"]).query(first)[column].astype(float)
        b = load(repo, row["file"]).query(second)[column].astype(float)
        if len(a) != 1 or len(b) != 1:
            raise ValueError(f"diff: i filtri selezionano {len(a)} e {len(b)} righe")
        return a.iloc[0] - b.iloc[0]
    table = load(repo, row["file"])
    if query and query.lower() != "nan":
        table = table.query(query)
    values = table[column].astype(float)
    if how == "value":
        if len(values) != 1:
            raise ValueError(f"il filtro seleziona {len(values)} righe, non 1")
        return values.iloc[0]
    if how in ("min", "max", "mean", "sum", "median"):
        return getattr(values, how)()
    raise ValueError(f"aggregazione sconosciuta {how}")


def main(repo, numbers, document):
    repo = Path(repo)
    doc = Path(document).read_text(encoding="utf-8")
    table = pd.read_csv(numbers, dtype=str, keep_default_na=False)
    bad, missing = [], []
    for _, row in table.iterrows():
        try:
            decimals = int(float(row["arrotondamento"]))
            got = compute(repo, row)
            scale = 100 if str(row["valore_nel_documento"]).strip().endswith("%") else 1
            expected = parse(row["valore_nel_documento"])
            if not np.isclose(round(got * scale, decimals), expected, atol=0.51 * 10 ** -decimals):
                bad.append((row["id"], row["valore_nel_documento"], round(got * scale, decimals + 2),
                            row["file"], row["filtro"]))
        except Exception as error:  # noqa: BLE001 - ogni riga non ricalcolabile va segnalata
            bad.append((row["id"], row["valore_nel_documento"], f"ERRORE: {error}", row["file"], row["filtro"]))
        if str(row["valore_nel_documento"]).strip() not in doc:
            missing.append((row["id"], row["valore_nel_documento"]))
    print(f"{len(table)} numeri controllati; discrepanze: {len(bad)}; valori non trovati nel testo: {len(missing)}")
    for item in bad:
        print("DISCREPANZA", " | ".join(map(str, item)))
    for item in missing:
        print("NON NEL TESTO", " | ".join(map(str, item)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
