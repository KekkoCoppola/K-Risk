"""Calcolo dell'eGFR e della classificazione KDIGO.

La colonna `GFR` del dataset non corrisponde a nessuna equazione standard
(CKD-EPI 2009/2021, MDRD, MDRD cinese, Cockcroft-Gault) e contiene valori fuori
scala, quindi l'eGFR viene ricalcolato con CKD-EPI 2021 come richiesto da KDIGO.
"""
import numpy as np
import pandas as pd

from src.config import CONFIG

COLUMNS = CONFIG["columns"]
TARGET = CONFIG["target"]

LEVELS = ["basso", "moderato", "alto", "molto alto"]
GFR_CATEGORIES = ["G5", "G4", "G3b", "G3a", "G2", "G1"]
GFR_BOUNDS = [-np.inf, 15, 30, 45, 60, 90, np.inf]
ACR_CATEGORIES = ["A1", "A2", "A3"]
ACR_BOUNDS = [-np.inf, 30, 300, np.inf]

# celle non elencate: rischio molto alto
RISK_MAP = {
    ("G1", "A1"): "basso", ("G2", "A1"): "basso",
    ("G1", "A2"): "moderato", ("G2", "A2"): "moderato", ("G3a", "A1"): "moderato",
    ("G1", "A3"): "alto", ("G2", "A3"): "alto",
    ("G3a", "A2"): "alto", ("G3b", "A1"): "alto",
}

# µmol/l -> mg/dl
CREATININE_FACTOR = 88.4
# CKD-EPI 2021 (senza coefficiente etnico): femmine, maschi
KAPPA = {True: 0.7, False: 0.9}
ALPHA = {True: -0.241, False: -0.302}
SEX_FACTOR = {True: 1.012, False: 1.0}


def is_female(df):
    return df[COLUMNS["gender"]] == TARGET["gender_female_code"]


def egfr(df):
    """eGFR in ml/min/1.73 m2 secondo CKD-EPI 2021."""
    creatinine = df[COLUMNS["creatinine"]] / CREATININE_FACTOR
    female = is_female(df)
    ratio = creatinine / female.map(KAPPA)
    return (
        142
        * np.minimum(ratio, 1) ** female.map(ALPHA)
        * np.maximum(ratio, 1) ** -1.200
        * 0.9938 ** df[COLUMNS["age"]]
        * female.map(SEX_FACTOR)
    )


def gfr_category(df):
    return pd.cut(egfr(df), GFR_BOUNDS, labels=GFR_CATEGORIES, right=False)


def acr_category(df):
    return pd.cut(df[COLUMNS["acr"]], ACR_BOUNDS, labels=ACR_CATEGORIES, right=False)


def kdigo_level(df):
    """Livello di rischio della heatmap KDIGO (asse G x asse A)."""
    cells = zip(gfr_category(df).astype(str), acr_category(df).astype(str))
    levels = [RISK_MAP.get(cell, "molto alto") for cell in cells]
    return pd.Series(
        pd.Categorical(levels, categories=LEVELS, ordered=True), index=df.index
    )


def target(df):
    """1 = marcatori di malattia renale cronica (ACR >= 30 oppure eGFR < 60)."""
    high_acr = df[COLUMNS["acr"]] >= TARGET["acr_threshold"]
    low_egfr = egfr(df) < TARGET["egfr_threshold"]
    return (high_acr | low_egfr).astype(int)
