"""Fold della cross-validation annidata, condivisi da imputazione, Fase A e Fase B.

Esterni: stratificati sul livello KDIGO, gli stessi del confronto delle imputazioni. Producono
le previsioni sui soggetti non visti usate per l'analisi per livello KDIGO.
Interni: dentro il training di ogni fold esterno, per ottimizzare gli iperparametri.
"""
from sklearn.model_selection import StratifiedKFold

from src.config import CONFIG
from src.data.kidney import kdigo_level

SEED = CONFIG["seed"]
OUTER = CONFIG["imputation"]["cv_folds"]
INNER = CONFIG["cv"]["inner_folds"]


def stratified_folds(df, k):
    cv = StratifiedKFold(k, shuffle=True, random_state=SEED)
    return list(cv.split(df, kdigo_level(df)))


def outer_folds(df):
    """(train, valid) come posizioni di riga in df."""
    return stratified_folds(df, OUTER)


def inner_folds(df, outer_train):
    """Fold interni del training esterno, come posizioni di riga in df (non nel sottoinsieme)."""
    subset = df.iloc[outer_train]
    return [(outer_train[tr], outer_train[va]) for tr, va in stratified_folds(subset, INNER)]
