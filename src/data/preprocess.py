"""Selezione delle feature, codifiche senza stato e preprocessor da stimare sul training.

Le trasformazioni riga per riga (selezione, codifiche) non imparano nulla dai dati e si
possono applicare prima della cross-validation. Imputazione e scaling invece imparano dai
dati: stanno nel preprocessor e vanno stimati solo sul training di ogni fold, senza y.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG

FEATURES = CONFIG["features"]
NUMERIC = FEATURES["numeric"]
CATEGORICAL = FEATURES["categorical"]
ORDINAL = FEATURES["ordinal"]
CONSEQUENCE = FEATURES["consequence"]
FEMALE = CONFIG["target"]["gender_female_code"]

# colonne che non devono mai entrare nel modello, qualunque sia il set di feature
FORBIDDEN = (
    FEATURES["exclude"]["leakage"]
    + FEATURES["exclude"]["renal_exams"]
    + FEATURES["unknown_columns"]
)

FEATURE_SETS = {
    "main": NUMERIC + CATEGORICAL,
    # analisi di sensibilità senza le variabili alterate dalla malattia renale
    "no_consequence": [c for c in NUMERIC + CATEGORICAL if c not in CONSEQUENCE],
}


def encode(df):
    """Codifiche senza stato.

    - HypertenHis: solo 1 o vuoto nei dati -> vuoto = nessuna anamnesi di ipertensione
    - Gender: 1 = maschio, 2 = femmina -> 0 = maschio, 1 = femmina
    - Smoking, Drinking, Tea: 1 = no, 2 = occasionalmente, 3 = regolarmente -> 0, 1, 2
    """
    df = df.copy()
    df["HypertenHis"] = df["HypertenHis"].fillna(0)
    df["Gender"] = (df["Gender"] == FEMALE).astype(int)
    df[ORDINAL] = df[ORDINAL] - 1
    return df


def select_features(df, feature_set="main"):
    """Matrice delle feature X per il set richiesto (main oppure no_consequence)."""
    columns = FEATURE_SETS[feature_set]
    leaked = set(columns) & set(FORBIDDEN)
    assert not leaked, f"colonne vietate fra le feature: {sorted(leaked)}"
    return encode(df)[columns]


def feature_types(columns):
    """Colonne numeriche e categoriche presenti in X, nell'ordine della config."""
    numeric = [c for c in NUMERIC if c in columns]
    categorical = [c for c in CATEGORICAL if c in columns]
    return numeric, categorical


def build_preprocessor(numeric_imputer, columns):
    """Due passi, output con le numeriche prima e le categoriche dopo:
    1. numeriche standardizzate (lo scaler ignora i NaN); categoriche imputate con la moda,
       senza scaling né one-hot (binarie e ordinali restano interi)
    2. imputer scelto su tutte le colonne: le categoriche, ormai complete, non vengono
       toccate ma fanno da predittori per le numeriche (es. DM per stimare HbA1c)
    Nessun indicatore di mancanza: i NaN dipendono dalla giornata di raccolta.
    """
    numeric, categorical = feature_types(columns)
    prepare = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", SimpleImputer(strategy="most_frequent"), categorical),
        ],
        verbose_feature_names_out=False,
    )
    return make_pipeline(prepare, numeric_imputer)
