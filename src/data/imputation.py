"""Confronto dei metodi di imputazione delle variabili numeriche (solo training).

Due criteri, in cross-validation stratificata sul livello KDIGO:
  (a) errore di ricostruzione: si nasconde una quota di valori osservati nel fold di
      validazione e si misura quanto l'imputer li ricostruisce (RMSE e MAE in unità
      standardizzate), solo sulle colonne che hanno valori mancanti nel training
  (b) prestazioni a valle: regressione logistica fissa, usata come strumento di misura,
      con PR-AUC e AUC
Ogni imputer usa come predittori anche le categoriche (già imputate con la moda).
Regola di scelta in due fasi (two_stage_choice): metodi con PR-AUC entro 1 errore standard
dal migliore, poi fra questi il più semplice con RMSE entro 1 errore standard dal migliore.
Il test set non viene mai letto.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from src.config import CONFIG, resolve
from src.data.kidney import kdigo_level, target
from src.data.preprocess import build_preprocessor, feature_types, select_features
from src.data.split import PROCESSED

SETTINGS = CONFIG["imputation"]
SEED = CONFIG["seed"]
OUTPUT = resolve(SETTINGS["output"])
# soglia di allarme: con sole feature non renali un'AUC così alta indica leakage
LEAKAGE_AUC = 0.9


def imputers():
    """Candidati in ordine di semplicità crescente (usato dalla regola di 1 errore standard)."""
    # max_features="sqrt" come il MissForest originale (mtry = radice del numero di colonne);
    # n_jobs=1: su foreste così piccole il parallelismo costa più di quanto rende
    forest = ExtraTreesRegressor(
        n_estimators=SETTINGS["forest_trees"],
        min_samples_leaf=SETTINGS["forest_min_samples_leaf"],
        max_features="sqrt",
        random_state=SEED,
    )
    return {
        "mediana": SimpleImputer(strategy="median"),
        "KNN": KNNImputer(n_neighbors=SETTINGS["knn_neighbors"]),
        "MICE": IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=SETTINGS["mice_max_iter"],
            skip_complete=True,
            random_state=SEED,
        ),
        # "MissForest con freni": pochi alberi, poche iterazioni,
        # le colonne complete fanno solo da predittori
        "MissForest": IterativeImputer(
            estimator=forest,
            max_iter=SETTINGS["forest_max_iter"],
            skip_complete=True,
            random_state=SEED,
        ),
    }


def folds(df):
    cv = StratifiedKFold(SETTINGS["cv_folds"], shuffle=True, random_state=SEED)
    return list(cv.split(df, kdigo_level(df)))


def reconstruction_error(preprocessor, train, valid, rng):
    """(a) RMSE e MAE sui valori nascosti delle numeriche, in unità standardizzate.

    preprocessor: già stimato sul fold di training; nel suo output le numeriche vengono prima.
    """
    numeric, _ = feature_types(train.columns)
    truth = preprocessor[0].named_transformers_["numeric"].transform(valid[numeric])
    incomplete = train[numeric].isna().any().to_numpy()
    hidden = ~np.isnan(truth) & incomplete & (rng.random(truth.shape) < SETTINGS["mask_fraction"])
    masked = valid.copy()
    masked[numeric] = valid[numeric].mask(hidden)
    imputed = preprocessor.transform(masked)[:, :len(numeric)]
    diff = imputed[hidden] - truth[hidden]
    return np.sqrt(np.mean(diff ** 2)), np.mean(np.abs(diff))


def evaluate_fold(imputer, X, y, tr, va, rng):
    """Un solo fit per fold: lo stesso imputer serve a entrambi i criteri."""
    model = make_pipeline(
        build_preprocessor(clone(imputer), X.columns),
        LogisticRegression(class_weight="balanced", max_iter=5000),
    )
    model.fit(X.iloc[tr], y.iloc[tr])
    p = model.predict_proba(X.iloc[va])[:, 1]
    rmse, mae = reconstruction_error(model[0], X.iloc[tr], X.iloc[va], rng)
    return {"RMSE": rmse, "MAE": mae,
            "PR-AUC": average_precision_score(y.iloc[va], p),
            "AUC": roc_auc_score(y.iloc[va], p)}


def compare(df, feature_set="main"):
    X, y = select_features(df, feature_set), target(df)
    rows = []
    for name, imputer in imputers().items():
        rng = np.random.default_rng(SEED)
        start = time.perf_counter()
        for k, (tr, va) in enumerate(folds(df)):
            scores = evaluate_fold(imputer, X, y, tr, va, rng)
            rows.append({"metodo": name, "fold": k, **scores})
        elapsed = time.perf_counter() - start
        for row in rows[-SETTINGS["cv_folds"]:]:
            row["secondi"] = elapsed
    return pd.DataFrame(rows)


def summarize(scores):
    """Media ed errore standard per metodo, nell'ordine di semplicità."""
    order = list(imputers())
    grouped = scores.groupby("metodo", sort=False)
    mean = grouped.mean().drop(columns="fold")
    se = grouped.sem().drop(columns=["fold", "secondi"]).add_suffix(" SE")
    table = mean.join(se).loc[order]
    return table[["RMSE", "RMSE SE", "MAE", "MAE SE", "PR-AUC", "PR-AUC SE",
                  "AUC", "AUC SE", "secondi"]]


def one_se_choice(table, metric="PR-AUC"):
    """Il metodo più semplice con metrica entro 1 errore standard dal migliore."""
    best = table[metric].idxmax()
    threshold = table.loc[best, metric] - table.loc[best, f"{metric} SE"]
    return next(name for name in table.index if table.loc[name, metric] >= threshold)


def two_stage_choice(table):
    """Regola in due fasi (introdotta dopo aver visto i risultati, vedi Notepad):
    1. metodi con PR-AUC entro 1 errore standard dal migliore (prestazioni a valle)
    2. fra questi, il più semplice con RMSE entro 1 errore standard dal migliore
       (ricostruzione dei valori, rilevante per SMOTE e CTGAN)
    """
    best = table["PR-AUC"].idxmax()
    floor = table.loc[best, "PR-AUC"] - table.loc[best, "PR-AUC SE"]
    candidates = table[table["PR-AUC"] >= floor]
    best = candidates["RMSE"].idxmin()
    ceiling = candidates.loc[best, "RMSE"] + candidates.loc[best, "RMSE SE"]
    return next(name for name in candidates.index if candidates.loc[name, "RMSE"] <= ceiling)


def chosen_imputer():
    """Imputer numerico fissato in config, da usare in tutte le fasi successive."""
    return imputers()[SETTINGS["method"]]


if __name__ == "__main__":
    train = pd.read_csv(PROCESSED / "train.csv")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for feature_set in ["main", "no_consequence"]:
        scores = compare(train, feature_set)
        table = summarize(scores)
        scores.to_csv(OUTPUT / f"imputation_folds_{feature_set}.csv", index=False)
        table.round(4).to_csv(OUTPUT / f"imputation_comparison_{feature_set}.csv")
        print(f"== set di feature: {feature_set}")
        print(table.round(4).to_string())
        print("regola a una fase (PR-AUC):", one_se_choice(table))
        print("regola a due fasi (PR-AUC, poi RMSE):", two_stage_choice(table))
        if table["AUC"].max() > LEAKAGE_AUC:
            print(f"ATTENZIONE: AUC > {LEAKAGE_AUC}, possibile leakage residuo")
        print()
