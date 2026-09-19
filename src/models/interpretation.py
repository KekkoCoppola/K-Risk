"""Interpretazione dei modelli finali della Fase A.

- logistiche: odds ratio per 1 deviazione standard (numeriche, standardizzate nel preprocessing) o
  per unità (categoriche); la logistica SCORED, non penalizzata, con intervallo di Wald al 95%
- Random Forest e XGBoost: valori SHAP esatti (TreeSHAP) su tutti i soggetti del training
Scelte fissate prima del calcolo: Notepad, "Interpretazione: scelte fissate prima del calcolo".
Legge i modelli finali e il training imputato (fold "full"): il test set non viene letto.
Descrive che cosa ha imparato il modello, non relazioni causali.
"""
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy import stats
from xgboost import XGBClassifier

from src.config import CONFIG, resolve
from src.data.imputed import load_fold
from src.data.preprocess import FEATURE_SETS, feature_types
from src.models.zoo import columns

SETTINGS = CONFIG["phase_a"]
OUTPUT = resolve(SETTINGS["output"]) / "interpretation"
MODELS_DIR = resolve(SETTINGS["models_dir"])
ALPHA = CONFIG["evaluation"]["alpha"]
LOGISTIC = ["lr_scored", "lr_penalized"]
TREES = ["random_forest", "xgboost"]
TOP = 15


def final_model(name, feature_set, models_dir=MODELS_DIR):
    return joblib.load(models_dir / f"phase_a_{name}_{feature_set}.joblib")


def training_data(name, feature_set):
    """Training intero imputato (fold "full") con le sole colonne del modello, nello stesso ordine
    usato per addestrare il modello finale."""
    X, _ = load_fold(feature_set)
    return X[columns(name, X.columns)]


def wald_se(model, X):
    """Errori standard dei coefficienti di una logistica non penalizzata: radice della diagonale di
    (X' W X)^-1, con W = p (1 - p) e l'intercetta nel disegno (Hosmer, Lemeshow & Sturdivant 2013)."""
    p = model.predict_proba(X)[:, 1]
    design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    information = design.T @ (design * (p * (1 - p))[:, None])
    return np.sqrt(np.diag(np.linalg.inv(information)))[1:]


def odds_ratios(model, X, name, feature_set, alpha=ALPHA):
    """Odds ratio per variabile; intervallo di Wald solo per il modello non penalizzato: con la
    penalizzazione gli errori standard usuali non valgono."""
    numeric, _ = feature_types(X.columns)
    beta = model.coef_[0]
    table = pd.DataFrame({"feature_set": feature_set, "model": name, "feature": X.columns,
                          "per": ["1 DS" if c in numeric else "unità" for c in X.columns],
                          "coefficient": beta, "odds_ratio": np.exp(beta)})
    if name == "lr_scored":
        se = wald_se(model, X)
        z = stats.norm.ppf(1 - alpha / 2)
        table["low"], table["high"] = np.exp(beta - z * se), np.exp(beta + z * se)
        table["p_value"] = 2 * stats.norm.sf(np.abs(beta / se))
    return table.iloc[np.argsort(-np.abs(beta), kind="stable")].reset_index(drop=True)


def shap_values(model, X):
    """Valori SHAP esatti (TreeSHAP, Lundberg et al. 2020): Random Forest sulla scala della
    probabilità della classe 1, XGBoost sulla scala logit.

    Per XGBoost si usa il TreeSHAP interno di XGBoost (pred_contribs): contributi identici a quelli
    del pacchetto shap, che però con XGBoost 3 sbaglia il valore base di una costante (verificato,
    test in tests/test_interpretation.py)."""
    if isinstance(model, XGBClassifier):
        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    values = np.asarray(shap.TreeExplainer(model).shap_values(X))
    return values[:, :, 1] if values.ndim == 3 else values


def importance(values, X, name, feature_set):
    table = pd.DataFrame({"feature_set": feature_set, "model": name, "feature": X.columns,
                          "mean_abs_shap": np.abs(values).mean(axis=0)})
    return table.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)


def top_features(odds, shap_importance, feature_set="main", top=TOP):
    """Prime variabili di ogni modello affiancate: logistica penalizzata per valore assoluto del
    coefficiente, alberi per importanza SHAP media."""
    lists = {"lr_penalized": odds.query("feature_set == @feature_set and model == 'lr_penalized'")["feature"]}
    for name in TREES:
        lists[name] = shap_importance.query("feature_set == @feature_set and model == @name")["feature"]
    table = pd.DataFrame({name: values.head(top).to_list() for name, values in lists.items()})
    table.insert(0, "rank", range(1, top + 1))
    return table


def run(output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    odds, importances = [], []
    for feature_set in FEATURE_SETS:
        for name in LOGISTIC:
            X = training_data(name, feature_set)
            odds.append(odds_ratios(final_model(name, feature_set), X, name, feature_set))
        for name in TREES:
            X = training_data(name, feature_set)
            values = shap_values(final_model(name, feature_set), X)
            importances.append(importance(values, X, name, feature_set))
            # valori per i grafici (rigenerabili, non versionati)
            np.savez_compressed(output / f"shap_{name}_{feature_set}.npz", values=values.astype(np.float32),
                                X=X.to_numpy(dtype=np.float32), columns=np.array(X.columns, dtype=str))
            print(f"SHAP {name} {feature_set}", flush=True)
    odds, importances = pd.concat(odds, ignore_index=True), pd.concat(importances, ignore_index=True)
    tables = {"odds_ratios": odds, "shap_importance": importances,
              "top_features": top_features(odds, importances)}
    for table_name, table in tables.items():
        table.to_csv(output / f"{table_name}.csv", index=False)
        print(output / f"{table_name}.csv", flush=True)
    return tables


if __name__ == "__main__":
    run()
