import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.data.folds import OUTER
from src.data.imputed import build_cache, fold_name, load_fold
from src.data.kidney import target
from src.data.split import PROCESSED
from src.models.phase_d import (CANDIDATES, DIAGNOSTICS, INTERPRETATION, REFERENCES, SETTINGS,
                                acr_label_sensitivity, acr_labels, components, consensus_top, evaluate,
                                full_training_rankings, label_noise_auroc, loader, noise_for_auroc, oof,
                                reference_record, same_day_auc, save_record, semi_synthetic_control,
                                stratified_subsample)

METHOD = "mediana"


@pytest.fixture(scope="module")
def train():
    return pd.read_csv(PROCESSED / "train.csv")


@pytest.fixture(scope="module")
def cache(train, tmp_path_factory):
    """Fold imputati con la mediana (veloce) in una cartella temporanea, solo set main."""
    directory = tmp_path_factory.mktemp("imputed")
    build_cache(train, "main", SimpleImputer(strategy="median"), directory, METHOD)
    return directory


def test_rule_fixed_in_config():
    assert SETTINGS["min_difference"] == 0.01
    assert SETTINGS["target_sensitivity"] == 0.90
    assert REFERENCES == ["lr_penalized", "random_forest", "xgboost"]
    assert {"ensemble_mean", "catboost", "lightgbm", "ebm", "tabpfn"} <= set(CANDIDATES)


def test_tri_ensemble_registered_with_nested_selection():
    spec = CANDIDATES["tri_ensemble_top21"]
    assert spec["kind"] == "tri_ensemble" and spec["top"] == 21 and spec["selection"] == "nested"
    assert DIAGNOSTICS["tri_ensemble_leaky"]["selection"] == "full_training"
    assert "tri_ensemble_leaky" not in CANDIDATES  # la versione con selezione sull'intero training non è un candidato


def test_consensus_top_by_hand():
    """Ranghi medi: a = 1/3, b = 1, c = 2, d = 8/3. A parità vale l'ordine della prima classifica."""
    rankings = [["a", "b", "c", "d"], ["b", "a", "d", "c"], ["a", "c", "b", "d"]]
    assert consensus_top(rankings, 2) == ["a", "b"]
    assert consensus_top(rankings, 3) == ["a", "b", "c"]
    assert consensus_top([["x", "y"], ["y", "x"]], 2) == ["x", "y"]
    with pytest.raises(ValueError):
        consensus_top([["a", "b"], ["a", "c"]], 1)


def test_full_training_rankings_cover_all_features():
    lists = full_training_rankings(REFERENCES)
    assert len(lists) == 3 and all(sorted(r) == sorted(lists[0]) for r in lists)
    odds = pd.read_csv(INTERPRETATION / "odds_ratios.csv").query("feature_set == 'main' and model == 'lr_penalized'")
    assert lists[0][0] == odds.loc[odds["coefficient"].abs().idxmax(), "feature"]


def test_nested_tri_ensemble_selects_on_training_rows_only(train, cache, tmp_path, monkeypatch):
    """La classifica di ogni fold deve vedere solo le righe di training: fallirebbe se la selezione
    usasse anche le righe del fold esterno, che è l'errore che il candidato serve a escludere."""
    import src.models.phase_d as phase_d
    seen = []

    def fake_rankings(X_tr, y_tr, outer, members, shap_rows):
        seen.append(set(X_tr.index))
        columns = list(X_tr.columns)
        return [columns, columns[::-1], columns]

    class Constant:
        def fit(self, X, y):
            self.p = float(np.mean(y))
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])

    monkeypatch.setattr(phase_d, "fold_rankings", fake_rankings)
    monkeypatch.setattr(phase_d, "fit", lambda name, params, X, y: Constant().fit(X, y))
    monkeypatch.setattr(phase_d, "loader", lambda df, y, data: loader(df, y, data, directory=cache, method=METHOD))
    spec = {"kind": "tri_ensemble", "data": "imputed", "top": 21, "selection": "nested", "shap_rows": 500}
    phase_d.run_tri_ensemble("tri", spec, train, output=tmp_path)
    assert len(seen) == OUTER
    for outer, training_rows in enumerate(seen):
        record = json.loads((tmp_path / "tri" / f"{fold_name(outer)}.json").read_text(encoding="utf-8"))
        assert training_rows.isdisjoint(record["rows"])
        assert len(record["features"]) == 21 and len(set(record["features"])) == 21


def test_components_make_the_target(train):
    albuminuria, low_egfr = components(train)
    assert np.array_equal(albuminuria | low_egfr, target(train).to_numpy())


def test_stratified_subsample_keeps_both_classes():
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    keep = stratified_subsample(y, 0.4, np.random.default_rng(0))
    assert len(keep) == 400 and y[keep].sum() == 40 and len(np.unique(keep)) == 400


def test_raw_loader_uses_the_same_rows(train, cache):
    y = target(train).to_numpy()
    imputed = loader(train, y, "imputed", directory=cache, method=METHOD)(0, 1)
    raw = loader(train, y, "raw", directory=cache, method=METHOD)(0, 1)
    for a, b in [(imputed[0], raw[0]), (imputed[2], raw[2])]:
        assert a.index.equals(b.index) and list(a.columns) == list(b.columns)
    assert np.array_equal(imputed[1], raw[1]) and np.array_equal(imputed[3], raw[3])
    assert raw[0].isna().any().any() and not imputed[0].isna().any().any()


def test_scaled_loader_uses_training_statistics(train, cache):
    X_tr, _, X_va, _ = loader(train, target(train), "imputed_scaled", directory=cache, method=METHOD)(0, None)
    sd = X_tr.std(ddof=0)
    assert np.allclose(X_tr.mean(), 0, atol=1e-9)
    assert np.allclose(sd[sd > 0], 1)
    assert not np.allclose(X_va.mean(), 0, atol=1e-9)


def test_positive_control_adds_the_extra_column(train, cache):
    X_tr, _, X_va, _ = loader(train, target(train), "imputed", ["UmALB"], cache, METHOD)(2, None)
    assert np.array_equal(X_va["UmALB"].to_numpy(), train["UmALB"].to_numpy()[X_va.index.to_numpy()])
    X_plain, _ = load_fold("main", 2, None, cache, METHOD)
    assert list(X_tr.columns) == [*X_plain.columns, "UmALB"]


def test_candidate_equal_to_reference_does_not_improve(train, tmp_path):
    """Un candidato identico al miglior riferimento ha differenza 0 e non supera la regola."""
    for outer in range(OUTER):
        save_record("ensemble_mean", outer, reference_record("random_forest", outer), tmp_path)
    metrics, comparison = evaluate(train, ["ensemble_mean"], output=tmp_path)
    row = comparison.iloc[0]
    assert row["reference"] == "random_forest"
    assert row["difference"] == pytest.approx(0) and not row["improves"]
    assert json.loads((tmp_path / "ensemble_mean" / "outer0.json").read_text())["rows"]
    assert set(metrics["role"]) == {"riferimento", "candidato"}


# --- audit del tetto dei dati (23/09/2026) ---------------------------------------------------

def test_audit_diagnostics_registered_outside_candidates():
    for name, kind in [("semi_synthetic_control", "semi_synthetic"), ("label_noise_auroc", "label_noise"),
                       ("acr_label_sensitivity", "label_sensitivity")]:
        assert DIAGNOSTICS[name]["kind"] == kind and name not in CANDIDATES
    assert DIAGNOSTICS["semi_synthetic_control"]["target_auroc"] == [0.75, 0.80]
    assert DIAGNOSTICS["acr_label_sensitivity"]["factor"] == {"observed": 176.8, "alternative": 100}


def test_noise_for_auroc_hits_the_target():
    """Più rumore, AUROC più bassa: la DS trovata deve dare l'AUROC chiesta, e un'AUROC più alta
    deve richiedere meno rumore."""
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    signal = y * 3.0 + rng.standard_normal(1000) * 0.3
    noise = rng.standard_normal(1000)
    sd_75 = noise_for_auroc(signal, y, noise, 0.75)
    sd_80 = noise_for_auroc(signal, y, noise, 0.80)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, signal + sd_75 * noise) == pytest.approx(0.75, abs=0.005)
    assert roc_auc_score(y, signal + sd_80 * noise) == pytest.approx(0.80, abs=0.005)
    assert sd_80 < sd_75


def test_same_day_auc_by_hand():
    """Giornata A: positivo 0,9 e negativo 0,8; giornata B: positivo 0,2 e negativo 0,1; giornata C
    solo negativi (ignorata). Coppie della stessa giornata: 2 concordanti su 2 -> 1,0. AUROC globale:
    il positivo 0,2 è sotto i negativi 0,8 e 0,5 -> 4/6."""
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.5])
    day = np.array(["A", "A", "B", "B", "C"])
    assert same_day_auc(y, p, day) == pytest.approx(1.0)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p) == pytest.approx(4 / 6)
    # nessuna giornata con entrambe le classi: non calcolabile
    assert np.isnan(same_day_auc(np.array([1, 0]), np.array([0.3, 0.2]), np.array(["A", "B"])))


def test_acr_labels_by_hand():
    """eGFR < 60 resta positivo anche sotto la soglia e non viene mai escluso dalla zona grigia."""
    acr = np.array([10.0, 20.0, 30.0, 40.0, 25.0])
    low_egfr = np.array([0, 0, 0, 0, 1])
    y, keep = acr_labels(acr, low_egfr, 30)
    assert y.tolist() == [0, 0, 1, 1, 1] and keep.all()
    y, keep = acr_labels(acr, low_egfr, 30, grey=(17.7, 35.4))
    assert keep.tolist() == [True, False, False, True, True]
    y, _ = acr_labels(acr, low_egfr, 53.04)
    assert y.tolist() == [0, 0, 0, 0, 1]


def test_acr_label_sensitivity_matches_the_target_at_30(train, tmp_path):
    """Sulla scala osservata, soglia 30 e nessuna zona grigia: stessa etichetta del progetto, quindi
    stessi positivi e stessa AUROC delle previsioni out-of-fold del riferimento."""
    spec = {**DIAGNOSTICS["acr_label_sensitivity"], "models": ["random_forest"]}
    table = acr_label_sensitivity(train, spec, output=tmp_path)
    y = target(train).to_numpy()
    row = table.query("scale == 'osservata' and threshold == 30 and grey_zone == ''").iloc[0]
    p, _ = oof("random_forest", len(y))
    from sklearn.metrics import roc_auc_score
    assert row["positives"] == y.sum() and row["n"] == len(y)
    assert row["auc"] == pytest.approx(roc_auc_score(y, p))
    # sulla scala alternativa la soglia 30 vera equivale a UMAUCR >= 30 x 176,8 / 100 = 53,04
    alt = table.query("scale == 'alternativa' and grey_zone == ''").iloc[0]
    same = table.query("scale == 'osservata' and threshold == 53.04 and grey_zone == ''").iloc[0]
    assert alt["positives"] == same["positives"] and alt["auc"] == pytest.approx(same["auc"])
    assert (tmp_path / "acr_label_sensitivity.csv").exists()


def test_label_noise_auroc_rows(train, tmp_path):
    spec = {**DIAGNOSTICS["label_noise_auroc"], "models": ["random_forest"]}
    table = label_noise_auroc(train, spec, output=tmp_path)
    y = target(train).to_numpy()
    overall = table.query("analysis == 'tutti'").iloc[0]
    assert overall["positives"] == y.sum()
    bands = table[table["analysis"].str.startswith("fascia ACR")]
    assert len(bands) == 4 and (bands["n"] - bands["positives"] == (y == 0).sum()).all()
    assert table["analysis"].str.startswith("coppie della stessa giornata").any()
    assert (tmp_path / "label_noise_auroc.csv").exists()


def test_semi_synthetic_control_wiring(train, cache, tmp_path, monkeypatch):
    """Con un "modello" che usa solo la feature semi-sintetica, l'AUROC out-of-fold deve coincidere
    con quella univariata: verifica che la colonna arrivi con i valori giusti sulle righe giuste."""
    import src.models.phase_d as phase_d

    class OnlyZ:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            # z così com'è (per l'AUROC conta solo l'ordine): una sigmoide creerebbe pareggi a 1,0
            z = X[phase_d.SEMI_SYNTHETIC_COLUMN].to_numpy()
            return np.column_stack([-z, z])

    monkeypatch.setattr(phase_d, "fit", lambda name, params, X, y: OnlyZ().fit(X, y))
    monkeypatch.setattr(phase_d, "loader", lambda df, y, data, extra: loader(df, y, data, extra, cache, METHOD))
    table = semi_synthetic_control(train, output=tmp_path)
    assert len(table) == 2
    for _, row in table.iterrows():
        assert row["z_auc"] == pytest.approx(row["target_auroc"], abs=0.005)
        assert row["oof_auc"] == pytest.approx(row["z_auc"], abs=1e-9) and row["passes"]
    assert (tmp_path / "semi_synthetic_control.csv").exists()
