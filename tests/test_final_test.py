"""Test della conferma finale su dati sintetici: nessun test legge data/processed/test.csv.

I punti delicati: il test set non deve essere letto senza autorizzazione né due volte, e ogni
controllo che non lo richiede va fatto prima di aprirlo; soglie e fasce sono quelle fissate in
cross-validation, mai ristimate; il preprocessore deve riprodurre la cache del training; le
affermazioni non devono decidere su tabelle incomplete.
"""
import inspect
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.impute import SimpleImputer

import src.models.final_test as ft
import src.models.phase_c as pc
from src.config import CONFIG
from src.data.imputed import load_fold
from src.data.kidney import LEVELS, kdigo_level, target
from src.data.preprocess import build_preprocessor, select_features
from src.data.split import PROCESSED
from tests.test_clinical_utility import creatinine_for

COLUMNS = CONFIG["columns"]
MODELS = ["dummy", "lr_scored", "lr_penalized", "random_forest", "xgboost"]
ARMS = ["fase_a", "none", "class_weight", "undersampling"]
TECHNIQUES = ["none", "class_weight", "undersampling"]


@pytest.fixture(scope="module")
def synthetic():
    """600 soggetti con livelli KDIGO noti e diabetici arricchiti di casi gravi; previsioni per 4
    bracci e 5 modelli; soglie e fasce fissate come se venissero dalla cross-validation."""
    rng = np.random.default_rng(1)
    n = 600
    codes = rng.choice(4, size=n, p=[0.85, 0.09, 0.04, 0.02])
    scre = np.where(codes == 3, creatinine_for(50.0), creatinine_for(100.0))
    acr = np.array([10.0, 100.0, 400.0, 400.0])[codes]
    diabetic = rng.random(n) < np.where(codes >= 2, 0.5, 0.12)
    test = pd.DataFrame({COLUMNS["id"]: np.arange(10_000, 10_000 + n), COLUMNS["creatinine"]: scre,
                         COLUMNS["acr"]: acr, COLUMNS["age"]: 50, COLUMNS["gender"]: 1,
                         COLUMNS["diabetes"]: diabetic.astype(int)})
    assert list(pd.Categorical(kdigo_level(test), LEVELS, ordered=True).codes) == list(codes)
    y = target(test).to_numpy()
    frames, rows_a, rows_b = [], [], []
    for arm in ARMS:
        shift = {"fase_a": 0.0, "none": 0.0, "class_weight": 2.0, "undersampling": 2.2}[arm]
        for model in MODELS:
            if model == "dummy":
                p = np.full(n, 0.098 if arm in ("fase_a", "none") else 0.5)
            else:
                p = expit(1.3 * codes + rng.normal(0, 1.5, n) - 3.0 + shift)
            frames.append(pd.DataFrame({"row": np.arange(n), "fold": "test", "model": model, "technique": arm,
                                        "probability": p,
                                        "probability_calibrated": expit(np.log(p / (1 - p)) - shift)}))
            if model == "dummy":
                continue
            cut = float(np.quantile(p[y == 1], 0.10))
            bands = list(np.quantile(p, [0.85, 0.94, 0.98]))
            row = {"model": model, "threshold": cut, **dict(zip(pc.BAND_COLUMNS, bands))}
            if arm == "fase_a":
                rows_a.append({"feature_set": "main", **row})
            else:
                rows_b.append({"technique": arm, **row})
    oof_auc = pd.DataFrame({"feature_set": "main", "model": MODELS, "auc": [0.5, 0.80, 0.80, 0.80, 0.80]})
    return pd.concat(frames, ignore_index=True), test, pd.DataFrame(rows_a), pd.DataFrame(rows_b), oof_auc


@pytest.fixture(scope="module")
def tables(synthetic):
    predictions, test, cut_a, cut_b, oof_auc = synthetic
    return ft.evaluate(predictions, test, cut_a, cut_b, oof_auc, techniques=TECHNIQUES, n_boot=20)


def expected_counts(test):
    levels = pd.Series(np.asarray(kdigo_level(test)))
    return {"n": len(test), "positives": int(target(test).sum()), "alto": int((levels == "alto").sum()),
            "molto alto": int((levels == "molto alto").sum()),
            "diabetici": int(test[COLUMNS["diabetes"]].sum())}


def verdict(tables, oof_auc, claim):
    return ft.claims_table(tables, oof_auc, TECHNIQUES).set_index("claim").loc[claim, "verdict"]


# --- protezione del test set ---------------------------------------------------------------------

def test_test_csv_is_read_only_inside_run():
    source = inspect.getsource(ft)
    assert source.count("test.csv") == 1
    assert "test.csv" in inspect.getsource(ft.run)


def test_config_keeps_the_test_closed():
    """Resta false finché l'utente non autorizza; dopo l'esecuzione torna false (protocollo)."""
    assert CONFIG["final_test"]["authorized"] is False


def test_run_refuses_before_reading_anything_without_authorization(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("run() ha letto un file senza autorizzazione")
    monkeypatch.setattr(ft.pd, "read_csv", forbidden)
    with pytest.raises(RuntimeError, match="authorized"):
        ft.run(output=tmp_path)


def test_authorize_blocks_a_second_run_without_a_reason(tmp_path):
    run_file = tmp_path / "RUN.json"
    ft.authorize({"authorized": True}, run_file)  # prima esecuzione: consentita
    run_file.write_text(json.dumps({"runs": [{"reason": "prima esecuzione"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="già stato usato"):
        ft.authorize({"authorized": True}, run_file)
    ft.authorize({"authorized": True}, run_file, rerun_reason="errore di codice dichiarato")
    with pytest.raises(RuntimeError):
        ft.authorize({"authorized": False}, run_file, rerun_reason="qualunque")


def test_run_checks_before_opening_and_records_every_use(synthetic, tmp_path, monkeypatch):
    """run() con file finti in tmp_path: i controlli senza test vengono prima dell'apertura; se il
    calcolo fallisce dopo l'apertura, RUN.json registra comunque l'uso; una seconda esecuzione è
    rifiutata senza motivo e, con un motivo, finisce in una cartella separata."""
    predictions, test, cut_a, cut_b, oof_auc = synthetic
    processed = tmp_path / "processed"
    processed.mkdir()
    train = test.assign(**{COLUMNS["id"]: test[COLUMNS["id"]] - 10_000})
    train.to_csv(processed / "train.csv", index=False)
    test.to_csv(processed / "test.csv", index=False)  # finto: righe sintetiche, non il test vero
    output = tmp_path / "out"
    order = []

    def fake_preflight(train):
        order.append("preflight")
        assert not (output / "RUN.json").exists() or len(order) > 1
        return {"preprocessor": None, "order": None, "loaded": {}, "cutpoints_a": cut_a,
                "cutpoints_b": cut_b, "oof_auc": oof_auc}

    def broken_evaluate(*args, **kwargs):
        raise ValueError("guasto simulato dopo l'apertura")

    real_evaluate = ft.evaluate
    monkeypatch.setattr(ft, "SETTINGS", {"authorized": True})
    monkeypatch.setattr(ft, "PROCESSED", processed)
    monkeypatch.setattr(ft, "EXPECTED", expected_counts(test))
    monkeypatch.setattr(ft, "preflight", fake_preflight)
    monkeypatch.setattr(ft, "transform_test", lambda *args: pd.DataFrame())
    monkeypatch.setattr(ft, "predict_all", lambda X, loaded: predictions)
    monkeypatch.setattr(ft, "evaluate", broken_evaluate)
    with pytest.raises(ValueError):
        ft.run(output=output, git_check=False)
    first = json.loads((output / "RUN.json").read_text(encoding="utf-8"))["runs"]
    assert order == ["preflight"] and len(first) == 1
    assert first[0]["status"] == "aperto" and first[0]["counts"] == expected_counts(test)
    assert len(first[0]["sha256"]) == 64
    with pytest.raises(RuntimeError, match="già stato usato"):
        ft.run(output=output, git_check=False)
    monkeypatch.setattr(ft, "evaluate", lambda *a, **k: real_evaluate(*a, techniques=TECHNIQUES, n_boot=5))
    ft.run(rerun_reason="guasto simulato nel test", output=output, git_check=False)
    runs = json.loads((output / "RUN.json").read_text(encoding="utf-8"))["runs"]
    assert [r["status"] for r in runs] == ["aperto", "completato"]
    assert runs[1]["reason"] == "guasto simulato nel test"
    assert (output / "run_2" / "claims.csv").exists()
    saved = pd.read_csv(output / "run_2" / "test_predictions.csv")
    assert {COLUMNS["id"], "target", "level", COLUMNS["diabetes"]} <= set(saved.columns)


def test_check_test_frame(synthetic):
    _, test, *_ = synthetic
    expected = expected_counts(test)
    train = test.assign(**{COLUMNS["id"]: test[COLUMNS["id"]] - 10_000})
    assert ft.check_test_frame(train, test, expected) == expected
    with pytest.raises(RuntimeError, match="identificativi"):
        ft.check_test_frame(test.head(5), test, expected)
    with pytest.raises(RuntimeError, match="numerosità"):
        ft.check_test_frame(train, test, {**expected, "positives": expected["positives"] + 1})


# --- preprocessore e modelli -----------------------------------------------------------------------

def test_transform_uses_the_training_fit_and_checks_the_cache():
    """"test" finto: righe del training, mai test.csv. La trasformazione è quella del
    preprocessore stimato sul training, e una cache diversa ferma tutto."""
    train = pd.read_csv(PROCESSED / "train.csv")
    fake_test = train.sample(40, random_state=0).reset_index(drop=True)
    X_train = select_features(train, "main")
    reference_pipe = build_preprocessor(SimpleImputer(strategy="median"), X_train.columns).fit(X_train)
    preprocessor, order = ft.fit_preprocessor(train, imputer=SimpleImputer(strategy="median"))
    out = ft.transform_test(preprocessor, order, fake_test)
    assert np.array_equal(out.to_numpy(), reference_pipe.transform(select_features(fake_test, "main")))
    reference = pd.DataFrame(reference_pipe.transform(X_train), columns=order)
    ft.fit_preprocessor(train, imputer=SimpleImputer(strategy="median"), reference=reference)
    broken = reference.copy()
    broken.iloc[0, 0] += 1e-9
    with pytest.raises(RuntimeError, match="non riproduce"):
        ft.fit_preprocessor(train, imputer=SimpleImputer(strategy="median"), reference=broken)


def test_refitted_preprocessor_reproduces_the_cache_bit_for_bit():
    """Test di riproducibilità permanente annunciato il 20/09/2026: il preprocessore vero
    (MissForest) ristimato sull'intero training riproduce la cache. Circa 1-2 minuti."""
    train = pd.read_csv(PROCESSED / "train.csv")
    reference, _ = load_fold("main")
    ft.fit_preprocessor(train, reference=reference)


def test_model_paths():
    assert ft.model_path("fase_a", "xgboost").name == "phase_a_xgboost_main.joblib"
    assert "depth_1_12" in str(ft.model_path("none", "xgboost"))
    assert ft.model_path("none", "random_forest") == ft.model_path("fase_a", "random_forest")
    path = ft.model_path("undersampling", "lr_scored")
    assert path.parent.name == "undersampling" and path.name == "phase_b_undersampling_lr_scored_main.joblib"


def test_all_final_models_and_platt_parameters_exist():
    """Controllo prima dell'esecuzione: ogni modello e ogni parametro di Platt necessario c'è."""
    if not ft.MODELS_DIR.exists():
        pytest.skip("modelli finali non presenti (non versionati)")
    for arm in [ft.PHASE_A_ARM, *ft.TECHNIQUES]:
        for name in ft.MODELS:
            assert ft.model_path(arm, name).exists(), (arm, name)
            params = ft.platt_params(arm, name)
            assert (params is None) == (arm == ft.PHASE_A_ARM or arm in ft.EXPLORATORY)


def test_load_models_forces_a_single_job():
    class Stub:
        def __init__(self):
            self.params = {"n_jobs": -1}

        def get_params(self):
            return self.params

        def set_params(self, **kwargs):
            self.params.update(kwargs)

    loaded = ft.load_models(["none"], ["random_forest"], loader=lambda path: Stub())
    assert all(model.params["n_jobs"] == 1 for model, _ in loaded.values())


def test_predict_one_refuses_wrong_columns():
    X = pd.DataFrame(np.zeros((3, 4)), columns=["Age", "Gender", "HGB", "Bpsys"])

    class Sklearn:
        n_features_in_ = 7

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    class Booster:
        feature_names = ["a", "b", "c", "d"]

    class Xgb(Sklearn):
        def get_booster(self):
            return Booster()

    with pytest.raises(RuntimeError, match="colonne"):
        ft.predict_one(Sklearn(), "lr_penalized", X)
    with pytest.raises(RuntimeError, match="colonne"):
        ft.predict_one(Xgb(), "xgboost", X)


def test_cutpoints_are_read_not_estimated_and_inputs_are_checked(synthetic):
    _, _, cut_a, cut_b, oof_auc = synthetic
    threshold, bands = ft.cutpoints_for("class_weight", "xgboost", cut_a, cut_b)
    row = cut_b[(cut_b["technique"] == "class_weight") & (cut_b["model"] == "xgboost")].iloc[0]
    assert threshold == row["threshold"] and list(bands) == list(row[pc.BAND_COLUMNS])
    assert ft.cutpoints_for("fase_a", "dummy", cut_a, cut_b) is None
    with pytest.raises(RuntimeError, match="tecniche principali"):
        ft.check_inputs(cut_a, cut_b, oof_auc, techniques=TECHNIQUES)


# --- valutazione -----------------------------------------------------------------------------------

def test_operating_point_uses_the_given_threshold(synthetic):
    """Soglia assurda di proposito: se venisse ristimata sul test non sarebbe rispettata."""
    predictions, test, cut_a, cut_b, oof_auc = synthetic
    cut_b = cut_b.copy()
    mask = (cut_b["technique"] == "none") & (cut_b["model"] == "lr_scored")
    cut_b.loc[mask, "threshold"] = 0.8
    out = ft.evaluate(predictions, test, cut_a, cut_b, oof_auc, techniques=TECHNIQUES, n_boot=5)
    row = out["q1_operating_points"].query("technique == 'none' and model == 'lr_scored' and group == 'tutti'")
    p = predictions.query("technique == 'none' and model == 'lr_scored'").sort_values("row")["probability"]
    assert row["alert_rate"].iloc[0] == pytest.approx(np.mean(p >= 0.8))


def test_tables_are_complete_and_coherent(tables):
    k_arms, k_real, k_groups = len(ARMS), 4, 3
    assert len(tables["q1_discrimination"]) == k_arms * len(MODELS) * k_groups
    assert len(tables["q1_operating_points"]) == k_arms * k_real * k_groups
    assert len(tables["primary_comparison"]) == k_real * 2  # 2 tecniche principali nel sintetico
    assert set(tables["calibration"]["probabilities"]) == {"grezza", "ricalibrata"}
    assert len(tables["decision_curve"]) == len(MODELS) * k_groups * 37
    claims = tables["claims"].set_index("claim")
    assert list(claims.index) == ["A", "B", "C", "D", "E"]
    assert set(claims["verdict"]) <= {"confermata", "non confermata", "contraddetta", "non contraddetta"}


def test_claim_a_detects_an_incompatible_auc(synthetic, tables):
    _, _, _, _, oof_auc = synthetic
    assert verdict(tables, oof_auc.assign(auc=0.99), "A") == "contraddetta"
    test_auc = tables["q1_discrimination"].query("technique == 'fase_a' and group == 'tutti'").set_index("model")["auc"]
    assert verdict(tables, oof_auc.assign(auc=oof_auc["model"].map(test_auc)), "A") == "non contraddetta"


def test_claim_b_branches(synthetic, tables):
    """Contraddetta solo con p di Holm (non il p grezzo) < 0,05 e a favore della tecnica."""
    _, _, _, _, oof_auc = synthetic
    base = tables["primary_comparison"].assign(p_value=0.5, p_holm=1.0, gained=0, lost=0)
    cases = [({"p_holm": 0.01, "gained": 5, "lost": 0}, "contraddetta"),
             ({"p_holm": 0.01, "gained": 0, "lost": 5}, "non contraddetta"),
             ({"p_value": 0.01, "p_holm": 0.08, "gained": 5, "lost": 0}, "non contraddetta")]
    for change, expected in cases:
        primary = base.copy()
        for column, value in change.items():
            primary.loc[primary.index[0], column] = value
        assert verdict({**tables, "primary_comparison": primary}, oof_auc, "B") == expected


def test_claim_c_needs_three_models_out_of_four(synthetic, tables):
    _, _, _, _, oof_auc = synthetic
    points = tables["q1_operating_points"]
    target_rows = points.query("technique == 'none' and group == 'diabetici' and model != 'dummy'").index
    for hits, expected in [(3, "confermata"), (2, "non confermata")]:
        changed = points.copy()
        changed.loc[target_rows, "alert_rate"] = [0.95] * hits + [0.5] * (4 - hits)
        assert verdict({**tables, "q1_operating_points": changed}, oof_auc, "C") == expected


def test_claim_d_needs_both_thresholds(synthetic, tables):
    _, _, _, _, oof_auc = synthetic
    curve = tables["decision_curve"].copy()
    rows = curve.query("group == 'tutti' and model != 'dummy' and threshold in [0.07, 0.1]").index
    curve.loc[rows, "beats_both"] = True
    assert verdict({**tables, "decision_curve": curve}, oof_auc, "D") == "confermata"
    failing = curve.query("group == 'tutti' and threshold == 0.1 and model in ['lr_scored', 'xgboost']").index
    curve.loc[failing, "beats_both"] = False
    assert verdict({**tables, "decision_curve": curve}, oof_auc, "D") == "non confermata"


def test_claim_e_needs_every_technique(synthetic, tables):
    """Confermata solo se TUTTE le tecniche superano "nessuna correzione": fallirebbe con any()."""
    _, _, _, _, oof_auc = synthetic
    assert verdict(tables, oof_auc, "E") == "confermata"
    naive = tables["naive"].copy()
    naive.loc[naive["technique"] == "class_weight", "recall"] = 0.0
    assert verdict({**tables, "naive": naive}, oof_auc, "E") == "non confermata"


def test_claims_refuse_incomplete_tables(synthetic, tables):
    """Tabelle con meno modelli o soglie del previsto: errore, non un esito deciso in silenzio."""
    _, _, _, _, oof_auc = synthetic
    incomplete = {
        "decision_curve": tables["decision_curve"].query("not (threshold == 0.1 and model == 'xgboost')"),
        "naive": tables["naive"].query("not (technique == 'none' and model == 'lr_scored')"),
        "q1_operating_points": tables["q1_operating_points"].query(
            "not (technique == 'none' and group == 'diabetici' and model == 'random_forest')"),
        "primary_comparison": tables["primary_comparison"].query(
            "not (model == 'xgboost' and technique == 'undersampling')"),
    }
    for name, table in incomplete.items():
        with pytest.raises(RuntimeError):
            ft.claims_table({**tables, name: table}, oof_auc, TECHNIQUES)


def test_severe_comparison_counts_by_hand(synthetic, tables):
    predictions, test, _, cut_b, _ = synthetic
    codes = np.asarray(pd.Categorical(kdigo_level(test), LEVELS, ordered=True).codes)
    severe = np.isin(codes, pc.SEVERE)
    row = tables["primary_comparison"].query("model == 'random_forest' and technique == 'undersampling'").iloc[0]
    p = (predictions.query("technique == 'undersampling' and model == 'random_forest'")
         .sort_values("row")["probability"].to_numpy())
    cut = cut_b.query("technique == 'undersampling' and model == 'random_forest'")["threshold"].iloc[0]
    assert row["severe_n"] == severe.sum()
    assert row["detected"] == int(np.sum(p[severe] >= cut))


def test_missing_calibrated_probabilities_are_an_error(synthetic):
    predictions, test, *_ = synthetic
    broken = predictions.copy()
    broken.loc[(broken["technique"] == "class_weight") & (broken["model"] == "xgboost"),
               "probability_calibrated"] = np.nan
    with pytest.raises(RuntimeError, match="valori mancanti"):
        ft.calibration_table(broken, test, TECHNIQUES, n_boot=5)
