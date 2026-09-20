"""Test della Fase C su dati sintetici: nessuna previsione reale viene letta.

Il punto delicato e' che soglia e fasce **non** vengano ristimate sul sottogruppo: i test usano
soglie deliberatamente assurde e verificano che vengano applicate cosi' come sono.
"""
import numpy as np
import pandas as pd
import pytest

import src.models.evaluation as ev
import src.models.phase_c as pc
from src.config import CONFIG
from src.data.kidney import LEVELS, egfr, kdigo_level, target

COLUMNS = CONFIG["columns"]
MODELS = ["dummy", "good", "weak"]
# soglie assurde di proposito: se il codice le ristimasse sul sottogruppo non le rispetterebbe
ABSURD = {"none": 0.80, "level_weight": 0.05}
ABSURD_BANDS = {"none": [0.82, 0.86, 0.90], "level_weight": [0.06, 0.07, 0.08]}


def creatinine_for(egfr_target, age=50, gender=1):
    """Creatinina (umol/l) che da' l'eGFR voluto con CKD-EPI 2021, per bisezione su kidney.egfr."""
    def value(scre):
        row = pd.DataFrame({COLUMNS["creatinine"]: [scre], COLUMNS["age"]: [age],
                            COLUMNS["gender"]: [gender]})
        return float(egfr(row).iloc[0])

    low, high = 1.0, 3000.0
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if value(middle) > egfr_target else (low, middle)
    return (low + high) / 2


@pytest.fixture(scope="module")
def synthetic():
    """800 soggetti su 4 livelli KDIGO, diabetici arricchiti di casi gravi come nei dati reali,
    2 tecniche x 3 modelli su 5 fold. Il classificatore di maggioranza prevede una costante."""
    rng = np.random.default_rng(0)
    n = 800
    codes = rng.choice(4, size=n, p=[0.80, 0.12, 0.05, 0.03])
    # eGFR e ACR scelti per cadere nella cella KDIGO voluta: G1/A1, G1/A2, G1/A3, G3a/A3
    scre_normal, scre_low = creatinine_for(100.0), creatinine_for(50.0)
    scre = np.where(codes == 3, scre_low, scre_normal)
    acr = np.array([10.0, 100.0, 400.0, 400.0])[codes]
    # diabetici: 15% dei livelli bassi, 45% dei gravi -> sottogruppo arricchito
    diabetic = rng.random(n) < np.where(codes >= 2, 0.45, 0.15)
    train = pd.DataFrame({COLUMNS["creatinine"]: scre, COLUMNS["acr"]: acr,
                          COLUMNS["age"]: 50, COLUMNS["gender"]: 1,
                          COLUMNS["diabetes"]: diabetic.astype(int)})
    assert list(pd.Categorical(kdigo_level(train), LEVELS, ordered=True).codes) == list(codes)

    y = target(train).to_numpy()
    folds = rng.permutation(np.arange(n) % 5)
    frames, cut_rows = [], []
    for technique in ("none", "level_weight"):
        for model in MODELS:
            if model == "dummy":
                # come nei dati reali del braccio level_weight: costante con un residuo di 1e-16
                p = np.full(n, float(y.mean())) + rng.choice([-1, 0, 1], n)[folds] * 3e-16
            else:
                noise = 1.2 if model == "good" else 3.0
                p = 1 / (1 + np.exp(-(codes + rng.normal(0, noise, n) - 2.0)))
            frames.append(pd.DataFrame({"row": np.arange(n), "fold": [f"outer{k}" for k in folds],
                                        "model": model, "feature_set": "main",
                                        "technique": technique, "probability": p}))
            if model != "dummy":  # come nei dati reali, il dummy non ha una riga di soglie
                cut_rows.append({"technique": technique, "model": model,
                                 "threshold": ABSURD[technique],
                                 **dict(zip(pc.BAND_COLUMNS, ABSURD_BANDS[technique]))})
    return pd.concat(frames, ignore_index=True), train, pd.DataFrame(cut_rows)


@pytest.fixture(scope="module")
def tables(synthetic):
    oof, train, cutpoints = synthetic
    return pc.evaluate(oof, train, cutpoints, n_boot=60)


# --- errore standard di DeLong ---------------------------------------------------------------

def test_delong_se_matches_evaluation():
    """Stessa formula di evaluation.delong: meta' ampiezza dell'intervallo diviso z, quando
    l'intervallo non viene troncato a [0, 1]."""
    z = ev.z_value()
    for seed in range(5):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, 300)
        p = y * 0.5 + rng.random(300)
        _, low, high = ev.delong(y, p)
        assert 0 < low and high < 1, "intervallo troncato: confronto non valido"
        assert pc.delong_se(y, p) == pytest.approx((high - low) / (2 * z))


def test_denoise_constant_removes_float_noise():
    """Costante dentro ogni fold con residuo di 1e-16: senza la correzione l'ordinamento fittizio
    porta l'AUC lontano da 0,5 (e' il caso reale del dummy nei bracci con i pesi)."""
    rng = np.random.default_rng(0)
    n = 400
    folds = np.arange(n) % 5
    y = (rng.random(n) < 0.25).astype(int)
    noisy = 0.5 + np.array([-3e-16, -1e-16, 0.0, 1e-16, 3e-16])[folds]
    g = pd.DataFrame({"fold": [f"outer{k}" for k in folds], "probability": noisy})
    assert ev.delong(y, noisy)[0] != pytest.approx(0.5, abs=1e-3)  # il problema esiste
    clean = pc.denoise_constant(noisy, g)
    assert np.ptp(clean) == 0
    assert ev.delong(y, clean)[0] == 0.5
    assert ev.pr_auc(y, clean)[0] == pytest.approx(y.mean())


def test_denoise_constant_leaves_real_models_alone():
    rng = np.random.default_rng(1)
    n = 400
    p = rng.random(n)
    g = pd.DataFrame({"fold": [f"outer{k}" for k in np.arange(n) % 5], "probability": p})
    assert pc.denoise_constant(p, g) is p


def test_delong_se_undefined_without_both_classes():
    assert np.isnan(pc.delong_se(np.ones(10), np.random.default_rng(0).random(10)))


# --- filtraggio del sottogruppo ---------------------------------------------------------------

def test_groups_partition_and_counts(synthetic):
    _, train, _ = synthetic
    masks = pc.groups(train)
    assert set(masks) == {pc.SUBGROUP, pc.REFERENCE, "tutti"}
    assert (masks[pc.SUBGROUP] | masks[pc.REFERENCE]).all()
    assert not (masks[pc.SUBGROUP] & masks[pc.REFERENCE]).any()
    assert masks[pc.SUBGROUP].sum() == int((train[COLUMNS["diabetes"]] == 1).sum())


def test_groups_rejects_unexpected_values(synthetic):
    _, train, _ = synthetic
    broken = train.copy()
    broken.loc[broken.index[0], COLUMNS["diabetes"]] = 9
    with pytest.raises(ValueError, match="valori inattesi"):
        pc.groups(broken)


def test_discrimination_uses_only_the_subgroup(tables, synthetic):
    _, train, _ = synthetic
    masks = pc.groups(train)
    table = tables["q1_discrimination"]
    for name, mask in masks.items():
        rows = table[table["group"] == name]
        assert (rows["n"] == mask.sum()).all()
        assert (rows["positives"] == target(train).to_numpy()[mask].sum()).all()


# --- soglia e fasce globali, mai ristimate ------------------------------------------------------

def test_threshold_is_the_given_one_not_refitted(tables, synthetic):
    """Con soglia 0,80 la sensibilita' nel sottogruppo deve essere lontanissima dallo 0,90 che il
    codice della Fase A otterrebbe ristimandola."""
    oof, train, _ = synthetic
    points = tables["q1_operating_points"]
    assert (points[points["technique"] == "none"]["threshold"] == ABSURD["none"]).all()
    assert (points[points["technique"] == "level_weight"]["threshold"] == ABSURD["level_weight"]).all()
    masks = pc.groups(train)
    y = target(train).to_numpy()
    for _, row in points.iterrows():
        g = oof[(oof["technique"] == row["technique"]) & (oof["model"] == row["model"])]
        p = pc.pooled(g, len(train))[masks[row["group"]]]
        expected = ev.operating_point(y[masks[row["group"]]], p, row["threshold"])
        assert row["recall"] == pytest.approx(expected["recall"])


def test_bands_are_the_given_ones_not_quantiles(tables, synthetic):
    oof, train, _ = synthetic
    masks = pc.groups(train)
    codes = np.asarray(pd.Categorical(kdigo_level(train), LEVELS, ordered=True).codes)
    bands = tables["q4_bands"]
    subset = bands[(bands["technique"] == "none") & (bands["model"] == "good")
                   & (bands["group"] == pc.SUBGROUP)]
    g = oof[(oof["technique"] == "none") & (oof["model"] == "good")]
    mask = masks[pc.SUBGROUP]
    assigned = ev.assign_bands(pc.pooled(g, len(train))[mask], ABSURD_BANDS["none"])
    for _, row in subset.iterrows():
        expected = int(np.sum((codes[mask] == LEVELS.index(row["level"]))
                              & (assigned == row["band"] - 1)))
        assert row["n"] == expected
    assert subset["n"].sum() == mask.sum()


def test_sensitivity_severe_row_is_the_union(tables):
    table = tables["q3_sensitivity"]
    for key, block in table.groupby(["technique", "model", "group"]):
        rows = block.set_index("level")
        severe = rows.loc[pc.SEVERE_LABEL]
        parts = rows.loc[[LEVELS[i] for i in pc.SEVERE]]
        assert severe["n"] == parts["n"].sum(), key
        assert severe["detected"] == parts["detected"].sum(), key


# --- confronto fra sottogruppi ------------------------------------------------------------------

def test_subgroup_comparison_only_auc_no_pr_auc(tables):
    """La differenza di PR-AUC non deve comparire: favorisce i gruppi ad alta prevalenza."""
    table = tables["subgroup_comparison"]
    assert (table["metric"] == "auc").all()
    assert not any("pr_auc" in column for column in table.columns)
    assert {"prevalence_group", "prevalence_reference"} <= set(table.columns)


def test_subgroup_comparison_arithmetic(tables):
    discrimination = tables["q1_discrimination"].set_index(["technique", "model", "group"])
    z = ev.z_value()
    for _, row in tables["subgroup_comparison"].iterrows():
        a = discrimination.loc[(row["technique"], row["model"], pc.SUBGROUP)]
        b = discrimination.loc[(row["technique"], row["model"], pc.REFERENCE)]
        assert row["difference"] == pytest.approx(a["auc"] - b["auc"])
        se = np.sqrt(a["auc_se"] ** 2 + b["auc_se"] ** 2)
        assert row["high"] - row["low"] == pytest.approx(2 * z * se)


def test_subgroup_comparison_skips_models_without_both_groups():
    rows = [{"technique": "none", "model": "solo", "group": pc.SUBGROUP, "auc": 0.7,
             "auc_se": 0.05, "prevalence": 0.26}]
    assert pc.subgroup_comparison(rows).empty


# --- soglia descrittiva -------------------------------------------------------------------------

def test_descriptive_threshold_reaches_the_target_inside_the_group(tables, synthetic):
    oof, train, _ = synthetic
    masks = pc.groups(train)
    y = target(train).to_numpy()
    for _, row in tables["threshold_descriptive"].iterrows():
        g = oof[(oof["technique"] == row["technique"]) & (oof["model"] == row["model"])]
        mask = masks[row["group"]]
        p = pc.pooled(g, len(train))[mask]
        detected = np.mean(p[y[mask] == 1] >= row["threshold_subgroup"])
        assert detected >= row["target_sensitivity"] - 1e-12
        assert row["threshold_global"] == ABSURD[row["technique"]]


# --- controllo di coerenza e guardie ------------------------------------------------------------

def test_dummy_check_passes_on_real_shaped_data(tables):
    dummy = tables["q1_discrimination"].query("model == 'dummy'")
    assert len(dummy) == 2 * 3  # 2 tecniche x 3 gruppi
    assert dummy["auc"].eq(0.5).all()
    assert np.allclose(dummy["pr_auc"], dummy["prevalence"])


def test_dummy_check_raises_when_broken(tables):
    broken = tables["q1_discrimination"].copy()
    broken.loc[broken["model"] == "dummy", "auc"] = 0.6
    with pytest.raises(RuntimeError, match="coerenza fallito"):
        pc.check_dummy(broken)
    with pytest.raises(RuntimeError, match="coerenza non verificabile"):
        pc.check_dummy(broken[broken["model"] != "dummy"])


def test_dummy_has_no_threshold_dependent_rows(tables):
    """Senza riga di soglie il classificatore di maggioranza entra solo nella domanda 1."""
    for name in ["q1_operating_points", "q2_levels", "q2_trend", "q3_sensitivity",
                 "q4_bands", "q4_kappa", "threshold_descriptive"]:
        assert "dummy" not in set(tables[name]["model"]), name


def test_missing_cutpoints_for_a_real_model_is_an_error(synthetic):
    """Difetto trovato in revisione: senza guardia, un modello vero assente da cutpoints.csv
    sparirebbe in silenzio da q3_sensitivity, cioe' dal risultato principale della Fase C."""
    oof, train, cutpoints = synthetic
    mutilated = cutpoints[~((cutpoints["technique"] == "none") & (cutpoints["model"] == "good"))]
    with pytest.raises(RuntimeError, match="nessuna soglia"):
        pc.evaluate(oof, train, mutilated, n_boot=5)


def test_constant_model_without_cutpoints_is_allowed(tables):
    """Il classificatore di maggioranza non ha soglia e non deve far fallire la valutazione."""
    assert "dummy" in set(tables["q1_discrimination"]["model"])


def test_primary_and_secondary_are_the_two_declared_arms():
    assert {pc.PRIMARY, pc.SECONDARY} == set(pc.TECHNIQUES)
    assert pc.PRIMARY != pc.SECONDARY


def test_missing_technique_is_an_error(synthetic):
    oof, train, cutpoints = synthetic
    with pytest.raises(RuntimeError, match="tecniche assenti"):
        pc.evaluate(oof[oof["technique"] == "none"], train, cutpoints, n_boot=5)


def test_incomplete_predictions_are_an_error(synthetic):
    oof, train, cutpoints = synthetic
    with pytest.raises(RuntimeError, match="senza previsione out-of-fold"):
        pc.evaluate(oof[oof["row"] != 0], train, cutpoints, n_boot=5)


def test_only_the_declared_techniques_are_used(tables):
    for name, table in tables.items():
        assert set(table["technique"]) <= set(pc.TECHNIQUES), name


def test_no_p_values_anywhere(tables):
    """Protocollo, punto 5: nessun test formale, quindi nessuna colonna di p."""
    for name, table in tables.items():
        assert not [c for c in table.columns if "p_value" in c or c == "p_holm"], name


def test_run_never_reads_the_test_set(monkeypatch, synthetic, tmp_path):
    """Guardia esplicita: `run` puo' aprire solo train.csv, le previsioni e le soglie."""
    oof, train, cutpoints = synthetic
    opened = []
    real = pd.read_csv

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        if str(path).endswith("train.csv"):
            return train
        if str(path) == str(pc.PREDICTIONS):
            return oof
        if str(path) == str(pc.CUTPOINTS):
            return cutpoints
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    pc.run(output=tmp_path)
    assert not [path for path in opened if "test.csv" in path]
    assert len(opened) == 3
    expected = {f"{name}.csv" for name in pc.TABLES + ["subgroup_comparison"]}
    assert {path.name for path in tmp_path.glob("*.csv")} == expected
