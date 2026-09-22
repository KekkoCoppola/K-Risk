"""Conferma finale sul test set: si esegue UNA volta sola.

Protocollo: Notepad, "Conferma finale sul test set — protocollo" (22/09/2026), e blocco final_test
di configs/config.yaml, entrambi fissati prima di scrivere questo codice. Nessun modello nuovo e
nessun riaddestramento: si usano i modelli finali già salvati; soglie e fasce sono quelle stimate
sulle previsioni out-of-fold e non vengono mai ristimate sul test.

Ordine di run(): autorizzazione e albero git pulito; poi tutti i controlli che non richiedono il
test (preprocessore ristimato identico bit a bit alla cache, 50 modelli, parametri di Platt, soglie,
AUROC out-of-fold); solo allora RUN.json registra l'apertura e il test viene letto. Una seconda
esecuzione richiede un motivo dichiarato, che resta scritto in RUN.json (regola d'oro).
`--check` esegue tutti i controlli e una prova completa sulle righe del training, senza leggere il test.
"""
import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.base import clone

import src.models.evaluation as ev
import src.models.phase_c as pc
from src.config import CONFIG, ROOT, resolve
from src.data.imputation import chosen_imputer
from src.data.imputed import load_fold
from src.data.kidney import LEVELS, kdigo_level, target
from src.data.preprocess import build_preprocessor, feature_types, select_features
from src.data.split import PROCESSED
from src.models.clinical_utility import (calibration_row, check_decision_curves, decision_curve,
                                         simple_indices)
from src.models.evaluation_b import CONFIRMATORY, mcnemar_exact, newcombe_paired
from src.models.phase_b import platt_apply
from src.models.zoo import columns

SETTINGS = CONFIG["final_test"]
OUTPUT = resolve(SETTINGS["output"])
FEATURE_SET = SETTINGS["feature_set"]
TECHNIQUES = SETTINGS["techniques"]
REAL = SETTINGS["real_models"]
CLAIMS = SETTINGS["claims"]
EXPECTED = SETTINGS["expected"]
MODELS = CONFIG["phase_a"]["models"]
MODELS_DIR = resolve(CONFIG["phase_a"]["models_dir"])
MODELS_B = resolve(CONFIG["phase_b"]["models_dir"])
PHASE_B = resolve(CONFIG["phase_b"]["output"])
CUTPOINTS_A = resolve(CONFIG["evaluation"]["output"]) / "cutpoints.csv"
CUTPOINTS_B = resolve(CONFIG["phase_b"]["evaluation"]["output"]) / "cutpoints.csv"
OOF_A = resolve(CONFIG["evaluation"]["output"]) / "q1_discrimination.csv"
EXPLORATORY = [t for t, spec in CONFIG["phase_b"]["techniques"].items() if spec.get("exploratory")]
ID = CONFIG["columns"]["id"]
DIABETES = CONFIG["columns"]["diabetes"]
ALPHA = CONFIG["evaluation"]["alpha"]
N_BOOT = CONFIG["evaluation"]["bootstrap"]
SEED = CONFIG["seed"]
# braccio della Fase A (XGBoost dello spazio 3-12); gli altri bracci sono le tecniche della Fase B,
# dove "none" usa XGBoost con profondità 1-12 come in tutte le analisi dalla Fase B in poi
PHASE_A_ARM = "fase_a"
TOTAL = "tutti"
TABLES = ["q1_discrimination", "q1_operating_points", "q2_levels", "q2_trend", "q3_sensitivity",
          "q4_bands", "q4_kappa"]


def italian(value, decimals=2):
    return f"{value:.{decimals}f}".replace(".", ",")


# --- modelli e parametri già salvati ----------------------------------------------------------

def model_path(arm, name, models_dir=MODELS_DIR, models_b=MODELS_B):
    """Modello finale addestrato sull'intero training per un braccio e un modello."""
    if arm == PHASE_A_ARM:
        return models_dir / f"phase_a_{name}_{FEATURE_SET}.joblib"
    if arm == "none":
        if name == "xgboost":
            return models_dir / "sensitivity" / "depth_1_12" / f"phase_a_xgboost_{FEATURE_SET}.joblib"
        return models_dir / f"phase_a_{name}_{FEATURE_SET}.joblib"
    return models_b / arm / f"phase_b_{arm}_{name}_{FEATURE_SET}.joblib"


def platt_params(arm, name, phase_b=PHASE_B):
    """(a, b) di Platt del modello finale; None per il braccio della Fase A e per CTGAN, che in
    Fase B non sono stati ricalibrati."""
    if arm == PHASE_A_ARM or arm in EXPLORATORY:
        return None
    record = json.loads((phase_b / arm / "calibration" / FEATURE_SET / f"{name}_full.json")
                        .read_text(encoding="utf-8"))
    return record["a"], record["b"]


def load_models(techniques=TECHNIQUES, models=MODELS, loader=joblib.load):
    """Tutti i modelli finali e i parametri di Platt, caricati prima di aprire il test. Chi ha
    n_jobs viene portato a 1: la Random Forest in parallelo non è riproducibile bit a bit."""
    loaded = {}
    for arm in [PHASE_A_ARM, *techniques]:
        for name in models:
            model = loader(model_path(arm, name))
            if "n_jobs" in model.get_params():
                model.set_params(n_jobs=1)
            loaded[arm, name] = (model, platt_params(arm, name))
    return loaded


def cutpoints_for(arm, name, cutpoints_a, cutpoints_b):
    """Soglia e fasce fissate sulle previsioni out-of-fold, o None se il modello non ne ha."""
    if arm == PHASE_A_ARM:
        rows = cutpoints_a[(cutpoints_a["feature_set"] == FEATURE_SET) & (cutpoints_a["model"] == name)]
    else:
        rows = cutpoints_b[(cutpoints_b["technique"] == arm) & (cutpoints_b["model"] == name)]
    if len(rows) > 1:
        raise RuntimeError(f"{arm} {name}: più di una riga di soglie")
    if rows.empty:
        return None
    row = rows.iloc[0]
    return float(row["threshold"]), row[pc.BAND_COLUMNS].to_numpy(dtype=float)


def check_inputs(cutpoints_a, cutpoints_b, oof_auc, techniques=TECHNIQUES, real=REAL):
    """Tutto ciò che le tabelle e le affermazioni richiedono, verificato prima di aprire il test."""
    missing = set(CONFIRMATORY) - set(techniques)
    if missing:
        raise RuntimeError(f"tecniche principali assenti da final_test.techniques: {sorted(missing)}")
    for arm in [PHASE_A_ARM, *techniques]:
        for name in real:
            if cutpoints_for(arm, name, cutpoints_a, cutpoints_b) is None:
                raise RuntimeError(f"{arm} {name}: nessuna soglia fissata")
    oof = oof_auc[oof_auc["feature_set"] == FEATURE_SET]
    absent = set(real) - set(oof["model"])
    if absent:
        raise RuntimeError(f"AUROC out-of-fold mancanti: {sorted(absent)}")


# --- preparazione del test ------------------------------------------------------------------

def check_test_frame(train, test, expected):
    """Controlli appena letto il test: nessun soggetto in comune col training e numerosità uguali
    a quelle note dallo split. Se uno fallisce non si calcola nessuna metrica."""
    shared = set(train[ID]) & set(test[ID])
    if shared:
        raise RuntimeError(f"{len(shared)} identificativi del test compaiono anche nel training")
    levels = pd.Series(np.asarray(kdigo_level(test)))
    found = {"n": len(test), "positives": int(target(test).sum()),
             "alto": int((levels == "alto").sum()), "molto alto": int((levels == "molto alto").sum()),
             "diabetici": int((test[DIABETES] == 1).sum())}
    wrong = {k: (found[k], v) for k, v in expected.items() if found[k] != v}
    if wrong:
        raise RuntimeError(f"numerosità del test diverse da quelle dello split (trovate, attese): {wrong}")
    return found


def fit_preprocessor(train, imputer=None, reference=None, feature_set=FEATURE_SET):
    """Preprocessore ristimato sull'intero training. reference: training intero imputato della
    cache (fold "full"); se data, la ristima deve riprodurlo bit a bit, altrimenti il test verrebbe
    trasformato in modo diverso da come sono stati addestrati i modelli."""
    X_train = select_features(train, feature_set)
    preprocessor = build_preprocessor(clone(imputer if imputer is not None else chosen_imputer()),
                                      X_train.columns).fit(X_train)
    numeric, categorical = feature_types(X_train.columns)
    order = numeric + categorical
    if reference is not None:
        refitted = preprocessor.transform(X_train)
        if list(reference.columns) != order or not np.array_equal(refitted, reference.to_numpy()):
            raise RuntimeError("il preprocessore ristimato non riproduce la cache del training intero")
    return preprocessor, order


def transform_test(preprocessor, order, test, feature_set=FEATURE_SET):
    """Una sola .transform: il test non entra mai nella stima."""
    return pd.DataFrame(preprocessor.transform(select_features(test, feature_set)), columns=order,
                        index=pd.RangeIndex(len(test)))


def predict_one(model, name, X):
    """Probabilità sulle colonne con cui il modello è stato addestrato, con controllo esplicito.
    XGBoost conosce i nomi delle colonne; i modelli sklearn solo il loro numero (la cache salva i
    nomi come np.str_), quindi ricevono un array nello stesso ordine usato in addestramento."""
    use = columns(name, X.columns)
    if hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
        if names is not None and list(names) != list(use):
            raise RuntimeError(f"{name}: colonne diverse da quelle di addestramento")
        return model.predict_proba(X[use])[:, 1]
    expected = getattr(model, "n_features_in_", len(use))
    if expected != len(use):
        raise RuntimeError(f"{name}: attese {expected} colonne, passate {len(use)}")
    return model.predict_proba(X[use].to_numpy())[:, 1]


def predict_all(X, loaded):
    """Probabilità di ogni modello finale, in formato lungo come le previsioni out-of-fold
    (fold = "test"): grezze e, dove esistono i parametri di Platt, ricalibrate."""
    frames = []
    for (arm, name), (model, params) in loaded.items():
        p = predict_one(model, name, X)
        calibrated = platt_apply(p, *params) if params is not None else np.full(len(p), np.nan)
        frames.append(pd.DataFrame({"row": np.arange(len(p)), "fold": "test", "model": name,
                                    "technique": arm, "probability": p,
                                    "probability_calibrated": calibrated}))
    return pd.concat(frames, ignore_index=True)


# --- valutazione ------------------------------------------------------------------------------

def pooled_test(g, n, column="probability"):
    """Probabilità di un gruppo nell'ordine delle righe del test; errore se ne manca qualcuna."""
    p = g.set_index("row")[column].reindex(range(n)).to_numpy()
    if np.isnan(p).any():
        raise RuntimeError(f"{g['technique'].iloc[0]} {g['model'].iloc[0]}: valori mancanti in {column}")
    return p


def severe_comparison(predictions, test, cutpoints_b, techniques, real=REAL, alpha=ALPHA):
    """Affermazione (B): casi gravi riconosciuti da ogni tecnica principale contro "nessuna
    correzione", ciascuna alla propria soglia fissata; McNemar esatto, Newcombe, Holm per modello."""
    codes = np.asarray(pd.Categorical(kdigo_level(test), LEVELS, ordered=True).codes)
    severe = np.isin(codes, pc.SEVERE)
    n = len(test)
    confirmatory = [t for t in CONFIRMATORY if t in techniques]
    rows = []
    for name in real:
        detected = {}
        for arm in ["none", *confirmatory]:
            g = predictions[(predictions["technique"] == arm) & (predictions["model"] == name)]
            cut = cutpoints_for(arm, name, None, cutpoints_b)
            if cut is None:
                raise RuntimeError(f"{arm} {name}: nessuna soglia fissata")
            detected[arm] = pooled_test(g, n)[severe] >= cut[0]
        block = []
        for arm in confirmatory:
            gained, lost, p_value = mcnemar_exact(detected[arm], detected["none"])
            difference, low, high = newcombe_paired(detected[arm], detected["none"], alpha)
            block.append({"model": name, "technique": arm, "severe_n": int(severe.sum()),
                          "detected": int(detected[arm].sum()), "detected_none": int(detected["none"].sum()),
                          "gained": gained, "lost": lost, "difference": difference, "low": low,
                          "high": high, "p_value": p_value})
        block = pd.DataFrame(block)
        if len(block):
            block["p_holm"] = ev.holm(block["p_value"])
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def naive_table(predictions, test, threshold=CLAIMS["naive_threshold"], alpha=ALPHA):
    """Affermazione (E): recall alla soglia 0,5 sulle probabilità grezze, per braccio e modello."""
    y = target(test).to_numpy()
    rows = []
    for (arm, name), g in predictions.groupby(["technique", "model"], sort=False):
        point = ev.operating_point(y, pooled_test(g, len(test)), threshold, alpha)
        rows.append({"technique": arm, "model": name, **point})
    return pd.DataFrame(rows)


def calibration_table(predictions, test, techniques, real=REAL, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """Calibrazione con IC da bootstrap semplice: "nessuna correzione" grezza e ricalibrata, le
    tecniche principali ricalibrate, in ciascun gruppo."""
    y_all = target(test).to_numpy()
    masks = pc.groups(test)
    samples = {g: list(simple_indices(int(m.sum()), n_boot, seed)) for g, m in masks.items()}
    arms = [("none", "grezza", "probability"), ("none", "ricalibrata", "probability_calibrated")]
    arms += [(t, "ricalibrata", "probability_calibrated") for t in CONFIRMATORY if t in techniques]
    rows = []
    for arm, kind, column in arms:
        for name in real:
            g = predictions[(predictions["technique"] == arm) & (predictions["model"] == name)]
            p_all = pooled_test(g, len(test), column)
            for group, mask in masks.items():
                row = calibration_row(y_all[mask], p_all[mask], samples[group], alpha)
                rows.append({"technique": arm, "probabilities": kind, "model": name, "group": group,
                             **row})
    return pd.DataFrame(rows)


def decision_table(predictions, test, models=MODELS):
    """Decision curve del braccio "nessuna correzione" su tutto il test e nei due sottogruppi, con
    i controlli di coerenza del blocco qualità."""
    y_all = target(test).to_numpy()
    rows = []
    for name in models:
        g = predictions[(predictions["technique"] == "none") & (predictions["model"] == name)]
        p_all = pc.denoise_constant(pooled_test(g, len(test)), g)
        for group, mask in pc.groups(test).items():
            rows += [{"group": group, "model": name, **r} for r in decision_curve(y_all[mask], p_all[mask])]
    table = pd.DataFrame(rows)
    check_decision_curves(table)
    return table


def require(index, expected, what):
    if set(index) != set(expected) or len(index) != len(expected):
        raise RuntimeError(f"{what}: attesi {sorted(expected)}, trovati {sorted(index)}")


def claims_table(tables, oof_auc, techniques, real=REAL, claims=CLAIMS):
    """Esito delle affermazioni (A)-(E) con i criteri fissati nel protocollo. Tabelle incomplete
    fermano il calcolo: un esito deciso su meno modelli del previsto sarebbe sbagliato in silenzio."""
    need = claims["min_models"]
    confirmatory = [t for t in CONFIRMATORY if t in techniques]
    rows = []

    disc = tables["q1_discrimination"]
    disc = disc[(disc["technique"] == PHASE_A_ARM) & (disc["group"] == TOTAL)
                & disc["model"].isin(real)].set_index("model")
    require(disc.index, real, "(A) AUROC del test")
    oof = oof_auc[oof_auc["feature_set"] == FEATURE_SET].set_index("model")["auc"]
    details, inside = [], 0
    for name in real:
        low, high, value = disc.loc[name, "auc_low"], disc.loc[name, "auc_high"], oof.loc[name]
        ok = low <= value <= high
        inside += ok
        where = "dentro" if ok else ("sopra (test più basso)" if value > high else "sotto (test più alto)")
        details.append(f"{name}: out-of-fold {italian(value, 3)}, test {italian(disc.loc[name, 'auc'], 3)} "
                       f"({italian(low, 3)}-{italian(high, 3)}), {where}")
    rows.append({"claim": "A", "statement": "AUROC del test compatibile con quella out-of-fold",
                 "criterion": f"stima out-of-fold dentro l'IC 95% del test per almeno {need} modelli su {len(real)}",
                 "detail": "; ".join(details),
                 "verdict": "non contraddetta" if inside >= need else "contraddetta"})

    primary = tables["primary_comparison"]
    for name in real:
        require(primary.loc[primary["model"] == name, "technique"], confirmatory, f"(B) {name}")
    against = primary[(primary["p_holm"] < 0.05) & (primary["gained"] > primary["lost"])]
    net = primary["gained"] - primary["lost"]
    rows.append({"claim": "B", "statement": "nessuna tecnica principale riconosce più casi gravi",
                 "criterion": "contraddetta se una tecnica ha p di Holm < 0,05 a favore della tecnica",
                 "detail": f"p di Holm minimo {italian(primary['p_holm'].min(), 3)}; differenze da "
                           f"{int(net.min()):+d} a {int(net.max()):+d} casi gravi",
                 "verdict": "contraddetta" if len(against) else "non contraddetta"})

    points = tables["q1_operating_points"]
    points = points[(points["technique"] == "none") & (points["group"] == "diabetici")
                    & points["model"].isin(real)].set_index("model")["alert_rate"]
    require(points.index, real, "(C) quota di allerta fra i diabetici")
    hits = int((points >= claims["diabetic_alert_rate"]).sum())
    rows.append({"claim": "C", "statement": "alla soglia globale il modello segnala quasi tutti i diabetici",
                 "criterion": f"quota di allerta fra i diabetici >= {italian(claims['diabetic_alert_rate'])} "
                              f"per almeno {need} modelli su {len(real)}",
                 "detail": "; ".join(f"{m}: {italian(v, 3)}" for m, v in points.items()),
                 "verdict": "confermata" if hits >= need else "non confermata"})

    curve = tables["decision_curve"]
    curve = curve[(curve["group"] == TOTAL) & curve["model"].isin(real)
                  & curve["threshold"].isin(claims["utility_thresholds"])]
    for name in real:
        found = sorted(curve.loc[curve["model"] == name, "threshold"])
        if found != sorted(claims["utility_thresholds"]):
            raise RuntimeError(f"(D) {name}: soglie trovate {found}, attese {claims['utility_thresholds']}")
    useful = curve.groupby("model")["beats_both"].all()
    rows.append({"claim": "D", "statement": "utilità clinica alle soglie 7% e 10%",
                 "criterion": f"net benefit sopra 'testare tutti' e 'nessuno' a entrambe le soglie per almeno "
                              f"{need} modelli su {len(real)}",
                 "detail": "; ".join(f"{m}: {'sì' if v else 'no'}" for m, v in useful.items()),
                 "verdict": "confermata" if int(useful.sum()) >= need else "non confermata"})

    naive = tables["naive"]
    naive = naive[naive["model"].isin(real)]
    for arm in ["none", *confirmatory]:
        require(naive.loc[naive["technique"] == arm, "model"], real, f"(E) {arm}")
    recall = naive.groupby("technique")["recall"].mean()
    higher = {t: recall[t] > recall["none"] for t in confirmatory}
    rows.append({"claim": "E", "statement": "alla soglia 0,5 il bilanciamento sembra migliorare il recall",
                 "criterion": "recall medio a 0,5 più alto di 'nessuna correzione' per tutte le tecniche principali",
                 "detail": f"none {italian(recall['none'], 3)}; "
                           + "; ".join(f"{t} {italian(recall[t], 3)}" for t in confirmatory),
                 "verdict": "confermata" if all(higher.values()) else "non confermata"})
    return pd.DataFrame(rows)


def evaluate(predictions, test, cutpoints_a, cutpoints_b, oof_auc, techniques=TECHNIQUES,
             n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """Tutte le tabelle della conferma finale, come dict nome -> DataFrame. Soglie e fasce sono
    quelle date, mai ristimate: si usano solo gli helper di phase_c a soglia fissa."""
    y_all = target(test).to_numpy()
    codes_all = np.asarray(pd.Categorical(kdigo_level(test), LEVELS, ordered=True).codes)
    masks = pc.groups(test)
    samples = {name: list(ev.stratified_indices(codes_all[mask], n_boot, seed)) for name, mask in masks.items()}
    rows = {name: [] for name in TABLES}
    for (arm, name), g in predictions.groupby(["technique", "model"], sort=False):
        p_all = pc.denoise_constant(pooled_test(g, len(test)), g)
        cut = cutpoints_for(arm, name, cutpoints_a, cutpoints_b)
        if cut is None and not ev.is_constant(g):
            raise RuntimeError(f"{arm} {name}: nessuna soglia fissata, ma le previsioni non sono costanti")
        for group, mask in masks.items():
            key = {"technique": arm, "model": name, "group": group}
            p, y, codes = p_all[mask], y_all[mask], codes_all[mask]
            rows["q1_discrimination"].append({**key, **pc.discrimination(p, y, alpha)})
            if cut is None:
                continue
            threshold, band_cuts = cut
            rows["q1_operating_points"].append({**key, **ev.operating_point(y, p, threshold, alpha)})
            rows["q2_levels"] += [{**key, **r} for r in pc.levels_table(p, codes, samples[group], alpha)]
            rows["q2_trend"].append({**key, **pc.trend_row(p, codes, samples[group], alpha)})
            rows["q3_sensitivity"] += [{**key, **r} for r in pc.sensitivity_table(p, codes, threshold, alpha)]
            band_rows, bands = pc.bands_table(p, codes, band_cuts)
            rows["q4_bands"] += [{**key, **r} for r in band_rows]
            rows["q4_kappa"].append({**key, **pc.kappa_row(codes, bands, samples[group], alpha)})
    tables = {name: pd.DataFrame(rows[name]) for name in TABLES}
    pc.check_dummy(tables["q1_discrimination"])
    tables["primary_comparison"] = severe_comparison(predictions, test, cutpoints_b, techniques, alpha=alpha)
    tables["naive"] = naive_table(predictions, test, alpha=alpha)
    tables["calibration"] = calibration_table(predictions, test, techniques, n_boot=n_boot, seed=seed,
                                              alpha=alpha)
    tables["decision_curve"] = decision_table(predictions, test)
    tables["claims"] = claims_table(tables, oof_auc, techniques)
    return tables


# --- controlli ed esecuzione unica ------------------------------------------------------------

def git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT,
                          check=True).stdout.strip()


def git_clean():
    """Il codice eseguito deve essere quello del commit registrato: niente modifiche in sospeso."""
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT,
                            check=True).stdout.strip()
    if status:
        raise RuntimeError(f"albero git non pulito: committare prima di eseguire\n{status}")


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}


def authorize(settings, run_file, rerun_reason=None):
    """Blocca l'esecuzione prima di leggere il test se manca l'autorizzazione o se il test è già
    stato usato senza un motivo dichiarato."""
    if settings.get("authorized") is not True:
        raise RuntimeError("final_test.authorized è false: il test set non viene letto")
    if run_file.exists() and not rerun_reason:
        raise RuntimeError(f"{run_file} esiste: il test è già stato usato. Per rieseguire serve "
                           f"--rerun-reason, che resta scritto e va dichiarato nella tesi")


def preflight(train, techniques=TECHNIQUES, loader=joblib.load):
    """Tutti i controlli che non richiedono il test: preprocessore identico bit a bit alla cache,
    modelli e parametri di Platt, soglie e AUROC out-of-fold complete."""
    reference, _ = load_fold(FEATURE_SET)
    preprocessor, order = fit_preprocessor(train, reference=reference)
    loaded = load_models(techniques, loader=loader)
    cutpoints_a, cutpoints_b, oof_auc = (pd.read_csv(CUTPOINTS_A), pd.read_csv(CUTPOINTS_B),
                                         pd.read_csv(OOF_A))
    check_inputs(cutpoints_a, cutpoints_b, oof_auc, techniques)
    return {"preprocessor": preprocessor, "order": order, "loaded": loaded, "cutpoints_a": cutpoints_a,
            "cutpoints_b": cutpoints_b, "oof_auc": oof_auc}


def dry_run(train, prepared, n_rows=1451, n_boot=50):
    """Prova completa su righe del TRAINING (numeri senza valore: righe viste in addestramento), per
    verificare che trasformazione, previsioni e tabelle girino sui file reali."""
    rows = train.sample(n_rows, random_state=SEED).reset_index(drop=True)
    X = transform_test(prepared["preprocessor"], prepared["order"], rows)
    predictions = predict_all(X, prepared["loaded"])
    return evaluate(predictions, rows, prepared["cutpoints_a"], prepared["cutpoints_b"],
                    prepared["oof_auc"], n_boot=n_boot)


def write_run(run_file, history):
    run_file.write_text(json.dumps({"runs": history}, indent=1, ensure_ascii=False), encoding="utf-8")


def run(rerun_reason=None, output=OUTPUT, git_check=True):
    run_file = output / "RUN.json"
    authorize(SETTINGS, run_file, rerun_reason)
    if git_check:
        git_clean()
    train = pd.read_csv(PROCESSED / "train.csv")
    prepared = preflight(train)
    # da qui il test viene aperto: l'uso resta registrato anche se qualcosa fallisce dopo
    output.mkdir(parents=True, exist_ok=True)
    history = json.loads(run_file.read_text(encoding="utf-8"))["runs"] if run_file.exists() else []
    entry = {"run": len(history) + 1, "started": datetime.now().isoformat(timespec="seconds"),
             "commit": git_commit() if git_check else "non verificato", "versions": versions(),
             "reason": rerun_reason or "prima esecuzione", "status": "aperto"}
    history.append(entry)
    write_run(run_file, history)
    path = PROCESSED / "test.csv"
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    test = pd.read_csv(path)
    entry["counts"] = check_test_frame(train, test, EXPECTED)
    write_run(run_file, history)
    X_test = transform_test(prepared["preprocessor"], prepared["order"], test)
    predictions = predict_all(X_test, prepared["loaded"])
    run_dir = output / f"run_{entry['run']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    codes = pd.Categorical(kdigo_level(test), LEVELS, ordered=True)
    subjects = pd.DataFrame({"row": np.arange(len(test)), ID: test[ID], "target": target(test).to_numpy(),
                             "level": np.asarray(codes), DIABETES: test[DIABETES].to_numpy()})
    predictions.merge(subjects, on="row").to_csv(run_dir / "test_predictions.csv", index=False)
    tables = evaluate(predictions, test, prepared["cutpoints_a"], prepared["cutpoints_b"],
                      prepared["oof_auc"])
    for name, table in tables.items():
        table.to_csv(run_dir / f"{name}.csv", index=False)
        print(f"{run_dir / name}.csv", flush=True)
    # tabelle nella sottocartella accanto a RUN.json: una per esecuzione, nessuna sovrascrittura
    entry.update(status="completato", finished=datetime.now().isoformat(timespec="seconds"),
                 tables=run_dir.name)
    write_run(run_file, history)
    print(tables["claims"][["claim", "verdict", "detail"]].to_string(index=False), flush=True)


def check():
    """--check: tutto ciò che precede l'apertura del test, più la prova sulle righe del training."""
    train = pd.read_csv(PROCESSED / "train.csv")
    prepared = preflight(train)
    print("preprocessore identico alla cache, modelli, parametri e soglie completi", flush=True)
    tables = dry_run(train, prepared)
    print("prova completa sulle righe del training superata (numeri senza valore):", flush=True)
    print(tables["claims"][["claim", "verdict"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="controlli e prova sul training, senza il test")
    group.add_argument("--run", action="store_true", help="esegue la conferma finale (una volta sola)")
    parser.add_argument("--rerun-reason", default=None,
                        help="motivo di una seconda esecuzione: resta in RUN.json e va dichiarato")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        run(args.rerun_reason)
