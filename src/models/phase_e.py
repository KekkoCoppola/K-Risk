"""Fase E: richieste dei relatori (analisi post-hoc ed esplorativa, registrata il 29/09/2026).

(1) Zoo: undici famiglie di modelli mai provate nelle fasi A-D, sul target composito binario, con la
    stessa validazione annidata; le sette famiglie già calcolate si riusano senza riaddestrarle
    (previsioni out-of-fold esterne dai loro record).
(2) ACR ed eGFR combinati, senza soglia: i modelli imparano log(ACR) e log(creatinina) come valori
    continui e non vedono mai la soglia; la definizione KDIGO entra solo alla fine, nella probabilità
    del target composito calcolata dalla distribuzione congiunta dei residui out-of-fold interni.
(3) Stacking: un meta-modello logistico impara a combinare le previsioni di tutti i modelli; la media
    dei ranghi è la combinazione senza parametri. Cross-fitting sui fold esterni.
(4) Pannello di metriche: ricalibrazione e soglie del fold k stimate solo sugli altri fold esterni.
Protocollo in config (phase_e), fissato prima del codice e dei calcoli. Il test set non viene letto.
"""
import argparse
import json
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from scipy import stats
from scipy.special import logit
from sklearn.compose import make_column_transformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, PolynomialFeatures, QuantileTransformer, SplineTransformer,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import src.models.evaluation as ev
from src.config import CONFIG, resolve
from src.data.folds import INNER, OUTER
from src.data.imputed import fold_name
from src.data.kidney import (CREATININE_FACTOR, KAPPA, LEVELS, SEX_FACTOR, egfr, is_female, kdigo_level,
                             target)
from src.data.preprocess import NUMERIC
from src.data.split import PROCESSED
from src.models import zoo
from src.models.evaluation_b import calibration
from src.models.phase_d import components, loader
from src.models.zoo import SEED, suggest

SETTINGS = CONFIG["phase_e"]
OUTPUT = resolve(SETTINGS["output"])
N_TRIALS = SETTINGS["n_trials"]
CLIP = SETTINGS["clip"]
ZOO = list(SETTINGS["zoo"])
REUSED = SETTINGS["reused"]
SPACES = SETTINGS["spaces"]
CONTINUOUS = SETTINGS["continuous"]
ARMS = CONTINUOUS["arms"]
STACKING = SETTINGS["stacking"]
EVALUATION = SETTINGS["evaluation"]
LGBM_SPACE = CONFIG["phase_d"]["spaces"]["lightgbm"]
REFERENCES = CONFIG["phase_d"]["references"]
COLUMNS = CONFIG["columns"]
THRESHOLDS = CONFIG["target"]
LOG_ACR_CUT = float(np.log(THRESHOLDS["acr_threshold"]))
# punteggi che non sono probabilità: la funzione di decisione si usa così com'è, senza logit
DECISION = {"svm_rbf"}
# braccio continuo le cui uscite (probabilità e margini previsti) entrano nello stacking
STACK_CONTINUOUS = "combinato_lgbm"
COMBINED = ["stacking", "rank_mean"]
# eccezioni di un tentativo di Optuna da registrare come fallito invece di fermare tutto (per esempio
# una QDA numericamente singolare): il numero di tentativi falliti finisce nel record
TRIAL_ERRORS = (ValueError, FloatingPointError, np.linalg.LinAlgError)


# --- record -------------------------------------------------------------------------------

def record_path(tag, outer, output=OUTPUT):
    return output / tag / f"{fold_name(outer)}.json"


def save_record(tag, outer, record, output=OUTPUT):
    path = record_path(tag, outer, output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def read_record(tag, outer, output=OUTPUT):
    return json.loads(record_path(tag, outer, output).read_text(encoding="utf-8"))


def done(tag, output=OUTPUT):
    return all(record_path(tag, outer, output).exists() for outer in range(OUTER))


def source_path(name, outer):
    """Record di una famiglia già calcolata: analytics/phase_a/.../<nome>_outerK.json oppure
    analytics/phase_d/<nome>/outerK.json."""
    directory = resolve(REUSED[name])
    for path in (directory / f"{name}_{fold_name(outer)}.json", directory / name / f"{fold_name(outer)}.json"):
        if path.exists():
            return path
    raise FileNotFoundError(f"{name}: nessun record per {fold_name(outer)} in {directory}")


# --- modelli ------------------------------------------------------------------------------

def mlp_arguments(params):
    """Argomenti di MLPClassifier/MLPRegressor: "64-32" = due strati da 64 e 32 neuroni; arresto
    anticipato sul 10% del training."""
    params = dict(params)
    hidden = tuple(int(n) for n in str(params.pop("hidden")).split("-"))
    return {"hidden_layer_sizes": hidden, "early_stopping": True, "max_iter": 500, "random_state": SEED,
            **params}


def plain_names(X):
    """Nomi di colonna come str di Python: i fold salvati li restituiscono come numpy.str_, che il
    ColumnTransformer di scikit-learn non riconosce come nomi."""
    return X.set_axis([str(c) for c in X.columns], axis=1)


def build(name, params=None):
    """Famiglie nuove della fase E; le altre (riusate) da zoo.build."""
    params = dict(params or {})
    if name == "extra_trees":
        return ExtraTreesClassifier(n_jobs=-1, random_state=SEED, **params)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=SEED, **params)
    if name == "svm_rbf":
        return SVC(kernel="rbf", **params)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=-1, **params)
    if name == "naive_bayes":
        return GaussianNB(**params)
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", **params)
    if name == "qda":
        return QuadraticDiscriminantAnalysis(**params)
    if name == "mlp":
        return MLPClassifier(**mlp_arguments(params))
    if name == "adaboost":
        depth = params.pop("max_depth")
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth, random_state=SEED),
                                  random_state=SEED, **params)
    if name == "spline_lr":
        # spline sulla variabile portata in quantili: nodi equispaziati sui ranghi = nodi ai quantili,
        # senza nodi doppi per le variabili con molti valori ripetuti
        spline = make_pipeline(QuantileTransformer(n_quantiles=1000),
                               SplineTransformer(n_knots=params["n_knots"], degree=3))
        prepare = make_column_transformer((spline, NUMERIC), remainder="passthrough")
        return make_pipeline(FunctionTransformer(plain_names), prepare,
                             LogisticRegression(C=params["C"], max_iter=5000))
    if name == "lr_interactions":
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             LogisticRegression(C=params["C"], max_iter=5000))
    return zoo.build(name, params)


def score(name, model, X):
    """Probabilità della classe 1, oppure funzione di decisione per i modelli senza probabilità."""
    return model.decision_function(X) if name in DECISION else model.predict_proba(X)[:, 1]


def to_logit(name, s):
    """Punteggio su scala logit (la funzione di decisione resta com'è): scala comune per Platt e
    per il meta-modello."""
    s = np.asarray(s, dtype=float)
    return s if name in DECISION else logit(np.clip(s, CLIP, 1 - CLIP))


def lgbm_regressor(params):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=SEED, subsample_freq=1, verbose=-1, **params)


# --- ottimizzazione -----------------------------------------------------------------------

def tune(data, space, fit_predict, metric, direction, n_trials=N_TRIALS):
    """Optuna come nelle fasi B e D (TPE con il seed del progetto, MedianPruner). Ogni elemento di
    data è (X_tr, y_tr, X_va, y_va); fit_predict(params, X_tr, y_tr, X_va) dà le previsioni e
    metric(y_va, previsioni) il punteggio. Conserva le previsioni interne dei tentativi completi e
    restituisce quelle del migliore: sono le previsioni out-of-fold interne, senza riaddestrare."""
    kept = {}

    def objective(trial):
        params = suggest(trial, space)
        scores, predictions = [], []
        for step, (X_tr, y_tr, X_va, y_va) in enumerate(data):
            predictions.append(np.asarray(fit_predict(params, X_tr, y_tr, X_va), dtype=float))
            if not np.isfinite(predictions[-1]).all():
                raise ValueError("previsioni non finite")
            scores.append(metric(y_va, predictions[-1]))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        kept[trial.number] = predictions
        return float(np.mean(scores))

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, catch=TRIAL_ERRORS)
    return study, kept[study.best_trial.number]


def trial_counts(study):
    states = [t.state for t in study.trials]
    return {"trials": len(states), "pruned": states.count(optuna.trial.TrialState.PRUNED),
            "failed": states.count(optuna.trial.TrialState.FAIL)}


def inner_layout(data):
    """Righe dei fold interni concatenate e fold interno di ogni riga."""
    rows = [X_va.index.to_numpy() for _, _, X_va, _ in data]
    return np.concatenate(rows), np.concatenate([np.full(len(r), j) for j, r in enumerate(rows)])


def check_inner(inner_rows, X_tr):
    assert sorted(inner_rows.tolist()) == sorted(X_tr.index.tolist()), \
        "i fold interni non coprono il training esterno"


# --- (1) zoo sul target composito ---------------------------------------------------------

def run_zoo(name, df, output=OUTPUT, n_trials=N_TRIALS, load=None):
    """Validazione annidata di una famiglia nuova: Optuna sui fold interni (PR-AUC), previsioni
    interne del tentativo migliore, riaddestramento sul training esterno. Si riprende da dove si è
    fermato."""
    load = load or loader(df, target(df).to_numpy(), "imputed")

    def fit_predict(params, X_tr, y_tr, X_va):
        return score(name, build(name, params).fit(X_tr, y_tr), X_va)

    for outer in range(OUTER):
        if record_path(name, outer, output).exists():
            continue
        start = time.perf_counter()
        data = [load(outer, j) for j in range(INNER)]
        study, inner = tune(data, SPACES[name], fit_predict, average_precision_score, "maximize", n_trials)
        inner_rows, inner_fold = inner_layout(data)
        X_tr, y_tr, X_va, _ = load(outer, None)
        check_inner(inner_rows, X_tr)
        record = {"candidate": name, "kind": "zoo", "fold": fold_name(outer), "params": study.best_params,
                  "inner_pr_auc": study.best_value, **trial_counts(study),
                  "inner_rows": inner_rows.tolist(), "inner_fold": inner_fold.tolist(),
                  "inner_score": np.concatenate(inner).tolist(), "rows": X_va.index.tolist(),
                  "score": fit_predict(study.best_params, X_tr, y_tr, X_va).tolist(),
                  "seconds": round(time.perf_counter() - start, 1)}
        save_record(name, outer, record, output)
        print(f"{name} {record['fold']}: {record['seconds']:.0f} s", flush=True)


def run_reused(name, df, output=OUTPUT, load=None):
    """Famiglia già calcolata: previsioni out-of-fold esterne e iperparametri copiati dal suo record,
    dopo aver controllato che le righe siano quelle del fold esterno. Nessun addestramento."""
    load = load or loader(df, target(df).to_numpy(), "imputed")
    for outer in range(OUTER):
        if record_path(name, outer, output).exists():
            continue
        path = source_path(name, outer)
        source = json.loads(path.read_text(encoding="utf-8"))
        _, _, X_va, _ = load(outer, None)
        assert X_va.index.tolist() == source["rows"], f"{name}: righe del fold esterno diverse dal record"
        save_record(name, outer, {"candidate": name, "kind": "reused",
                                  "source": path.relative_to(resolve(".")).as_posix(), "fold": fold_name(outer),
                                  "params": source.get("params", {}), "rows": source["rows"],
                                  "score": source["probability"]}, output)


# --- (2) ACR ed eGFR combinati, senza soglia -----------------------------------------------

def log_targets(df):
    """Bersagli continui (n, 2): log(UMAUCR) e log(SCRE)."""
    names = CONTINUOUS["targets"]
    t = np.log(df[[names["acr"], names["creatinine"]]].to_numpy(dtype=float))
    if not np.isfinite(t).all():
        raise ValueError("ACR o creatinina mancanti o non positivi: il logaritmo non è definito")
    return t


def creatinine_cut(df, value=THRESHOLDS["egfr_threshold"]):
    """log della creatinina sierica (µmol/l) a cui CKD-EPI 2021 dà eGFR = value per età e sesso:
    eGFR < value se e solo se SCRE è sopra questa creatinina. Ramo con creatinina/kappa >= 1:
    value = 142 r^-1,2 0,9938^età (1,012 se femmina), quindi r = (142 0,9938^età f / value)^(1/1,2)."""
    female = is_female(df)
    factor = 0.9938 ** df[COLUMNS["age"]].to_numpy(dtype=float) * female.map(SEX_FACTOR).to_numpy(dtype=float)
    ratio = (142 * factor / value) ** (1 / 1.2)
    if (ratio < 1).any():
        raise ValueError("soglia nel ramo creatinina/kappa < 1: formula chiusa non valida")
    return np.log(ratio * female.map(KAPPA).to_numpy(dtype=float) * CREATININE_FACTOR)


def margins(t, cut):
    """Margini KDIGO (n, 2) in larghezze di categoria, >= 0 per ACR >= 30 / eGFR <= 60."""
    units = CONTINUOUS["margin_units"]
    t = np.asarray(t, dtype=float)
    return np.column_stack([(t[:, 0] - LOG_ACR_CUT) / units["acr"], (t[:, 1] - cut) / units["creatinine"]])


def margin_target(df):
    """Bersaglio unico M = max(margine ACR, margine eGFR): positivo se e solo se il target vale 1."""
    return margins(log_targets(df), creatinine_cut(df)).max(axis=1)


def joint_probability(bound_a, bound_c, res_a, res_c):
    """1 - quota dei residui (e_a, e_c) con e_a < bound_a e e_c < bound_c, soggetto per soggetto:
    distribuzione congiunta empirica, nessuna ipotesi di normalità né di indipendenza."""
    below = (np.asarray(res_a)[None, :] < np.asarray(bound_a)[:, None]) & \
            (np.asarray(res_c)[None, :] < np.asarray(bound_c)[:, None])
    return 1 - below.mean(axis=1)


def joint_scores(mu_in, t_in, cut_in, fold_in, mu_out, cut_out, scale_in=None, scale_out=None):
    """Probabilità del target composito. Fold esterno: residui di tutte le righe interne. Righe
    interne (per stacking, ricalibrazione e soglie): residui degli altri fold interni. Con le scale
    (variante eteroschedastica) residui e distanze dalla soglia sono divisi per la scala del soggetto."""
    mu_in, t_in, mu_out = (np.asarray(a, dtype=float) for a in (mu_in, t_in, mu_out))
    scale_in = np.ones_like(mu_in) if scale_in is None else np.asarray(scale_in, dtype=float)
    scale_out = np.ones_like(mu_out) if scale_out is None else np.asarray(scale_out, dtype=float)
    z = (t_in - mu_in) / scale_in
    bound_in = (np.column_stack([np.full(len(mu_in), LOG_ACR_CUT), cut_in]) - mu_in) / scale_in
    bound_out = (np.column_stack([np.full(len(mu_out), LOG_ACR_CUT), cut_out]) - mu_out) / scale_out
    p_out = joint_probability(bound_out[:, 0], bound_out[:, 1], z[:, 0], z[:, 1])
    p_in = np.empty(len(mu_in))
    for j in np.unique(fold_in):
        here = np.asarray(fold_in) == j
        p_in[here] = joint_probability(bound_in[here, 0], bound_in[here, 1], z[~here, 0], z[~here, 1])
    return p_in, p_out


def exceedance(mu, residuals):
    """P(mu + e >= 0) con la distribuzione empirica dei residui."""
    ordered = np.sort(np.asarray(residuals, dtype=float))
    return 1 - np.searchsorted(ordered, -np.asarray(mu, dtype=float), side="left") / len(ordered)


def margin_scores(mu_in, m_in, fold_in, mu_out):
    residuals = np.asarray(m_in, dtype=float) - np.asarray(mu_in, dtype=float)
    p_in = np.empty(len(residuals))
    for j in np.unique(fold_in):
        here = np.asarray(fold_in) == j
        p_in[here] = exceedance(np.asarray(mu_in)[here], residuals[~here])
    return p_in, exceedance(mu_out, residuals)


def mse(y, prediction):
    return float(np.mean((np.asarray(y, dtype=float) - prediction) ** 2))


def relative_mse(y, prediction):
    """Errore quadratico di ogni colonna diviso per la sua varianza (1 - R^2), mediato sulle colonne."""
    y = np.asarray(y, dtype=float)
    return float(np.mean(np.mean((y - prediction) ** 2, axis=0) / y.var(axis=0)))


def lgbm_fit_predict(params, X_tr, y_tr, X_va):
    return lgbm_regressor(params).fit(X_tr, y_tr).predict(X_va)


def mlp_multitask_fit_predict(params, X_tr, t_tr, X_va):
    """Rete con due uscite e strati condivisi, bersagli standardizzati sul training."""
    mean, sd = t_tr.mean(axis=0), t_tr.std(axis=0)
    model = MLPRegressor(**mlp_arguments(params)).fit(X_tr, (t_tr - mean) / sd)
    return model.predict(X_va) * sd + mean


def run_continuous(arm, df, output=OUTPUT, n_trials=N_TRIALS, load=None):
    """Un braccio continuo: medie previste (Optuna sull'errore quadratico dei fold interni),
    residui out-of-fold interni, probabilità del target composito. Si riprende da dove si è fermato."""
    spec = ARMS[arm]
    if spec.get("from"):
        run_heteroscedastic(arm, df, output, load)
        return
    t = log_targets(df)
    cut = creatinine_cut(df)
    is_margin = spec.get("target") == "margin"
    y = margins(t, cut).max(axis=1) if is_margin else t
    load = load or loader(df, y, "imputed")
    for outer in range(OUTER):
        if record_path(arm, outer, output).exists():
            continue
        start = time.perf_counter()
        data = [load(outer, j) for j in range(INNER)]
        X_tr, y_tr, X_va, _ = load(outer, None)
        inner_rows, inner_fold = inner_layout(data)
        check_inner(inner_rows, X_tr)
        record = {"candidate": arm, "kind": "continuous", "fold": fold_name(outer), "model": spec["model"]}
        if spec["model"] == "lightgbm" and is_margin:
            study, inner = tune(data, LGBM_SPACE, lgbm_fit_predict, mse, "minimize", n_trials)
            mu_in = np.concatenate(inner)
            mu_out = lgbm_fit_predict(study.best_params, X_tr, y_tr, X_va)
            record.update(params=study.best_params, inner_mse=study.best_value, **trial_counts(study))
            p_in, p_out = margin_scores(mu_in, y[inner_rows], inner_fold, mu_out)
        else:
            if spec["model"] == "lightgbm":
                mu_in, mu_out, params, losses, counts = [], [], {}, {}, {}
                for k, part in enumerate(["acr", "creatinine"]):
                    part_data = [(a, b[:, k], c, d[:, k]) for a, b, c, d in data]
                    study, inner = tune(part_data, LGBM_SPACE, lgbm_fit_predict, mse, "minimize", n_trials)
                    mu_in.append(np.concatenate(inner))
                    mu_out.append(lgbm_fit_predict(study.best_params, X_tr, y_tr[:, k], X_va))
                    params[part], losses[part], counts[part] = study.best_params, study.best_value, trial_counts(study)
                mu_in, mu_out = np.column_stack(mu_in), np.column_stack(mu_out)
                record.update(params=params, inner_mse=losses, trials=counts)
            elif spec["model"] == "mlp_multitask":
                study, inner = tune(data, SPACES["mlp"], mlp_multitask_fit_predict, relative_mse, "minimize",
                                    n_trials)
                mu_in = np.concatenate(inner)
                mu_out = mlp_multitask_fit_predict(study.best_params, X_tr, y_tr, X_va)
                record.update(params=study.best_params, inner_relative_mse=study.best_value, **trial_counts(study))
            else:
                raise ValueError(f"{arm}: modello sconosciuto {spec['model']}")
            p_in, p_out = joint_scores(mu_in, t[inner_rows], cut[inner_rows], inner_fold, mu_out,
                                       cut[X_va.index.to_numpy()])
        record.update(inner_rows=inner_rows.tolist(), inner_fold=inner_fold.tolist(),
                      inner_mu=np.asarray(mu_in).tolist(), inner_score=p_in.tolist(),
                      rows=X_va.index.tolist(), mu=np.asarray(mu_out).tolist(), score=p_out.tolist(),
                      seconds=round(time.perf_counter() - start, 1))
        save_record(arm, outer, record, output)
        print(f"{arm} {record['fold']}: {record['seconds']:.0f} s", flush=True)


def run_heteroscedastic(arm, df, output=OUTPUT, load=None):
    """Stesse medie del braccio di partenza; scala dei residui per soggetto da un LightGBM su
    log(e^2) con parametri fissi (cross-fitting sui fold interni per le righe interne, modello su tutte
    le righe interne per il fold esterno). La costante che separa exp(E[log e^2]/2) dalla deviazione
    standard si semplifica: residui e distanze dalla soglia sono divisi per la stessa scala."""
    base = ARMS[arm]["from"]
    t = log_targets(df)
    cut = creatinine_cut(df)
    settings = CONTINUOUS["scale_model"]
    load = load or loader(df, t, "imputed")
    for outer in range(OUTER):
        if record_path(arm, outer, output).exists():
            continue
        start = time.perf_counter()
        source = read_record(base, outer, output)
        inner_rows, inner_fold = np.asarray(source["inner_rows"]), np.asarray(source["inner_fold"])
        mu_in, mu_out = np.asarray(source["inner_mu"]), np.asarray(source["mu"])
        X_tr, _, X_va, _ = load(outer, None)
        assert X_va.index.tolist() == source["rows"], "righe del fold esterno diverse dal braccio di partenza"
        X_in = X_tr.loc[inner_rows]
        residuals = t[inner_rows] - mu_in
        scale_in, scale_out = np.empty_like(residuals), np.empty_like(mu_out)
        for k in range(2):
            log_square = np.log(residuals[:, k] ** 2 + 1e-8)
            for j in np.unique(inner_fold):
                here = inner_fold == j
                model = lgbm_regressor(settings).fit(X_in[~here], log_square[~here])
                scale_in[here, k] = np.exp(0.5 * model.predict(X_in[here]))
            scale_out[:, k] = np.exp(0.5 * lgbm_regressor(settings).fit(X_in, log_square).predict(X_va))
        p_in, p_out = joint_scores(mu_in, t[inner_rows], cut[inner_rows], inner_fold, mu_out,
                                   cut[X_va.index.to_numpy()], scale_in, scale_out)
        record = {"candidate": arm, "kind": "continuous", "from": base, "fold": fold_name(outer),
                  "scale_model": settings, "inner_rows": inner_rows.tolist(), "inner_fold": inner_fold.tolist(),
                  "inner_mu": mu_in.tolist(), "inner_scale": scale_in.tolist(), "inner_score": p_in.tolist(),
                  "rows": source["rows"], "mu": mu_out.tolist(), "scale": scale_out.tolist(),
                  "score": p_out.tolist(), "seconds": round(time.perf_counter() - start, 1)}
        save_record(arm, outer, record, output)
        print(f"{arm} {record['fold']}: {record['seconds']:.0f} s", flush=True)


# --- (3) i modelli imparano fra loro ------------------------------------------------------

def pooled_scores(tag, n, output=OUTPUT):
    """Punteggi out-of-fold esterni nell'ordine delle righe di train.csv, e fold esterno di ogni riga."""
    s, fold = np.full(n, np.nan), np.full(n, -1)
    for outer in range(OUTER):
        record = read_record(tag, outer, output)
        s[record["rows"]] = record["score"]
        fold[record["rows"]] = outer
    assert not np.isnan(s).any(), f"{tag}: soggetti senza previsione"
    return s, fold


def pooled_mu(tag, n, output=OUTPUT):
    """Medie previste out-of-fold (n, 2) di un braccio continuo, nell'ordine delle righe."""
    mu = np.full((n, 2), np.nan)
    for outer in range(OUTER):
        record = read_record(tag, outer, output)
        mu[record["rows"]] = record["mu"]
    assert not np.isnan(mu).any(), f"{tag}: soggetti senza media prevista"
    return mu


def stacking_bases():
    return [*ZOO, *REUSED, STACK_CONTINUOUS]


def within_fold_percentile(s, fold):
    """Percentile di ogni punteggio dentro il suo fold esterno, ranghi medi / (n + 1), in (0, 1). Non
    usa etichette: rende confrontabili i punteggi dei 5 modelli di un fold ciascuno, che possono avere
    scale diverse (correzione del 29/09/2026, dichiarata in config)."""
    s, fold = np.asarray(s, dtype=float), np.asarray(fold)
    out = np.empty(len(s))
    for k in np.unique(fold):
        here = fold == k
        out[here] = stats.rankdata(s[here]) / (here.sum() + 1)
    return out


def base_matrix(bases, df, output=OUTPUT):
    """Colonne del meta-modello come percentili dentro il fold: per ogni modello il punteggio
    out-of-fold, per il braccio continuo anche i due margini previsti. Più il fold di ogni riga e le
    colonne che sono modelli (per la media dei ranghi)."""
    n = len(df)
    cut = creatinine_cut(df)
    names, columns, folds, models = [], [], [], []
    for name in bases:
        s, fold = pooled_scores(name, n, output)
        folds.append(fold)
        models.append(len(columns))
        names.append(name)
        columns.append(within_fold_percentile(s, fold))
        if name == STACK_CONTINUOUS:
            m = margins(pooled_mu(name, n, output), cut)
            names += [f"{name}: margine ACR", f"{name}: margine eGFR"]
            columns += [within_fold_percentile(m[:, 0], fold), within_fold_percentile(m[:, 1], fold)]
    assert all(np.array_equal(f, folds[0]) for f in folds), "fold esterni diversi fra i modelli"
    return names, np.column_stack(columns), folds[0], models


def run_combined(df, output=OUTPUT, bases=None):
    """Stacking e media dei ranghi con cross-fitting sui fold esterni. Stacking: per il fold k il
    meta-modello (logistica L2 sui logit dei percentili, C scelto per PR-AUC in una CV a 5 fold) si
    stima sulle righe e sulle etichette degli altri fold e si applica al fold k. Media dei ranghi: media
    dei percentili dei modelli dentro il fold, nessun parametro stimato sulle etichette."""
    bases = bases or stacking_bases()
    y = target(df).to_numpy()
    spec = STACKING["meta"]
    names, P, fold, models = base_matrix(bases, df, output)
    Z = logit(P)
    for outer in range(OUTER):
        train, test = fold != outer, fold == outer
        rows = np.flatnonzero(test)
        scaler = StandardScaler().fit(Z[train])
        cv = StratifiedKFold(spec["cv"], shuffle=True, random_state=SEED)
        meta = LogisticRegressionCV(Cs=spec["Cs"], cv=cv, scoring=spec["scoring"], max_iter=5000)
        meta.fit(scaler.transform(Z[train]), y[train])
        C = float(np.ravel(meta.C_)[0])
        save_record("stacking", outer, {
            "candidate": "stacking", "kind": "combined", "fold": fold_name(outer), "bases": names, "C": C,
            "coefficients": dict(zip(names, np.ravel(meta.coef_).tolist())),
            "intercept": float(np.ravel(meta.intercept_)[0]), "rows": rows.tolist(),
            "score": meta.predict_proba(scaler.transform(Z[test]))[:, 1].tolist()}, output)
        save_record("rank_mean", outer, {
            "candidate": "rank_mean", "kind": "combined", "fold": fold_name(outer), "bases": list(bases),
            "rows": rows.tolist(), "score": P[test][:, models].mean(axis=1).tolist()}, output)
        print(f"stacking e media dei ranghi {fold_name(outer)}: C = {C:.4g}", flush=True)


# --- (4) valutazione ----------------------------------------------------------------------

def cross_fitted(tag, s, fold, y, rules):
    """Ricalibrazione di Platt e soglie del fold k stimate sugli altri fold esterni e applicate al
    fold k: (q, allerte per regola, soglie). Nessuna etichetta del fold k entra nella stima. Il
    punteggio entra come logit del suo percentile dentro il fold, così la stessa ricalibrazione vale
    per i modelli dei diversi fold anche se le loro scale sono diverse."""
    x = logit(within_fold_percentile(s, fold))[:, None]
    q = np.empty(len(s))
    flags = {rule: np.zeros(len(s), bool) for rule in rules}
    cuts = []
    for outer in range(OUTER):
        here = fold == outer
        model = LogisticRegression(C=np.inf, max_iter=5000).fit(x[~here], y[~here])
        q[here] = model.predict_proba(x[here])[:, 1]
        rest = model.predict_proba(x[~here])[:, 1]
        for rule, threshold in rules.items():
            cut = threshold(y[~here], rest)
            flags[rule][here] = q[here] >= cut
            cuts.append({"model": tag, "fold": outer, "rule": rule, "threshold": float(cut)})
    return q, flags, cuts


def classification(y, flagged):
    """Misure con soglia da un vettore di allerta (positivo = allerta)."""
    y, flagged = np.asarray(y) == 1, np.asarray(flagged, dtype=bool)
    tp, fp = int(np.sum(flagged & y)), int(np.sum(flagged & ~y))
    fn, tn = int(np.sum(~flagged & y)), int(np.sum(~flagged & ~y))
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": sensitivity, "specificity": specificity,
            "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn), "mcc": (tp * tn - fp * fn) / denominator if denominator else np.nan,
            "balanced_accuracy": (sensitivity + specificity) / 2, "alert_rate": (tp + fp) / len(y)}


SEVERE = [LEVELS.index(level) for level in EVALUATION["severe_levels"]]
VERY_HIGH = LEVELS.index("molto alto")


def proportion(name, k, total):
    """Proporzione con IC di Wilson, in tre colonne."""
    value, low, high = ev.wilson(int(k), int(total))
    return {name: value, f"{name}_low": low, f"{name}_high": high}


def operating_row(y, flagged, codes):
    """Punto operativo, con IC di Wilson per le proporzioni: sensibilità (complessiva, sui casi gravi e
    per livello KDIGO), specificità, VPP, VPN, accuratezza, quota di allerta; rapporti di verosimiglianza
    e rapporto di odds diagnostico; F1, F2 (pesa due volte la sensibilità), MCC, kappa di Cohen,
    accuratezza bilanciata; esami evitati ogni 100 soggetti, casi persi ogni 1000, esami per caso
    trovato. Il tasso di non informazione (accuratezza di chi dice sempre "negativo") è 1 - prevalenza."""
    base = classification(y, flagged)
    tp, fp, fn, tn = base["tp"], base["fp"], base["fn"], base["tn"]
    flagged, codes = np.asarray(flagged, dtype=bool), np.asarray(codes)
    n = len(flagged)
    counts = {"sensitivity": (tp, tp + fn), "specificity": (tn, tn + fp), "ppv": (tp, tp + fp),
              "npv": (tn, tn + fn), "accuracy": (tp + tn, n), "alert_rate": (tp + fp, n),
              "severe_sensitivity": (np.sum(flagged & np.isin(codes, SEVERE)), np.isin(codes, SEVERE).sum())}
    for i, level in enumerate(LEVELS[1:], start=1):
        counts[f"sensitivity_{level.replace(' ', '_')}"] = (np.sum(flagged & (codes == i)), np.sum(codes == i))
    row = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    for name, (k, total) in counts.items():
        row.update(proportion(name, k, total))
    sensitivity, specificity = base["sensitivity"], base["specificity"]
    lr_positive = sensitivity / (1 - specificity) if specificity < 1 else np.inf
    lr_negative = (1 - sensitivity) / specificity if specificity > 0 else np.nan
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    row.update(no_information_rate=(fp + tn) / n, lr_positive=lr_positive, lr_negative=lr_negative,
               diagnostic_odds_ratio=lr_positive / lr_negative if lr_negative else np.inf,
               f1=base["f1"], f2=5 * tp / (5 * tp + 4 * fn + fp), mcc=base["mcc"],
               kappa=((tp + tn) / n - expected) / (1 - expected), balanced_accuracy=base["balanced_accuracy"],
               youden_j=sensitivity + specificity - 1, tests_avoided_per_100=100 * (tn + fn) / n,
               missed_per_1000=1000 * fn / n, tests_per_case=(tp + fp) / tp if tp else np.nan)
    return row


def top_rows(y, q, codes, fractions=EVALUATION["top_fractions"]):
    """Il f% dei soggetti con probabilità più alta: precisione, lift = precisione / prevalenza, quota
    dei positivi e dei casi gravi presi. La classifica usa solo le probabilità, mai le etichette."""
    y, codes = np.asarray(y) == 1, np.asarray(codes)
    order = np.argsort(-np.asarray(q, dtype=float), kind="stable")
    severe = np.isin(codes, SEVERE)
    prevalence = y.mean()
    rows = []
    for fraction in fractions:
        top = np.zeros(len(y), bool)
        top[order[:int(round(fraction * len(y)))]] = True
        row = {"fraction": fraction, "n_top": int(top.sum())}
        row.update(proportion("precision", np.sum(top & y), top.sum()))
        row.update({f"lift{s}": row[f"precision{s}"] / prevalence for s in ("", "_low", "_high")})
        row.update(proportion("recall", np.sum(top & y), y.sum()))
        row.update(proportion("severe_recall", np.sum(top & severe), severe.sum()))
        rows.append(row)
    return rows


def net_benefit(y, q, t):
    """Beneficio netto del modello alla soglia di probabilità t e di "testare tutti" (Vickers & Elkin
    2006): veri positivi / n - falsi positivi / n * t / (1 - t)."""
    y = np.asarray(y) == 1
    flagged = np.asarray(q) >= t
    odds = t / (1 - t)
    model = np.sum(flagged & y) / len(y) - np.sum(flagged & ~y) / len(y) * odds
    return model, y.mean() - (1 - y.mean()) * odds


def net_benefit_rows(y, q, thresholds=EVALUATION["net_benefit_thresholds"], n_boot=EVALUATION["bootstrap"],
                     seed=SEED):
    """Net benefit ed esami evitati ogni 100 rispetto a testare tutti, (NB - NB tutti) / (t / (1 - t))
    x 100, con IC da bootstrap semplice (percentile)."""
    y, q = np.asarray(y), np.asarray(q, dtype=float)
    rng = np.random.default_rng(seed)
    samples = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    rows = []
    for t in thresholds:
        odds = t / (1 - t)
        model, everyone = net_benefit(y, q, t)
        boot = np.array([net_benefit(y[idx], q[idx], t) for idx in samples])
        low, high = ev.percentile_interval(boot[:, 0])
        avoided_low, avoided_high = ev.percentile_interval(100 * (boot[:, 0] - boot[:, 1]) / odds)
        rows.append({"threshold": t, "net_benefit": model, "net_benefit_low": low, "net_benefit_high": high,
                     "net_benefit_all": everyone, "avoided_per_100": 100 * (model - everyone) / odds,
                     "avoided_per_100_low": avoided_low, "avoided_per_100_high": avoided_high})
    return rows


def expected_calibration_error(y, q, bins=10):
    """Media, pesata per numerosità, di |probabilità media - proporzione osservata| nei decili di q."""
    y, q = np.asarray(y, dtype=float), np.asarray(q, dtype=float)
    labels = pd.qcut(q, bins, labels=False, duplicates="drop")
    return float(sum(np.sum(labels == b) / len(q) * abs(q[labels == b].mean() - y[labels == b].mean())
                     for b in np.unique(labels)))


def partial_auc_high_sensitivity(y, q, min_sensitivity=0.80):
    """AUROC parziale standardizzata (McClish) nella zona con sensibilità >= min_sensitivity:
    classi e punteggi scambiati, così la zona diventa FPR <= 1 - min_sensitivity."""
    return roc_auc_score(1 - np.asarray(y), -np.asarray(q, dtype=float), max_fpr=1 - min_sensitivity)


def subgroup_masks(df):
    """Sottogruppi descrittivi fissati in config prima dei risultati."""
    age = df[COLUMNS["age"]].to_numpy(dtype=float)
    diabetes = df[COLUMNS["diabetes"]].to_numpy() == 1
    female = is_female(df).to_numpy()
    hypertension = df["HypertenHis"].fillna(0).to_numpy() == 1
    return {"diabetici": diabetes, "non diabetici": ~diabetes, "donne": female, "uomini": ~female,
            "età < 40": age < 40, "età 40-59": (age >= 40) & (age < 60), "età >= 60": age >= 60,
            "ipertensione nota": hypertension, "ipertensione non nota": ~hypertension}


def subgroup_rows(y, q, flagged, codes, masks):
    """Per sottogruppo: numerosità, prevalenza, AUROC (DeLong), PR-AUC e suo lift, e il punto a
    sensibilità 0,90 con la soglia globale (dagli altri fold)."""
    rows = []
    for name, mask in masks.items():
        yy, qq = y[mask], q[mask]
        row = {"subgroup": name, "n": int(mask.sum()), "positives": int(yy.sum()), "prevalence": yy.mean()}
        if 2 <= yy.sum() <= len(yy) - 2:
            row["auc"], row["auc_low"], row["auc_high"] = ev.delong(yy, qq)
            row["pr_auc"] = average_precision_score(yy, qq)
            row["pr_auc_lift"] = row["pr_auc"] / yy.mean()
        point = operating_row(yy, flagged[mask], codes[mask])
        row.update({k: point[k] for k in ("sensitivity", "specificity", "ppv", "npv", "alert_rate",
                                           "severe_sensitivity")})
        row["severe_cases"] = int(np.isin(codes[mask], SEVERE).sum())
        rows.append(row)
    return rows


def role(tag):
    if tag in REFERENCES:
        return "riferimento (fase A)"
    if tag in REUSED:
        return "già calcolato (fase D)"
    if tag in ZOO:
        return "zoo nuovo"
    if tag in ARMS:
        return "ACR + eGFR continui"
    return "combinazione di modelli"


def sensitivity_rule(target_sensitivity):
    return lambda y, q: ev.threshold_at_sensitivity(y, q, target_sensitivity)


# punti operativi: sensibilità fissata e Youden; le soglie vengono sempre dagli altri fold esterni
RULES = {**{f"sens{round(s * 100)}": sensitivity_rule(s) for s in EVALUATION["sensitivity_grid"]},
         "youden": ev.threshold_youden}


def evaluate(df, tags, output=OUTPUT):
    """Pannello di metriche sulle previsioni out-of-fold esterne (probabilità ricalibrate con
    cross-fitting), confronti con il riferimento della Fase A con AUROC media sui fold più alta (t
    corretto di Nadeau-Bengio, Holm fra i candidati della fase E, separatamente per AUROC e PR-AUC) e
    qualità dei regressori. Tabelle: metrics (una riga per modello), operating_points, top,
    net_benefit, subgroups, folds, thresholds, comparison, regression."""
    y = target(df).to_numpy()
    n = len(y)
    codes = np.asarray(pd.Categorical(kdigo_level(df), LEVELS, ordered=True).codes)
    albuminuria, low_egfr = components(df)
    negatives = y == 0
    acr = df[COLUMNS["acr"]].to_numpy(dtype=float)
    masks = {"auc_egfr": negatives | (low_egfr == 1),
             "auc_albuminuria_only": negatives | ((albuminuria == 1) & (low_egfr == 0)),
             "auc_acr_300": negatives | (acr >= 300),
             "auc_severe": negatives | np.isin(codes, SEVERE),
             **{f"auc_{level.replace(' ', '_')}": (codes == 0) | (codes == i) for i, level in enumerate(LEVELS)
                if i > 0}}
    prevalence = y.mean()
    groups = subgroup_masks(df)
    continuous = {"spearman_log_acr": np.log(acr), "spearman_egfr": egfr(df).to_numpy()}
    rows, fold_rows, cut_rows, operating, top, benefit, subgroups = [], [], [], [], [], [], []
    for tag in tags:
        s, fold = pooled_scores(tag, n, output)
        q, flags, cuts = cross_fitted(tag, s, fold, y, RULES)
        cut_rows += cuts
        folds = pd.DataFrame([{"model": tag, "fold": k, "auc": roc_auc_score(y[fold == k], s[fold == k]),
                               "pr_auc": average_precision_score(y[fold == k], s[fold == k])}
                              for k in range(OUTER)])
        fold_rows.append(folds)
        key = {"model": tag, "role": role(tag)}
        points = {rule: operating_row(y, flags[rule], codes) for rule in RULES}
        operating += [{**key, "rule": rule, **point} for rule, point in points.items()]
        ranking = top_rows(y, q, codes)
        top += [{**key, **r} for r in ranking]
        benefit += [{**key, **r} for r in net_benefit_rows(y, q)]
        subgroups += [{**key, **r} for r in subgroup_rows(y, q, flags["sens90"], codes, groups)]
        auc, auc_low, auc_high = ev.delong(y, q)
        ap, ap_low, ap_high = ev.pr_auc(y, q)
        cal = calibration(y, q)
        fpr, tpr, _ = roc_curve(y, q)
        top10 = next(r for r in ranking if np.isclose(r["fraction"], 0.10))
        rows.append({**key, "auc": auc, "auc_low": auc_low, "auc_high": auc_high,
                     "auc_folds_mean": folds["auc"].mean(), "auc_raw_pooled": roc_auc_score(y, s),
                     "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr)),
                     "pauc_sens80": partial_auc_high_sensitivity(y, q),
                     "pr_auc": ap, "pr_auc_low": ap_low, "pr_auc_high": ap_high,
                     "pr_auc_folds_mean": folds["pr_auc"].mean(), "prevalence": prevalence,
                     "pr_auc_lift": ap / prevalence, "brier": cal["brier"],
                     "brier_scaled": 1 - cal["brier"] / (prevalence * (1 - prevalence)),
                     "log_loss": log_loss(y, np.clip(q, CLIP, 1 - CLIP)),
                     "calibration_intercept": cal["intercept"], "calibration_slope": cal["slope"],
                     "oe_ratio": y.sum() / q.sum(), "ece": expected_calibration_error(y, q),
                     **{name: roc_auc_score(y[mask] if name in ("auc_egfr", "auc_albuminuria_only",
                                                                "auc_acr_300", "auc_severe")
                                            else (codes[mask] > 0).astype(int), q[mask])
                        for name, mask in masks.items()},
                     "kdigo_concordance": ev.jonckheere(q, codes)["concordance"],
                     **{name: stats.spearmanr(q, value).statistic for name, value in continuous.items()},
                     **{f"sens90_{k}": points["sens90"][k] for k in
                        ("sensitivity", "specificity", "ppv", "npv", "severe_sensitivity",
                         "sensitivity_molto_alto", "alert_rate", "lr_positive", "lr_negative",
                         "diagnostic_odds_ratio", "f2", "tests_avoided_per_100", "missed_per_1000")},
                     "top10_lift": top10["lift"], "top10_recall": top10["recall"],
                     "top10_severe_recall": top10["severe_recall"]})
    metrics = pd.DataFrame(rows)
    folds = pd.concat(fold_rows, ignore_index=True)
    comparison = compare(metrics, folds, n)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"metrics": metrics, "operating_points": pd.DataFrame(operating), "top": pd.DataFrame(top),
              "net_benefit": pd.DataFrame(benefit), "subgroups": pd.DataFrame(subgroups), "folds": folds,
              "thresholds": pd.DataFrame(cut_rows), "comparison": comparison,
              "regression": regression_quality(df, [t for t in tags if t in ARMS], output)}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    return tables


def compare(metrics, folds, n):
    """Candidati della fase E contro il riferimento della Fase A con AUROC media sui fold più alta."""
    references = metrics[metrics["model"].isin(REFERENCES)]
    best = references.loc[references["auc_folds_mean"].idxmax(), "model"]
    candidates = [m for m in metrics["model"] if m in ZOO or m in ARMS or m in COMBINED]
    n_test = n / OUTER
    rows = []
    for metric in ("auc", "pr_auc"):
        wide = folds.pivot(index="fold", columns="model", values=metric)
        block = pd.DataFrame([{"metric": metric, "candidate": c, "reference": best,
                               **ev.corrected_ttest(wide[c] - wide[best], n - n_test, n_test)}
                              for c in candidates])
        if len(block):
            block["p_holm"] = ev.holm(block["p_value"])
            block["improves"] = ((block["difference"] >= EVALUATION["min_difference"]) & (block["low"] > 0)
                                 & (block["p_holm"] < 0.05))
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def regression_quality(df, arms, output=OUTPUT):
    """R^2 e Spearman delle medie previste sul fold esterno, per bersaglio continuo."""
    t = log_targets(df)
    m = margins(t, creatinine_cut(df)).max(axis=1)
    rows = []
    for arm in arms:
        records = [read_record(arm, outer, output) for outer in range(OUTER)]
        mu = np.concatenate([np.asarray(r["mu"]) for r in records])
        index = np.concatenate([r["rows"] for r in records])
        if mu.ndim == 1:
            pairs = {"margine KDIGO": (m[index], mu)}
        else:
            pairs = {"log ACR": (t[index, 0], mu[:, 0]), "log creatinina": (t[index, 1], mu[:, 1])}
        for name, (truth, prediction) in pairs.items():
            rows.append({"model": arm, "target": name,
                         "r2": 1 - np.sum((truth - prediction) ** 2) / np.sum((truth - truth.mean()) ** 2),
                         "spearman": stats.spearmanr(truth, prediction).statistic})
    return pd.DataFrame(rows)


def all_tags():
    return [*REFERENCES, *[r for r in REUSED if r not in REFERENCES], *ZOO, *ARMS, *COMBINED]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zoo", nargs="*", choices=ZOO, help="famiglie nuove (senza nomi: tutte)")
    parser.add_argument("--reused", nargs="*", choices=list(REUSED), help="famiglie riusate (senza nomi: tutte)")
    parser.add_argument("--continuous", nargs="*", choices=list(ARMS), help="bracci continui (senza nomi: tutti)")
    parser.add_argument("--combined", action="store_true", help="stacking e media dei ranghi")
    parser.add_argument("--evaluate", action="store_true", help="pannello di metriche e confronti")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="Variables are collinear")
    train = pd.read_csv(PROCESSED / "train.csv")
    for name in (ZOO if args.zoo == [] else args.zoo or []):
        run_zoo(name, train)
    for name in (list(REUSED) if args.reused == [] else args.reused or []):
        run_reused(name, train)
    for arm in (list(ARMS) if args.continuous == [] else args.continuous or []):
        run_continuous(arm, train)
    if args.combined:
        run_combined(train)
    if args.evaluate:
        tables = evaluate(train, [t for t in all_tags() if done(t)])
        columns = ["model", "auc", "pr_auc", "pr_auc_lift", "auc_severe", "auc_egfr", "sens90_severe_sensitivity",
                   "sens90_npv", "top10_lift"]
        print(tables["metrics"][columns].round(3).to_string(index=False))
        print(tables["comparison"].round(4).to_string(index=False))
        print(tables["regression"].round(3).to_string(index=False))
