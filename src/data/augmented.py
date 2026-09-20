"""Tecniche di bilanciamento della Fase B, applicate solo alla parte di training di ogni fold.

- pesi (class_weight, level_weight): nessun dato nuovo, pesi per esempio passati a fit
- campionamento (undersampling, oversampling, smote_nc, smote_nc_level, ctgan, ctgan_level): il
  training di ogni fold viene bilanciato una volta sola e salvato in
  data/augmented/<tecnica>/<set>/<fold>.npz, così tutti i modelli vedono gli stessi dati
La validazione resta sempre reale; il livello KDIGO serve solo per pesi e generazione e non entra
mai fra le variabili. Protocollo: Notepad, "Fase B — protocollo fissato prima dei risultati".
"""
import argparse
import hashlib
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.config import CONFIG, resolve
from src.data.imputed import all_folds, fold_name, load_fold
from src.data.kidney import LEVELS, kdigo_level, target
from src.data.preprocess import feature_types
from src.data.split import PROCESSED

SETTINGS = CONFIG["phase_b"]
TECHNIQUES = SETTINGS["techniques"]
OUTPUT = resolve(CONFIG["data"]["augmented"])
LOG = resolve(SETTINGS["output"]) / "augmentation"
RATIO = SETTINGS["ratio"]
SEED = CONFIG["seed"]
SYNTHETIC = -1  # valore di "source" per le righe generate


# --- pesi ---------------------------------------------------------------------------------

def class_weights(y):
    """Pesi inversamente proporzionali alla prevalenza: le due classi pesano uguale, media 1."""
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2)
    return len(y) / (2 * counts[y])


def level_weights(y, codes, costs):
    """Pesi per singolo esempio (Zadrozny et al. 2003): fra i positivi proporzionali al costo del
    livello KDIGO, poi riscalati perché positivi e negativi pesino uguale (media 1)."""
    y, codes = np.asarray(y), np.asarray(codes)
    cost = np.array([costs.get(level, 1.0) for level in LEVELS], dtype=float)
    w = np.where(y == 1, cost[codes], 1.0)
    for label in (0, 1):
        mask = y == label
        w[mask] *= (len(y) / 2) / w[mask].sum()
    return w


# --- campionamento ------------------------------------------------------------------------

def random_under(X, y, seed):
    sampler = RandomUnderSampler(sampling_strategy=RATIO, random_state=seed)
    X_bal, y_bal = sampler.fit_resample(X, y)
    return X_bal, y_bal, X.index.to_numpy()[sampler.sample_indices_]


def random_over(X, y, seed):
    sampler = RandomOverSampler(sampling_strategy=RATIO, random_state=seed)
    X_bal, y_bal = sampler.fit_resample(X, y)
    return X_bal, y_bal, X.index.to_numpy()[sampler.sample_indices_]


def with_synthetic(X, y, X_new, n_new):
    """Righe reali seguite da n_new righe sintetiche positive."""
    X_bal = pd.concat([X.reset_index(drop=True), X_new.reset_index(drop=True)], ignore_index=True)
    y_bal = np.r_[np.asarray(y), np.ones(n_new, dtype=int)]
    source = np.r_[X.index.to_numpy(), np.full(n_new, SYNTHETIC)]
    return X_bal, y_bal, source


def categorical_indices(columns):
    _, categorical = feature_types(columns)
    return [list(columns).index(c) for c in categorical]


def smote_nc(X, y, seed, k=5):
    """SMOTE-NC (Chawla et al. 2002): le categoriche prendono il valore più frequente fra i vicini."""
    sampler = SMOTENC(categorical_features=categorical_indices(X.columns), k_neighbors=k,
                      sampling_strategy=RATIO, random_state=seed)
    X_res, _ = sampler.fit_resample(X, y)
    n_new = len(X_res) - len(X)
    return with_synthetic(X, y, X_res.iloc[len(X):], n_new)


def level_quotas(counts, total):
    """Righe da generare per livello, proporzionali alle numerosità (metodo del resto più grande)."""
    assert total >= 0, f"quote negative: {total}"
    counts = np.asarray(counts, dtype=float)
    exact = total * counts / counts.sum()
    quotas = np.floor(exact).astype(int)
    quotas[np.argsort(-(exact - quotas), kind="stable")[: total - quotas.sum()]] += 1
    return quotas


def smote_nc_level(X, y, codes, seed, k=5):
    """SMOTE-NC dentro ciascun livello positivo: i vicini sono cercati solo fra i positivi dello
    stesso livello, e le proporzioni dei livelli restano quelle osservate."""
    y, codes = np.asarray(y), np.asarray(codes)
    categorical = categorical_indices(X.columns)
    levels = [i for i in range(1, len(LEVELS)) if np.any((y == 1) & (codes == i))]
    counts = [int(np.sum((y == 1) & (codes == i))) for i in levels]
    quotas = level_quotas(counts, int(RATIO * np.sum(y == 0)) - int(y.sum()))
    new = []
    for level, n, q in zip(levels, counts, quotas):
        if q == 0:
            continue
        if n == 1:  # un solo caso: nessun vicino, si duplica
            new.append(X[(y == 1) & (codes == level)].iloc[[0] * q])
            continue
        rows = (y == 0) | (codes == level)
        sampler = SMOTENC(categorical_features=categorical, k_neighbors=min(k, n - 1),
                          sampling_strategy={1: n + q}, random_state=seed + level)
        X_res, _ = sampler.fit_resample(X[rows], y[rows])
        new.append(X_res.iloc[rows.sum():])
    X_new = pd.concat(new) if new else X.iloc[:0]
    return with_synthetic(X, y, X_new, len(X_new))


def ctgan_sample(X, y, seed, spec, codes=None):
    """CTGAN (Xu et al. 2019) addestrato sull'intero training del fold, con la classe (o il livello
    KDIGO) come colonna discreta di condizionamento; si tengono solo i campioni positivi della
    categoria richiesta, fino a 1:1 (il condizionamento di CTGAN non è rigido)."""
    import torch
    from ctgan import CTGAN

    torch.manual_seed(seed)
    y = np.asarray(y)
    data = X.reset_index(drop=True).copy()
    _, categorical = feature_types(X.columns)
    missing = int(RATIO * np.sum(y == 0)) - int(y.sum())
    if codes is None:
        data["condition"] = y
        wanted = {1: missing}
    else:
        codes = np.asarray(codes)
        data["condition"] = codes
        levels = [i for i in range(1, len(LEVELS)) if np.any(codes == i)]
        wanted = dict(zip(levels, level_quotas([np.sum(codes == i) for i in levels], missing)))
    model = CTGAN(epochs=spec["epochs"], batch_size=spec["batch_size"], enable_gpu=False, verbose=False)
    model.set_random_state(seed)
    model.fit(data, discrete_columns=categorical + ["condition"])
    new = []
    for value, n in wanted.items():
        # il condizionamento di CTGAN è "morbido": si estrae a lotti e si scartano i campioni della
        # categoria sbagliata (con i livelli rari l'accettazione è bassa, va registrata)
        kept, drawn = [], 0
        batch = max(10 * int(n), 2000)
        for _ in range(spec.get("max_rounds", 200)):
            sample = model.sample(batch, condition_column="condition", condition_value=value)
            drawn += batch
            kept.append(sample[sample["condition"] == value])
            if sum(len(part) for part in kept) >= n:
                break
        kept = pd.concat(kept)
        print(f"  condizione {value}: {len(kept)} campioni validi su {drawn} estratti "
              f"({100 * len(kept) / drawn:.1f}%), ne servono {n}", flush=True)
        if len(kept) < n:
            raise RuntimeError(f"CTGAN: {len(kept)} campioni validi su {n} per la condizione {value}")
        new.append(kept.iloc[:n].drop(columns="condition"))
    X_new = pd.concat(new)[list(X.columns)].astype(X.dtypes.to_dict())
    return with_synthetic(X, y, X_new, len(X_new))


def balance(technique, X, y, codes, seed):
    """(X_bil, y_bil, source): source = riga di train.csv, -1 per le righe sintetiche."""
    spec = TECHNIQUES[technique]
    if technique == "undersampling":
        return random_under(X, y, seed)
    if technique == "oversampling":
        return random_over(X, y, seed)
    if technique == "smote_nc":
        return smote_nc(X, y, seed, spec["k_neighbors"])
    if technique == "smote_nc_level":
        return smote_nc_level(X, y, codes, seed, spec["k_neighbors"])
    if technique == "ctgan":
        return ctgan_sample(X, y, seed, spec)
    if technique == "ctgan_level":
        return ctgan_sample(X, y, seed, spec, codes)
    raise ValueError(f"tecnica di campionamento sconosciuta: {technique}")


# --- cache --------------------------------------------------------------------------------

def fold_seed(outer, inner):
    """Seed diverso per ogni fold, riproducibile."""
    return SEED + 100 * (5 if outer is None else outer) + (0 if inner is None else inner + 1)


def cache_path(technique, feature_set, outer=None, inner=None, directory=OUTPUT):
    return directory / technique / feature_set / f"{fold_name(outer, inner)}.npz"


def labels(df):
    """Target e codice del livello KDIGO (0 = basso ... 3 = molto alto) per riga di df."""
    return target(df).to_numpy(), np.asarray(kdigo_level(df).cat.codes)


def imputed_kwargs(imputed):
    """Cartella e metodo dei fold imputati (i test usano una cache temporanea)."""
    return {} if imputed is None else {"directory": imputed[0], "method": imputed[1]}


def build_cache(df, technique, feature_set="main", directory=OUTPUT, imputed=None, log=LOG,
                outer_only=None):
    """Bilancia e salva i fold che mancano (si può interrompere e riprendere). Le tecniche
    esplorative servono solo per i fold esterni e il training intero."""
    outer_only = TECHNIQUES[technique].get("exploratory", False) if outer_only is None else outer_only
    y_all, codes_all = labels(df)
    rows = []
    for outer, inner, _, _ in all_folds(df):
        path = cache_path(technique, feature_set, outer, inner, directory)
        if path.exists() or (outer_only and inner is not None):
            continue
        start = time.perf_counter()
        X, _ = load_fold(feature_set, outer, inner, **imputed_kwargs(imputed))
        index = X.index.to_numpy()
        X_bal, y_bal, source = balance(technique, X, y_all[index], codes_all[index], fold_seed(outer, inner))
        values = X_bal.to_numpy(dtype=float)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X_train=values, y_train=np.asarray(y_bal, dtype=int), source=source,
                            columns=np.array(X.columns, dtype=str), technique=np.array(technique),
                            seed=np.array(fold_seed(outer, inner)))
        real = source != SYNTHETIC
        row = {"fold": fold_name(outer, inner), "rows": len(y_bal), "negatives": int(np.sum(y_bal == 0)),
               "positives": int(np.sum(y_bal == 1)), "synthetic": int(np.sum(~real)),
               "seconds": round(time.perf_counter() - start, 1),
               "sha256": hashlib.sha256(values.tobytes()).hexdigest()[:16]}
        for i, level in enumerate(LEVELS[1:], start=1):
            row[f"real_{level}"] = int(np.sum(codes_all[source[real]] == i))
        rows.append(row)
        print(f"{technique} {feature_set} {row['fold']}: {row['rows']} righe, {row['synthetic']} sintetiche, "
              f"{row['seconds']:.0f} s", flush=True)
    if rows and log is not None:
        log.mkdir(parents=True, exist_ok=True)
        file = log / f"{technique}.csv"
        previous = pd.read_csv(file) if file.exists() else None
        pd.concat([previous, pd.DataFrame(rows)]).to_csv(file, index=False)


def load_balanced(technique, feature_set, outer=None, inner=None, directory=OUTPUT):
    """(X_train, y_train) bilanciati; l'indice di X è la riga di train.csv (-1 se sintetica)."""
    with np.load(cache_path(technique, feature_set, outer, inner, directory), allow_pickle=False) as data:
        X = pd.DataFrame(data["X_train"], columns=list(data["columns"]), index=data["source"])
        y = data["y_train"]
    return X, y


def training_set(technique, feature_set, y_all, codes_all, outer=None, inner=None, directory=OUTPUT,
                 imputed=None):
    """(X_tr, y_tr, pesi) della parte di training di un fold per una tecnica ("none" = Fase A)."""
    if technique == "none" or TECHNIQUES[technique]["kind"] == "weight":
        X, _ = load_fold(feature_set, outer, inner, **imputed_kwargs(imputed))
        index = X.index.to_numpy()
        y = np.asarray(y_all)[index]
        if technique == "none":
            return X, y, None
        if technique == "class_weight":
            return X, y, class_weights(y)
        return X, y, level_weights(y, np.asarray(codes_all)[index], TECHNIQUES[technique]["costs"])
    X, y = load_balanced(technique, feature_set, outer, inner, directory)
    return X, y, None


def run(techniques, feature_set="main"):
    df = pd.read_csv(PROCESSED / "train.csv")
    for technique in techniques:
        build_cache(df, technique, feature_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sampling = [t for t, s in TECHNIQUES.items() if s["kind"] == "sample"]
    parser.add_argument("--techniques", nargs="+", choices=sampling,
                        default=[t for t in sampling if not TECHNIQUES[t].get("exploratory")])
    run(parser.parse_args().techniques)
