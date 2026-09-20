"""Fase C: sottogruppo diabetico (domanda 6) sulle previsioni out-of-fold della Fase B.

Nessun riaddestramento: si filtrano i soggetti diabetici nelle previsioni gia' salvate.
- bracci: "nessuna correzione" (analisi principale) e pesi per livello KDIGO (descrittivo)
- soglia e fasce: quelle globali di `analytics/phase_b/evaluation/cutpoints.csv`, mai ristimate sul
  sottogruppo (Van Calster et al. 2025, Box 1: la soglia e' un argomento medico, non statistico)
- confronto fra sottogruppi: solo differenza di AUROC fra gruppi disgiunti. La differenza di PR-AUC
  non viene calcolata perche' dipende dalla prevalenza e favorisce il gruppo con piu' positivi
  (Matos et al. 2026; McDermott et al. 2024): fra i diabetici la prevalenza e' 25,9% contro 8,7%
- nessun test formale: con 68 positivi e 16 casi gravi la potenza e' nulla, quindi solo stime con
  intervalli (Riley et al. 2024: servono almeno 100 eventi; TRIPOD+AI item 23a)
Legge solo train.csv e le previsioni out-of-fold: il test set non viene letto.
Protocollo: Notepad, "Fase C - protocollo fissato prima dei risultati".
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

import src.models.evaluation as ev
from src.config import CONFIG, resolve
from src.data.kidney import LEVELS, kdigo_level, target
from src.data.split import PROCESSED
from src.models.evaluation_b import pooled

SETTINGS = CONFIG["phase_c"]
OUTPUT = resolve(SETTINGS["evaluation"]["output"])
TECHNIQUES = SETTINGS["techniques"]
# braccio principale e unico braccio di confronto, descrittivo (protocollo, punto 1)
PRIMARY = SETTINGS["primary"]
if len(TECHNIQUES) != 2 or PRIMARY not in TECHNIQUES:
    raise ValueError(f"phase_c: attesi due bracci con primary fra loro, trovati {TECHNIQUES} "
                     f"e primary={PRIMARY}")
SECONDARY = next(t for t in TECHNIQUES if t != PRIMARY)
SUBGROUP = SETTINGS["subgroup"]
PREDICTIONS = resolve(CONFIG["phase_b"]["output"]) / "oof_predictions.csv"
CUTPOINTS = resolve(CONFIG["phase_b"]["evaluation"]["output"]) / "cutpoints.csv"
DIABETES = CONFIG["columns"]["diabetes"]
SEVERE = [LEVELS.index(level) for level in CONFIG["phase_b"]["evaluation"]["severe_levels"]]
SEVERE_LABEL = "gravi (alto + molto alto)"
TARGET = CONFIG["evaluation"]["target_sensitivity"]
ALPHA = CONFIG["evaluation"]["alpha"]
N_BOOT = CONFIG["evaluation"]["bootstrap"]
KAPPA_WEIGHTS = CONFIG["evaluation"]["kappa_weights"]
SEED = CONFIG["seed"]
BAND_COLUMNS = [f"band_{b}_from" for b in range(2, len(LEVELS) + 1)]
# gruppo di confronto del sottogruppo; "tutti" e' il riferimento gia' noto dalla Fase B
REFERENCE = "non diabetici"
TABLES = ["q1_discrimination", "q1_operating_points", "q2_levels", "q2_trend", "q3_sensitivity",
          "q4_bands", "q4_kappa", "threshold_descriptive"]


def groups(train):
    """Maschere dei gruppi sulle righe di train.csv, nell'ordine usato nelle tabelle."""
    dm = train[DIABETES].to_numpy()
    unexpected = set(np.unique(dm)) - {0, 1}
    if unexpected:
        raise ValueError(f"{DIABETES}: valori inattesi {sorted(unexpected)}")
    return {SUBGROUP: dm == 1, REFERENCE: dm == 0, "tutti": np.ones(len(train), dtype=bool)}


# --- discriminazione -----------------------------------------------------------------------

def delong_se(y, p):
    """Errore standard dell'AUC (DeLong et al. 1988), con la stessa formula di `evaluation.delong`.

    Serve esplicitamente perche' la differenza fra due gruppi disgiunti somma le varianze, e
    l'intervallo restituito da `evaluation.delong` e' troncato a [0, 1], quindi non sempre permette
    di risalire all'errore standard. `test_delong_se_matches_evaluation` verifica che i due
    percorsi coincidano quando il troncamento non interviene.
    """
    y = np.asarray(y) == 1
    p = np.asarray(p, dtype=float)
    pos, neg = p[y], p[~y]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return np.nan
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    v10 = (ranks[:m] - stats.rankdata(pos)) / n
    v01 = 1 - (ranks[m:] - stats.rankdata(neg)) / m
    return float(np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n))


def denoise_constant(p, g, atol=1e-12):
    """Toglie il rumore numerico da un modello che prevede una costante dentro ogni fold.

    Il classificatore di maggioranza delle tecniche con i pesi prevede 0,5, ma il valore salvato e'
    0,4999999999999938-0,4999999999999941: aggregando i fold quel residuo di ordine 1e-16 crea un
    ordinamento fittizio fra soggetti e porta l'AUC a 0,45 invece di 0,5. Quando la previsione e'
    costante dentro ogni fold e i fold coincidono a meno della precisione di macchina, i valori
    vengono riportati alla loro media. Sui modelli reali la condizione non si verifica mai.
    """
    if ev.is_constant(g) and np.ptp(p) < atol:
        return np.full_like(p, p.mean())
    return p


def discrimination(p, y, alpha=ALPHA):
    """Domanda 1 senza soglia. La PR-AUC va letta accanto alla prevalenza della stessa riga: fra
    gruppi con prevalenza diversa non e' confrontabile (Van Calster et al. 2025)."""
    auc, auc_low, auc_high = ev.delong(y, p, alpha)
    ap, ap_low, ap_high = ev.pr_auc(y, p, alpha)
    return {"n": int(len(y)), "positives": int(np.sum(y)), "prevalence": float(np.mean(y)),
            "auc": auc, "auc_low": auc_low, "auc_high": auc_high, "auc_se": delong_se(y, p),
            "pr_auc": ap, "pr_auc_low": ap_low, "pr_auc_high": ap_high}


def subgroup_comparison(discrimination_rows, subgroup=SUBGROUP, reference=REFERENCE, alpha=ALPHA):
    """Differenza di AUROC fra due gruppi disgiunti: SE(differenza) = sqrt(SE_1^2 + SE_2^2).

    Solo AUROC: Matos et al. 2026 raccomandano AUROC Parity e sconsigliano AUPRC Parity, che
    favorisce i sottogruppi ad alta prevalenza. Le prevalenze restano in tabella per ricordarlo.
    """
    table = pd.DataFrame(discrimination_rows).set_index(["technique", "model", "group"])
    z = ev.z_value(alpha)
    rows = []
    for technique, model in dict.fromkeys(table.index.droplevel("group")):
        if any((technique, model, g) not in table.index for g in (subgroup, reference)):
            continue
        a, b = table.loc[(technique, model, subgroup)], table.loc[(technique, model, reference)]
        difference = a["auc"] - b["auc"]
        se = np.sqrt(a["auc_se"] ** 2 + b["auc_se"] ** 2)
        rows.append({"technique": technique, "model": model, "metric": "auc",
                     "group": subgroup, "reference": reference,
                     "auc_group": a["auc"], "auc_reference": b["auc"],
                     "prevalence_group": a["prevalence"], "prevalence_reference": b["prevalence"],
                     "difference": difference, "low": difference - z * se,
                     "high": difference + z * se})
    return pd.DataFrame(rows)


# --- domande 2, 3, 4 alla soglia e alle fasce globali ---------------------------------------

def levels_table(p, codes, samples, alpha=ALPHA):
    """Domanda 2: probabilita' media per livello KDIGO, con IC bootstrap stratificato."""
    rows = []
    for i, level in enumerate(LEVELS):
        here = codes == i
        if not here.any():
            continue
        low, high = ev.percentile_interval([p[idx][codes[idx] == i].mean() for idx in samples], alpha)
        rows.append({"level": level, "n": int(here.sum()), "mean_probability": float(p[here].mean()),
                     "low": low, "high": high, "median_probability": float(np.median(p[here]))})
    return rows


def trend_row(p, codes, samples, alpha=ALPHA):
    """Domanda 2: concordanza di Jonckheere-Terpstra con IC bootstrap. Il p del test non viene
    riportato: in Fase C non si fanno test formali (protocollo, punto 5)."""
    low, high = ev.percentile_interval(
        [ev.jonckheere(p[idx], codes[idx])["concordance"] for idx in samples], alpha)
    return {"concordance": ev.jonckheere(p, codes)["concordance"], "low": low, "high": high}


def sensitivity_table(p, codes, cut, alpha=ALPHA):
    """Domanda 3: sensibilita' per livello alla soglia globale fissa, piu' la riga aggregata sui
    casi gravi, che e' il risultato principale della Fase C."""
    selections = [(level, codes == i) for i, level in enumerate(LEVELS[1:], start=1)]
    selections.append((SEVERE_LABEL, np.isin(codes, SEVERE)))
    rows = []
    for level, here in selections:
        n, k = int(here.sum()), int(np.sum(p[here] >= cut))
        value, low, high = ev.wilson(k, n, alpha)
        rows.append({"level": level, "n": n, "detected": k, "missed": n - k,
                     "sensitivity": value, "low": low, "high": high})
    return rows


def bands_table(p, codes, band_cuts):
    """Domanda 4: fasce globali contro livelli KDIGO. Le soglie delle fasce non si ricalcolano sul
    sottogruppo, quindi le fasce possono risultare sbilanciate: e' l'informazione cercata."""
    bands = ev.assign_bands(p, band_cuts)
    table = pd.crosstab(pd.Categorical(np.asarray(LEVELS)[codes], LEVELS),
                        pd.Categorical(bands, range(len(LEVELS))), dropna=False)
    rows = [{"level": level, "band": band + 1, "n": int(table.loc[level, band])}
            for level in LEVELS for band in range(len(LEVELS))]
    return rows, bands


def kappa_row(codes, bands, samples, alpha=ALPHA):
    """Kappa pesato fasce/livelli. A differenza della Fase A le fasce non vengono ristimate dentro
    ogni campione bootstrap: qui le soglie sono fisse per protocollo."""
    boot = [cohen_kappa_score(codes[idx], bands[idx], labels=range(len(LEVELS)),
                              weights=KAPPA_WEIGHTS) for idx in samples]
    low, high = ev.percentile_interval(boot, alpha)
    return {"kappa": cohen_kappa_score(codes, bands, labels=range(len(LEVELS)),
                                       weights=KAPPA_WEIGHTS),
            "low": low, "high": high, "observed_agreement": float(np.mean(codes == bands))}


def threshold_descriptive(p, y, codes, cut, target_sensitivity=TARGET, alpha=ALPHA):
    """Soglia che darebbe sensibilita' complessiva `target_sensitivity` fra i soli soggetti del
    gruppo, contro la soglia globale. Solo descrittiva: misura quanto la politica globale sia
    arbitraria su questo gruppo e non viene usata in nessun'altra tabella (protocollo, punto 7)."""
    severe = np.isin(codes, SEVERE)
    row = {"threshold_global": float(cut),
           "threshold_subgroup": float(ev.threshold_at_sensitivity(y, p, target_sensitivity)),
           "target_sensitivity": target_sensitivity, "severe_n": int(severe.sum())}
    for name in ("global", "subgroup"):
        point = ev.operating_point(y, p, row[f"threshold_{name}"], alpha)
        row[f"recall_{name}"] = point["recall"]
        row[f"specificity_{name}"] = point["specificity"]
        row[f"alert_rate_{name}"] = point["alert_rate"]
        row[f"severe_detected_{name}"] = int(np.sum(p[severe] >= row[f"threshold_{name}"]))
    return row


# --- controllo di coerenza ------------------------------------------------------------------

def check_dummy(discrimination_table, tolerance=1e-9):
    """Il classificatore di maggioranza prevede una costante, quindi dentro ogni gruppo deve dare
    AUC 0,5 e PR-AUC pari alla prevalenza di quel gruppo. Se non succede, il filtraggio del
    sottogruppo e' sbagliato (protocollo, punto 6)."""
    dummy = discrimination_table[discrimination_table["model"] == "dummy"]
    if dummy.empty:
        raise RuntimeError("classificatore di maggioranza assente: coerenza non verificabile")
    bad = dummy[(dummy["auc"] - 0.5).abs().gt(tolerance)
                | (dummy["pr_auc"] - dummy["prevalence"]).abs().gt(tolerance)]
    if len(bad):
        columns = ["technique", "model", "group", "auc", "pr_auc", "prevalence"]
        raise RuntimeError(f"controllo di coerenza fallito:\n{bad[columns].to_string(index=False)}")


# --- valutazione completa -------------------------------------------------------------------

def evaluate(oof, train, cutpoints, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """Tutte le tabelle della Fase C, come dict nome -> DataFrame.

    oof: previsioni out-of-fold della Fase B (row, fold, model, feature_set, technique, probability),
    row = posizione del soggetto in train. cutpoints: soglia e fasce globali per tecnica e modello.
    """
    y_all = target(train).to_numpy()
    codes_all = np.asarray(pd.Categorical(kdigo_level(train), LEVELS, ordered=True).codes)
    masks = groups(train)
    # stessi campioni bootstrap per tutte le tecniche e tutti i modelli dello stesso gruppo
    samples = {name: list(ev.stratified_indices(codes_all[mask], n_boot, seed))
               for name, mask in masks.items()}
    cutpoints = cutpoints.set_index(["technique", "model"])
    rows = {name: [] for name in TABLES}
    selected = oof[oof["technique"].isin(TECHNIQUES)]
    missing = set(TECHNIQUES) - set(selected["technique"])
    if missing:
        raise RuntimeError(f"tecniche assenti dalle previsioni: {sorted(missing)}")
    for (technique, model), g in selected.groupby(["technique", "model"], sort=False):
        p_all = denoise_constant(pooled(g, len(train)), g)
        for name, mask in masks.items():
            key = {"technique": technique, "model": model, "group": name}
            p, y, codes = p_all[mask], y_all[mask], codes_all[mask]
            rows["q1_discrimination"].append({**key, **discrimination(p, y, alpha)})
            if (technique, model) not in cutpoints.index:
                # solo un modello che prevede una costante puo' non avere soglia: per lui soglia e
                # fasce non hanno senso e resta il controllo di coerenza. Se manca la riga di un
                # modello vero, sparirebbe in silenzio da q3_sensitivity, cioe' dal risultato
                # principale della Fase C: e' un errore, non un caso da saltare.
                if not ev.is_constant(g):
                    raise RuntimeError(f"{technique} {model}: nessuna soglia in {CUTPOINTS.name}, "
                                       f"ma le previsioni non sono costanti dentro i fold")
                continue
            cut = float(cutpoints.loc[(technique, model), "threshold"])
            band_cuts = cutpoints.loc[(technique, model), BAND_COLUMNS].to_numpy(dtype=float)
            rows["q1_operating_points"].append({**key, **ev.operating_point(y, p, cut, alpha)})
            rows["q2_levels"] += [{**key, **r} for r in levels_table(p, codes, samples[name], alpha)]
            rows["q2_trend"].append({**key, **trend_row(p, codes, samples[name], alpha)})
            rows["q3_sensitivity"] += [{**key, **r} for r in sensitivity_table(p, codes, cut, alpha)]
            band_rows, bands = bands_table(p, codes, band_cuts)
            rows["q4_bands"] += [{**key, **r} for r in band_rows]
            rows["q4_kappa"].append({**key, **kappa_row(codes, bands, samples[name], alpha)})
            rows["threshold_descriptive"].append(
                {**key, **threshold_descriptive(p, y, codes, cut, TARGET, alpha)})
    tables = {name: pd.DataFrame(rows[name]) for name in TABLES}
    check_dummy(tables["q1_discrimination"])
    tables["subgroup_comparison"] = subgroup_comparison(rows["q1_discrimination"], alpha=alpha)
    return tables


def run(output=OUTPUT):
    train = pd.read_csv(PROCESSED / "train.csv")
    tables = evaluate(pd.read_csv(PREDICTIONS), train, pd.read_csv(CUTPOINTS))
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
        print(f"{output / name}.csv", flush=True)


if __name__ == "__main__":
    run()
