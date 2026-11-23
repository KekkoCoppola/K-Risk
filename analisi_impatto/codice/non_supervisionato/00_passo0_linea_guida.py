"""Passo 0 del piano non supervisionato: 71 modelli contro la linea guida, a parità di testati.

[PREDITTIVO] Esplorazione dichiarata, fatta prima della preregistrazione
(`analisi_impatto/non_supervisionato.md`, sezione 5). Nessun addestramento: si leggono
le previsioni OOF già salvate (5 fold esterni, solo training). Il test set non si legge.

Per ogni modello e ogni regola, dentro ogni fold esterno il modello "testa" lo stesso
numero di soggetti della regola (i punteggi più alti; pareggi rotti con un ordine casuale
fisso, seed 42). Poi si contano i positivi trovati dall'uno, dall'altro, da entrambi.

Uscite (solo statistiche aggregate) in `analisi_impatto/04_esperimenti/non_supervisionato/passo0/`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data import kidney  # noqa: E402

ANALYTICS = ROOT / "analytics"
MEMBERS = ROOT / "analisi_impatto/codice/giro2/out/01_membri_rashomon.csv"
OUT = ROOT / "analisi_impatto/04_esperimenti/non_supervisionato/passo0"
SEED = 42
N_BOOT = 1000
HEADLINE = ["phase_a/main/random_forest", "phase_d/tabpfn"]


def record_paths(member_id):
    """Percorsi dei 5 record OOF: `<dir>/<modello>_outerK.json` (A, B) o `<id>/outerK.json` (D, E)."""
    p = Path(member_id)
    a = [ANALYTICS / p.parent / f"{p.name}_outer{k}.json" for k in range(5)]
    if all(x.exists() for x in a):
        return a
    b = [ANALYTICS / p / f"outer{k}.json" for k in range(5)]
    if all(x.exists() for x in b):
        return b
    raise FileNotFoundError(member_id)


def load_scores(member_id, n):
    score = np.full(n, np.nan)
    fold = np.full(n, -1)
    for k, path in enumerate(record_paths(member_id)):
        r = json.loads(path.read_text(encoding="utf-8"))
        key = "probability" if "probability" in r else "score"
        rows = np.asarray(r["rows"], dtype=int)
        score[rows] = np.asarray(r[key], dtype=float)
        fold[rows] = k
    if np.isnan(score).any() or (fold < 0).any():
        raise ValueError(f"OOF incompleto: {member_id}")
    return score, fold


def select_at_parity(score, fold, rule, tiebreak):
    """Dentro ogni fold, il modello testa tanti soggetti quanti ne testa la regola."""
    chosen = np.zeros(len(score), dtype=bool)
    for k in np.unique(fold):
        idx = np.flatnonzero(fold == k)
        m = int(rule[idx].sum())
        order = np.lexsort((tiebreak[idx], -score[idx]))
        chosen[idx[order[:m]]] = True
    return chosen


def counts(sel_model, sel_rule, y):
    pos = y == 1
    return {
        "testati": int(sel_rule.sum()),
        "pos_modello": int((sel_model & pos).sum()),
        "pos_regola": int((sel_rule & pos).sum()),
        "pos_entrambi": int((sel_model & sel_rule & pos).sum()),
        "pos_solo_modello": int((sel_model & ~sel_rule & pos).sum()),
        "pos_solo_regola": int((~sel_model & sel_rule & pos).sum()),
        "pos_unione": int(((sel_model | sel_rule) & pos).sum()),
        "testati_unione": int((sel_model | sel_rule).sum()),
    }


def boot_diff(sel_model, sel_rule, y, rng):
    """IC bootstrap (percentili) della differenza di positivi trovati, selezioni fisse."""
    pos = y == 1
    d = (sel_model & pos).astype(int) - (sel_rule & pos).astype(int)
    n = len(d)
    draws = rng.integers(0, n, size=(N_BOOT, n))
    stats = d[draws].sum(axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def profile(mask, df, y, egfr, acr, level):
    """Profilo aggregato dei positivi in `mask`."""
    m = mask & (y == 1)
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    sub_acr = acr[m]
    return {
        "n": n,
        "eta_mediana": float(np.median(df.loc[m, "Age"])),
        "quota_donne": float(kidney.is_female(df)[m].mean()),
        "quota_DM": float((df.loc[m, "DM"] == 1).mean()),
        "quota_sola_albuminuria": float(((acr >= 30) & (egfr >= 60))[m].mean()),
        "quota_egfr_sotto_60": float((egfr < 60)[m].mean()),
        "quota_acr_30_300": float(((sub_acr >= 30) & (sub_acr < 300)).mean()),
        "quota_acr_almeno_300": float((sub_acr >= 300).mean()),
        "quota_acr_almeno_45": float((sub_acr >= 45).mean()),
        **{f"quota_livello_{l.replace(' ', '_')}": float((level[m] == l).mean()) for l in kidney.LEVELS},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    n = len(df)
    y = kidney.target(df).to_numpy()
    egfr = kidney.egfr(df).to_numpy()
    acr = df["UMAUCR"].to_numpy()
    level = kidney.kdigo_level(df).astype(str).to_numpy()

    ht = df["HypertenHis"].fillna(0).eq(1).to_numpy()
    bp = ((df["Bpsys"] >= 140) | (df["Bpdia"] >= 90)).to_numpy()
    dm = df["DM"].eq(1).to_numpy()
    dmhis = df["DMhis"].eq(1).to_numpy()
    rules = {
        "R-a_DM": dm | ht,
        "R-b_DM": dm | ht | bp,
        "R-a_DMhis": dmhis | ht,
        "R-b_DMhis": dmhis | ht | bp,
    }

    members = pd.read_csv(MEMBERS)
    tiebreak = np.random.default_rng(SEED).permutation(n)
    rng = np.random.default_rng(SEED)

    rows, profiles, selections = [], [], {}
    fold_ref = None
    for _, mem in members.iterrows():
        score, fold = load_scores(mem["id"], n)
        if fold_ref is None:
            fold_ref = fold
        elif not np.array_equal(fold, fold_ref):
            raise ValueError(f"fold diversi: {mem['id']}")
        for rname, rule in rules.items():
            sel = select_at_parity(score, fold, rule, tiebreak)
            c = counts(sel, rule, y)
            lo, hi = boot_diff(sel, rule, y, rng)
            rows.append({"id": mem["id"], "fase": mem["fase"], "nome": mem["nome"],
                         "ruolo": mem["ruolo"], "regola": rname, **c,
                         "diff_pos": c["pos_modello"] - c["pos_regola"],
                         "diff_ic_basso": lo, "diff_ic_alto": hi})
            selections[(mem["id"], rname)] = sel
            if mem["id"] in HEADLINE:
                for nome, mask in [("solo_modello", sel & ~rule), ("solo_regola", ~sel & rule),
                                   ("entrambi", sel & rule), ("nessuno", ~sel & ~rule)]:
                    profiles.append({"id": mem["id"], "regola": rname, "gruppo": nome,
                                     **profile(mask, df, y, egfr, acr, level)})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "01_modelli_contro_regola.csv", index=False)
    pd.DataFrame(profiles).to_csv(OUT / "02_profilo_positivi.csv", index=False)

    # sintesi sui 71 modelli
    summ = []
    for rname, g in res.groupby("regola"):
        summ.append({
            "regola": rname, "testati": int(g["testati"].iloc[0]),
            "pos_regola": int(g["pos_regola"].iloc[0]), "positivi_totali": int(y.sum()),
            "modelli": len(g),
            "modelli_con_piu_positivi": int((g["diff_pos"] > 0).sum()),
            "modelli_con_meno_positivi": int((g["diff_pos"] < 0).sum()),
            "modelli_ic_sopra_0": int((g["diff_ic_basso"] > 0).sum()),
            "modelli_ic_sotto_0": int((g["diff_ic_alto"] < 0).sum()),
            "pos_modello_mediana": float(g["pos_modello"].median()),
            "pos_modello_min": int(g["pos_modello"].min()),
            "pos_modello_max": int(g["pos_modello"].max()),
            "pos_unione_mediana": float(g["pos_unione"].median()),
            "testati_unione_mediana": float(g["testati_unione"].median()),
        })
    pd.DataFrame(summ).to_csv(OUT / "03_sintesi.csv", index=False)

    # consenso: positivi fuori dalla regola trovati da almeno metà dei modelli "modello"
    cons = []
    only_models = members.loc[members["ruolo"].str.startswith("modello"), "id"].tolist()
    for rname, rule in rules.items():
        votes = np.sum([selections[(m, rname)] for m in only_models], axis=0)
        outside = (y == 1) & ~rule
        cons.append({"regola": rname, "modelli": len(only_models),
                     "positivi_fuori_regola": int(outside.sum()),
                     "trovati_da_almeno_meta": int((outside & (votes >= len(only_models) / 2)).sum()),
                     "trovati_da_tutti": int((outside & (votes == len(only_models))).sum()),
                     "trovati_da_nessuno": int((outside & (votes == 0)).sum())})
    pd.DataFrame(cons).to_csv(OUT / "04_consenso.csv", index=False)
    print(pd.DataFrame(summ).to_string(index=False))
    print(pd.DataFrame(cons).to_string(index=False))


if __name__ == "__main__":
    main()
