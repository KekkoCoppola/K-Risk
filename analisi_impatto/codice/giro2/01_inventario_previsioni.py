"""Giro 2, idea Rashomon: inventario delle previsioni out-of-fold salvate per paziente.

Per ogni insieme di record (una cartella e un nome di modello, 5 fold esterni) controlla:
- formato (campi delle righe e del punteggio, campi extra come mu, previsioni interne);
- allineamento: le righe sono posizioni in train.csv e coincidono con i fold esterni del progetto;
- completezza: copertura dei 4.350 soggetti, valori mancanti o non finiti;
- scala del punteggio (probabilità in [0, 1] o no), numero di valori distinti;
- doppioni: impronta del vettore (copia esatta) e impronta dei ranghi (stesso ordinamento, per
  esempio Platt). Serve solo a contare i modelli DISTINTI: non si misura il disaccordo;
- iperparametri, tentativi di Optuna, secondi registrati.
NON calcola nessuna metrica di discriminazione né confronti fra modelli. La cartella
analytics/test/ è esclusa: se ne contano solo i file, senza aprirli.
Uscite: out/01_inventario_previsioni.csv (un insieme per riga), out/01_inventario_sintesi.csv,
out/01_file_lunghi.csv, out/01_log_optuna.csv.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from comune import ANALYTICS, OUT, components, inner_folds, load_train, outer_folds

PREFIXED = re.compile(r"^(?P<model>.+)_outer(?P<k>\d)\.json$")
PLAIN = re.compile(r"^outer(?P<k>\d)\.json$")
SCORE_KEYS = ("score", "probability", "p_raw", "probability_calibrated")
DEFINITORIE = {"positive_control": "74 + UmALB (definitoria)",
               "semi_synthetic_control/z75": "74 + z da UMAUCR (definitoria)",
               "semi_synthetic_control/z80": "74 + z da UMAUCR (definitoria)"}


def fingerprint(v, decimals=12):
    return hashlib.sha1(np.round(np.asarray(v, float), decimals).tobytes()).hexdigest()[:12]


def rank_fingerprint(v, fold=None):
    """Impronta dei ranghi; con fold, ranghi calcolati dentro ogni fold esterno (una trasformazione
    monotona stimata fold per fold, come Platt, conserva solo questi)."""
    v = np.asarray(v, float)
    if fold is None:
        return hashlib.sha1(rankdata(v).tobytes()).hexdigest()[:12]
    r = np.empty_like(v)
    for k in np.unique(fold):
        r[fold == k] = rankdata(v[fold == k])
    return hashlib.sha1(r.tobytes()).hexdigest()[:12]


def discover():
    groups = {}
    for path in ANALYTICS.rglob("*.json"):
        rel = path.relative_to(ANALYTICS).as_posix()
        if rel.startswith("test/"):
            continue
        m = PREFIXED.match(path.name) or PLAIN.match(path.name)
        if not m:
            continue
        model = m.groupdict().get("model") or ""
        key = (path.parent.relative_to(ANALYTICS).as_posix(), model)
        groups.setdefault(key, {})[int(m["k"])] = path
    return groups


def sibling_rows(directory, model, k):
    """Righe per i record di Platt della Fase B, che non le salvano: dal record grezzo della stessa
    tecnica, oppure (tecnica none) dal riferimento della Fase A (XGBoost: profondità 1-12)."""
    technique = directory.split("/")[1]
    if technique == "none":
        base = "phase_a/sensitivity/depth_1_12/main" if model == "xgboost" else "phase_a/main"
    else:
        base = f"phase_b/{technique}/main"
    record = json.loads((ANALYTICS / base / f"{model}_outer{k}.json").read_text(encoding="utf-8"))
    return record["rows"], base


def classify(directory, model, first):
    """Fase, ruolo, bersaglio e variabili di ingresso di un insieme di record."""
    parts = directory.split("/")
    name = model or parts[-1]
    full = f"{directory}/{model}" if model else directory
    role, target, inputs = "modello", "composito", "74 ammesse"
    if parts[0] == "phase_a":
        phase = "A (sensibilità 1-12)" if "sensitivity" in parts else "A"
        if parts[-1] == "no_consequence":
            inputs = "74 meno le conseguenze (no_consequence)"
    elif parts[0] == "phase_b":
        if directory in ("phase_b/none", "phase_b/class_weights"):
            phase, role = "B (ORFANO: nessun codice lo produce)", "orfano"
        elif "calibration" in parts:
            phase, role = "B (Platt)", "trasformazione monotona"
        else:
            phase = "B"
    elif parts[0] == "phase_d":
        phase = "D"
        key = "/".join(parts[1:])
        if key in DEFINITORIE:
            role, inputs = "controllo (diagnostica)", DEFINITORIE[key]
        elif key == "tri_ensemble_leaky":
            role, inputs = "diagnostica (selezione sull'intero training)", "21 scelte con leakage di selezione"
        elif key in ("target_decomposition/albuminuria", "continuous_target/albuminuria"):
            role, target = "componente", "albuminuria (ACR >= 30)"
        elif key == "target_decomposition/egfr":
            role, target = "componente", "eGFR < 60"
        elif key in ("ensemble_mean", "tri_ensemble_top21", "target_decomposition", "continuous_target/combinato"):
            role = "combinazione di altri modelli"
        if key.startswith("tri_ensemble"):
            inputs = inputs if key == "tri_ensemble_leaky" else "21 scelte dentro il fold"
    elif parts[0] == "phase_e":
        kind = first.get("kind", "")
        phase = f"E ({kind})"
        if kind == "reused":
            role = "copia (riuso)"
        elif kind == "combined":
            role = "combinazione di altri modelli"
        elif kind == "continuous":
            role = "modello (bersagli continui log ACR e log SCRE)"
    else:
        phase = parts[0]
    if name in ("dummy",):
        role, inputs = "maggioranza (controllo)", "nessuna"
    if name == "lr_scored":
        inputs = "5 predittori SCORED"
    return {"id": full, "fase": phase, "nome": name, "ruolo": role, "bersaglio": target, "ingressi": inputs}


def main():
    df = load_train()
    n = len(df)
    comp = components(df)
    folds = outer_folds(df)
    valid = [va for _, va in folds]
    train = [tr for tr, _ in folds]
    inner = [inner_folds(df, tr) for tr in train]
    fold_of = np.full(n, -1)
    for k, va in enumerate(valid):
        fold_of[va] = k

    rows_out, vectors = [], {}
    for (directory, model), files in sorted(discover().items()):
        records = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in files.items()}
        first = records[min(records)]
        info = classify(directory, model, first)
        s = np.full(n, np.nan)
        seen = np.zeros(n, int)
        fold_ok, order_ok, rows_from, ytrue_ok = True, True, "record", None
        inner_ok, has_inner, has_mu = None, False, False
        params_present, params_differ, seconds, trials = 0, False, 0.0, []
        score_key = next((k for k in SCORE_KEYS if k in first), None)
        param_sets = []
        for k, rec in sorted(records.items()):
            rows = rec.get("rows", rec.get("valid_idx"))
            if rows is None:
                rows, rows_from = sibling_rows(directory, model, k)
            rows = np.asarray(rows, int)
            values = np.asarray(rec[score_key], float)
            assert len(rows) == len(values), (directory, model, k)
            s[rows] = values
            np.add.at(seen, rows, 1)
            fold_ok &= set(rows.tolist()) == set(valid[k].tolist())
            order_ok &= np.array_equal(rows, valid[k])
            if "y_true" in rec:
                ok = np.array_equal(np.asarray(rec["y_true"], int), comp["y"][rows])
                ytrue_ok = ok if ytrue_ok is None else (ytrue_ok and ok)
            if "inner_rows" in rec:
                has_inner = True
                ir, jf = np.asarray(rec["inner_rows"], int), np.asarray(rec["inner_fold"], int)
                ok = set(ir.tolist()) == set(train[k].tolist()) and len(ir) == len(train[k])
                ok &= all(set(ir[jf == j].tolist()) == set(va.tolist()) for j, (_, va) in enumerate(inner[k]))
                ok &= np.isfinite(np.asarray(rec["inner_score"], float)).all()
                inner_ok = ok if inner_ok is None else (inner_ok and ok)
            has_mu |= "mu" in rec
            if isinstance(rec.get("params"), dict) and rec["params"]:
                params_present += 1
                param_sets.append(json.dumps(rec["params"], sort_keys=True))
            seconds += float(rec.get("seconds", 0) or 0) + float(rec.get("train_time", 0) or 0)
            if isinstance(rec.get("trials"), int):
                trials.append(rec["trials"])
        params_differ = len(set(param_sets)) > 1
        complete = len(records) == 5 and (seen == 1).all() and np.isfinite(s).all()
        finite = s[np.isfinite(s)]
        row = {**info, "cartella": directory, "file_modello": model or "outerK.json",
               "fold_presenti": len(records), "campo_punteggio": score_key, "righe_da": rows_from,
               "soggetti_coperti": int((seen > 0).sum()), "righe_doppie": int((seen > 1).sum()),
               "completo_4350": bool(complete), "fold_uguali_al_progetto": bool(fold_ok),
               "stesso_ordine_delle_righe": bool(order_ok),
               "y_true_coerente": ytrue_ok, "non_finiti": int((~np.isfinite(s) & (seen > 0)).sum()),
               "min": float(finite.min()) if finite.size else np.nan,
               "max": float(finite.max()) if finite.size else np.nan,
               "in_0_1": bool(finite.size and finite.min() >= 0 and finite.max() <= 1),
               "valori_distinti": int(np.unique(finite).size),
               "previsioni_interne_oof": has_inner, "previsioni_interne_coerenti": inner_ok,
               "stima_continua_mu": has_mu, "fold_con_iperparametri": params_present,
               "iperparametri_diversi_fra_fold": params_differ,
               "tentativi_optuna_per_fold": (f"{min(trials)}-{max(trials)}" if trials else ""),
               "secondi_registrati": round(seconds, 1)}
        if complete:
            row["impronta"] = fingerprint(s)
            row["impronta_ranghi"] = rank_fingerprint(s)
            row["impronta_ranghi_nel_fold"] = rank_fingerprint(s, fold_of)
            vectors[info["id"]] = s
        rows_out.append(row)

    inv = pd.DataFrame(rows_out)
    # doppioni: stesso vettore (copia esatta) o stesso ordinamento (trasformazione monotona)
    first_of = {}
    for _, r in inv[inv.completo_4350].iterrows():
        first_of.setdefault(r.impronta, r.id)
    inv["copia_esatta_di"] = [("" if (not r.completo_4350 or first_of[r.impronta] == r.id) else first_of[r.impronta])
                              for _, r in inv.iterrows()]
    first_rank = {}
    for _, r in inv[inv.completo_4350].iterrows():
        first_rank.setdefault(r.impronta_ranghi, r.id)
    inv["stesso_ordinamento_di"] = [("" if (not r.completo_4350 or first_rank[r.impronta_ranghi] == r.id)
                                     else first_rank[r.impronta_ranghi]) for _, r in inv.iterrows()]
    first_fold_rank = {}
    for _, r in inv[inv.completo_4350].iterrows():
        first_fold_rank.setdefault(r.impronta_ranghi_nel_fold, r.id)
    inv["stesso_ordinamento_nel_fold_di"] = [
        ("" if (not r.completo_4350 or first_fold_rank[r.impronta_ranghi_nel_fold] == r.id)
         else first_fold_rank[r.impronta_ranghi_nel_fold]) for _, r in inv.iterrows()]
    inv.to_csv(OUT / "01_inventario_previsioni.csv", index=False)

    # file lunghi: coincidono con i record JSON?
    long_rows = []

    def same_record(s, tol=1e-12):
        """Primo record JSON completo uguale al vettore entro tol (i CSV differiscono dai JSON solo per
        l'arrotondamento in virgola mobile, circa 1e-16)."""
        for name, v in vectors.items():
            if np.max(np.abs(v - s)) <= tol:
                return name
        return ""
    specs = [("phase_a/oof_predictions.csv", ["model", "feature_set"], ["probability"]),
             ("phase_a/sensitivity/depth_1_12/oof_predictions.csv", ["model", "feature_set"], ["probability"]),
             ("phase_b/oof_predictions.csv", ["technique", "model", "feature_set"], ["probability", "probability_calibrated"]),
             ("phase_b/class_weights/oof_predictions.csv", ["technique", "model"], ["p_raw", "p_platt", "p_isotonic"])]
    for rel, keys, values in specs:
        table = pd.read_csv(ANALYTICS / rel)
        row_col = "row" if "row" in table else "row_idx"
        for key, g in table.groupby(keys):
            for v in values:
                s = np.full(n, np.nan)
                s[g[row_col].to_numpy()] = g[v].to_numpy()
                ok = np.isfinite(s).all() and len(g) == n
                long_rows.append({"file": rel, "chiave": "/".join(map(str, key if isinstance(key, tuple) else (key,))),
                                  "colonna": v, "righe": len(g), "completo_4350": bool(ok),
                                  "uguale_al_record": same_record(s) if ok else "",
                                  "in_0_1": bool(ok and np.nanmin(s) >= 0 and np.nanmax(s) <= 1)})
    pd.DataFrame(long_rows).to_csv(OUT / "01_file_lunghi.csv", index=False)

    # log di Optuna: tentativi completi per modello e fold (nessun valore riportato)
    log_rows = []
    for path in sorted(ANALYTICS.rglob("trials/*.csv")):
        rel = path.relative_to(ANALYTICS).as_posix()
        if rel.startswith("test/"):
            continue
        t = pd.read_csv(path)
        params = [c for c in t.columns if c.startswith("params_")]
        log_rows.append({"file": rel, "tentativi": len(t),
                         "completi": int((t.state == "COMPLETE").sum()) if "state" in t else np.nan,
                         "interrotti": int((t.state == "PRUNED").sum()) if "state" in t else np.nan,
                         "iperparametri": len(params), "ha_valore_interno": "value" in t,
                         "ha_durata": "duration" in t})
    pd.DataFrame(log_rows).to_csv(OUT / "01_log_optuna.csv", index=False)

    # sintesi
    test_files = sum(1 for p in (ANALYTICS / "test").rglob("*") if p.is_file())
    complete = inv[inv.completo_4350]
    composite = complete[complete.bersaglio == "composito"]
    distinct_vec = composite[composite.copia_esatta_di == ""]
    distinct_rank = composite[composite.stesso_ordinamento_di == ""]
    members = distinct_vec[distinct_vec.ruolo.isin(["modello", "modello (bersagli continui log ACR e log SCRE)",
                                                    "combinazione di altri modelli"])
                           & distinct_vec.ingressi.isin(["74 ammesse", "74 meno le conseguenze (no_consequence)",
                                                         "5 predittori SCORED", "21 scelte dentro il fold"])]
    members_no_comb = members[members.ruolo != "combinazione di altri modelli"]
    members_main = members_no_comb[members_no_comb.ingressi == "74 ammesse"]
    summary = [
        ("insiemi di record trovati (cartella x modello)", len(inv)),
        ("insiemi completi (5 fold, 4.350 soggetti, valori finiti)", len(complete)),
        ("insiemi incompleti", int((~inv.completo_4350).sum())),
        ("insiemi con righe diverse dai fold esterni del progetto", int((~inv.fold_uguali_al_progetto).sum())),
        ("completi sul bersaglio composito", len(composite)),
        ("... vettori distinti (tolte le copie esatte)", len(distinct_vec)),
        ("... ordinamenti globali distinti (tolte anche le trasformazioni monotone)", len(distinct_rank)),
        ("... ordinamenti distinti dentro i fold (tolto anche Platt stimato fold per fold)",
         int((composite.stesso_ordinamento_nel_fold_di == "").sum())),
        ("membri candidati per Rashomon: composito, input ammessi, vettore distinto, niente controlli/orfani/maggioranza", len(members)),
        ("... senza le combinazioni di altri modelli", len(members_no_comb)),
        ("... senza combinazioni e solo con le 74 ammesse", len(members_main)),
        ("... di cui con punteggio fuori da [0, 1]", int((~members.in_0_1).sum())),
        ("insiemi con previsioni OOF interne (fold interni)", int(inv.previsioni_interne_oof.sum())),
        ("... con previsioni interne coerenti con i fold interni del progetto", int((inv.previsioni_interne_coerenti == True).sum())),  # noqa: E712
        ("insiemi con iperparametri in tutti i 5 fold", int((inv.fold_con_iperparametri == 5).sum())),
        ("log di Optuna (file)", len(log_rows)),
        ("file in analytics/test (contati, mai aperti)", test_files),
    ]
    pd.DataFrame(summary, columns=["voce", "valore"]).to_csv(OUT / "01_inventario_sintesi.csv", index=False)
    members.assign(tipo="membro")[["id", "fase", "nome", "ruolo", "ingressi", "in_0_1", "valori_distinti",
                                   "previsioni_interne_oof", "fold_con_iperparametri", "secondi_registrati"]] \
        .to_csv(OUT / "01_membri_rashomon.csv", index=False)
    for voce, valore in summary:
        print(f"{valore:>6}  {voce}")


if __name__ == "__main__":
    main()
