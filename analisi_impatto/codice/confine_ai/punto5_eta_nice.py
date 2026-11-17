"""Punto 5: l'età e la regola NICE. PREDITTIVO, a parità di persone testate.

Piano fissato prima: analisi_impatto/04_esperimenti/confine_ai/piano_punto5.md
Jinuo: solo training, OOF del RF Fase A. NHANES: RF in CV come nel punto 4c.
"""
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import punto4_nhanes as p4  # noqa: E402

ROOT, p0, SEED = p4.ROOT, p4.p0, p4.SEED
sys.path.insert(0, str(ROOT))
from src.data import kidney  # noqa: E402

OUT = ROOT / "analisi_impatto/04_esperimenti/confine_ai/punto5"
FIGDIR = ROOT / "analytics/soglia_ai/punto5"
ETA = 60


def logit_cv(Z, y, fold):
    s = np.zeros(len(y))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        s[te] = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    return s


def confronti(nome, y, fold, age, rule, rule_age, ai, logi):
    tiebreak = np.random.default_rng(SEED).permutation(len(y))
    rows = []
    for budget, ref_nome, ref in [("budget R-a", "R-a: diabete o ipertensione", rule),
                                  ("budget R-a + età", f"R-a + età ≥ {ETA}", rule_age)]:
        c_ref = p0.counts(ref, ref, y)
        rows.append({"dataset": nome, "budget": budget, "strategia": ref_nome, "testati": int(ref.sum()),
                     "positivi": c_ref["pos_regola"], "diff_vs_regola": 0, "ic_basso": np.nan, "ic_alto": np.nan})
        strat = [("logistica semplice (età, sesso, diabete, ipertensione)", logi), ("AI (Random Forest)", ai)]
        if budget == "budget R-a":
            strat.insert(0, ("solo età (i più anziani)", age + 1e-6 * tiebreak / len(y)))
        for sn, sc in strat:
            sel = p0.select_at_parity(sc, fold, ref, tiebreak)
            c = p0.counts(sel, ref, y)
            lo, hi = p0.boot_diff(sel, ref, y, np.random.default_rng(SEED))
            rows.append({"dataset": nome, "budget": budget, "strategia": sn, "testati": c["testati"],
                         "positivi": c["pos_modello"], "diff_vs_regola": c["pos_modello"] - c["pos_regola"],
                         "ic_basso": lo, "ic_alto": hi})
    return rows


def jinuo():
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    y = kidney.target(df).to_numpy()
    ai, fold = p0.load_scores("phase_a/main/random_forest", len(df))
    age = df["Age"].to_numpy(float)
    dm = df["DM"].eq(1).to_numpy()
    ht = df["HypertenHis"].fillna(0).eq(1).to_numpy()
    rule = dm | ht
    rule_age = rule | (age >= ETA)
    Z = np.column_stack([age, kidney.is_female(df).astype(float), dm, ht])
    logi = logit_cv(Z, y, fold)
    return confronti("Jinuo (training, n = 4.350)", y, fold, age, rule, rule_age, ai, logi)


def nhanes():
    d, _ = p4.load()
    X = d[p4.FEAT].astype(float).to_numpy()
    y = d["y"].to_numpy()
    ai = np.zeros(len(d))
    fold = np.zeros(len(d), int)
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, n_jobs=-1, random_state=SEED)
        ai[te] = rf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        fold[te] = k
    age = d["RIDAGEYR"].to_numpy(float)
    rule = (d["diab_noto"].eq(1) | d["iper_nota"].eq(1)).to_numpy()
    rule_age = rule | (age >= ETA)
    Z = np.column_stack([age, d["female"].astype(float), d["diab_noto"], d["iper_nota"]])
    logi = logit_cv(Z, y, fold)
    return confronti(f"NHANES 2005–2018 (n = {len(d):,})".replace(",", "."), y, fold, age, rule, rule_age, ai, logi)


def figura(t):
    colori = {"R-a": "#8a8a8a", "solo": "#b9b9b9", "logistica": "#c46b3c", "AI": "#2f6db5"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (ds, g) in zip(axes, t.groupby("dataset", sort=False)):
        xpos, labels, x = [], [], 0
        top = g["positivi"].max()
        for b, gb in g.groupby("budget", sort=False):
            start = x
            for _, r in gb.iterrows():
                key = next(k for k in colori if r.strategia.startswith(k))
                ax.bar(x, r.positivi, 0.8, color=colori[key])
                txt = f"{r.positivi}" if r.diff_vs_regola == 0 else \
                    f"{r.positivi}\n{r.diff_vs_regola:+d}\n({r.ic_basso:+.0f}/{r.ic_alto:+.0f})"
                ax.text(x, r.positivi + top * 0.01, txt, ha="center", va="bottom", fontsize=8)
                xpos.append(x)
                corto = {"R-a:": "Regola\nattuale", "R-a +": "Regola\n+ età ≥ 60", "solo": "Solo\netà",
                         "logistica": "Logistica\nsemplice", "AI": "AI"}
                labels.append(next(v for k, v in corto.items() if r.strategia.startswith(k)))
                x += 1
            ax.text((start + x - 1) / 2, top * 1.33, f"{b}: {int(gb.testati.iloc[0])} testati",
                    ha="center", fontsize=9, color="#3a3a3a")
            x += 0.8
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.set_ylim(0, top * 1.42)
        ax.set_ylabel("Malati KDIGO trovati")
        ax.set_title(ds, fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("L'AI fa meglio delle linee guida anche quando alla regola si aggiunge l'età?", fontsize=11)
    fig.text(0.5, 0.01, "Regola attuale = diabete o ipertensione noti. Logistica semplice = età, sesso, diabete, "
             "ipertensione. AI = Random Forest senza esami renali.\nSopra le barre: malati trovati, differenza "
             "dalla regola dello stesso gruppo e IC 95%.", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT / "fig1_eta_nice.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(jinuo() + nhanes())
    t.to_csv(OUT / "P5_eta_nice.csv", index=False)
    figura(t)
    shutil.copy(OUT / "fig1_eta_nice.png", FIGDIR / "fig1_eta_nice.png")
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
