"""Punto 6. PREDITTIVO.
6A: punteggio NON supervisionato (k-means k=4 primario, autoencoder secondario) sulle 74 colonne
    senza esami renali e senza etichette, contro la regola R-a/R-b a parità di testati (Jinuo, training).
6B: validazione esterna vera: RF addestrato sul training Jinuo (14 variabili comuni) applicato a
    NHANES senza riaddestramento.
Piano fissato prima: analisi_impatto/04_esperimenti/confine_ai/piano_punto6.md
"""
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import punto4_nhanes as p4  # noqa: E402

ROOT, p0, SEED = p4.ROOT, p4.p0, p4.SEED
sys.path.insert(0, str(ROOT))
from src.data import kidney  # noqa: E402
from src.config import CONFIG  # noqa: E402

OUT = ROOT / "analisi_impatto/04_esperimenti/confine_ai/punto6"
FIGDIR = ROOT / "analytics/soglia_ai/punto6"
B = 1000


def boot_auc(y, s, rng):
    v = []
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


# ------------------------------------------------------------------ 6A
def prepara(Xtr, Xte):
    med = Xtr.median()
    a, b = Xtr.fillna(med), Xte.fillna(med)
    mu, sd = a.mean(), a.std().replace(0, 1)
    return ((a - mu) / sd).to_numpy(), ((b - mu) / sd).to_numpy()


def analisi_6a():
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    y = kidney.target(df).to_numpy()
    cols = CONFIG["features"]["numeric"] + CONFIG["features"]["categorical"]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X["HypertenHis"] = X["HypertenHis"].fillna(0)
    rf, fold = p0.load_scores("phase_a/main/random_forest", len(df))
    km, ae = np.zeros(len(df)), np.zeros(len(df))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        a, b = prepara(X[tr], X[te])
        m = KMeans(4, n_init=10, random_state=SEED).fit(a)
        big = int(np.bincount(m.labels_).argmax())
        km[te] = np.linalg.norm(b - m.cluster_centers_[big], axis=1)
        net = MLPRegressor(hidden_layer_sizes=(32, 8, 32), activation="relu", early_stopping=True,
                           validation_fraction=0.1, max_iter=500, random_state=SEED).fit(a, a)
        ae[te] = ((net.predict(b) - b) ** 2).mean(1)
    dm = df["DM"].eq(1).to_numpy()
    ht = df["HypertenHis"].fillna(0).eq(1).to_numpy()
    bp = ((df["Bpsys"] >= 140) | (df["Bpdia"] >= 90)).to_numpy()
    regole = {"R-a (diabete o ipertensione)": dm | ht, "R-b (anche pressione ≥ 140/90)": dm | ht | bp}
    tiebreak = np.random.default_rng(SEED).permutation(len(df))
    rng = np.random.default_rng(SEED)
    auc = {}
    for nome, s in [("k-means k=4 (non supervisionato, primario)", km),
                    ("autoencoder (non supervisionato)", ae), ("Random Forest (supervisionato, Fase A)", rf)]:
        auc[nome] = (roc_auc_score(y, s), *boot_auc(y, s, rng))
    rows = []
    for rn, rule in regole.items():
        rows.append({"analisi": "6A Jinuo", "budget": rn, "strategia": "regola", "testati": int(rule.sum()),
                     "positivi": int((rule & (y == 1)).sum()), "diff": 0, "ic_basso": np.nan, "ic_alto": np.nan,
                     "auroc": np.nan, "auroc_ic": ""})
        for nome, s in [("k-means k=4 (non supervisionato, primario)", km),
                        ("autoencoder (non supervisionato)", ae), ("Random Forest (supervisionato, Fase A)", rf)]:
            sel = p0.select_at_parity(s, fold, rule, tiebreak)
            c = p0.counts(sel, rule, y)
            lo, hi = p0.boot_diff(sel, rule, y, np.random.default_rng(SEED))
            rows.append({"analisi": "6A Jinuo", "budget": rn, "strategia": nome, "testati": c["testati"],
                         "positivi": c["pos_modello"], "diff": c["pos_modello"] - c["pos_regola"],
                         "ic_basso": lo, "ic_alto": hi, "auroc": auc[nome][0],
                         "auroc_ic": f"{auc[nome][1]:.3f}–{auc[nome][2]:.3f}"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 6B
COMUNI = ["eta", "female", "BMI", "girovita", "PAS", "PAD", "HbA1c", "glicemia", "TG", "HDL", "acido_urico",
          "emoglobina", "diabete", "ipertensione"]


def jinuo_comuni():
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    X = pd.DataFrame({
        "eta": df["Age"], "female": kidney.is_female(df).astype(float), "BMI": df["BMI"], "girovita": df["waist1"],
        "PAS": df["Bpsys"], "PAD": df["Bpdia"], "HbA1c": df["HbA1c"], "glicemia": df["FPG"], "TG": df["TG"],
        "HDL": df["HDL"], "acido_urico": df["SUA"], "emoglobina": df["HGB"],
        "diabete": df["DM"].eq(1).astype(float), "ipertensione": df["HypertenHis"].fillna(0).eq(1).astype(float)})
    return X[COMUNI].astype(float), kidney.target(df).to_numpy(), p0.load_scores("phase_a/main/random_forest", len(df))[1]


def nhanes_comuni(d):
    X = pd.DataFrame({
        "eta": d["RIDAGEYR"], "female": d["female"].astype(float), "BMI": d["BMXBMI"], "girovita": d["BMXWAIST"],
        "PAS": d["SBP"], "PAD": d["DBP"], "HbA1c": d["LBXGH"], "glicemia": d["LBXGLU"] / 18.016,
        "TG": d["LBXTR"] / 88.57, "HDL": d["LBDHDD"] / 38.67, "acido_urico": d["LBXSUA"] * 59.48,
        "emoglobina": d["LBXHGB"] * 10, "diabete": d["diab_noto"], "ipertensione": d["iper_nota"]})
    return X[COMUNI].astype(float)


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=20, n_jobs=-1, random_state=SEED)


def analisi_6b():
    Xj, yj, fold_j = jinuo_comuni()
    # riferimento: AUROC interna Jinuo con le 14 variabili (OOF sugli stessi fold)
    oof = np.zeros(len(yj))
    for k in np.unique(fold_j):
        tr, te = fold_j != k, fold_j == k
        oof[te] = rf().fit(Xj[tr].to_numpy(), yj[tr]).predict_proba(Xj[te].to_numpy())[:, 1]
    rng = np.random.default_rng(SEED)
    auc_int = (roc_auc_score(yj, oof), *boot_auc(yj, oof, rng))
    # modello finale Jinuo, applicato una volta a NHANES
    model = rf().fit(Xj.to_numpy(), yj)
    d, _ = p4.load()
    Xn = nhanes_comuni(d)
    s_tr = model.predict_proba(Xn.to_numpy())[:, 1]
    y = d["y"].to_numpy()
    # riferimento: riaddestrato su NHANES (OOF, come punto 4c ma con le 14 variabili comuni)
    s_re = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xn, y):
        s_re[te] = rf().fit(Xn.iloc[tr].to_numpy(), y[tr]).predict_proba(Xn.iloc[te].to_numpy())[:, 1]
    rule = (d["diab_noto"].eq(1) | d["iper_nota"].eq(1)).to_numpy()
    zero = np.zeros(len(d), int)
    tiebreak = np.random.default_rng(SEED).permutation(len(d))
    rows = [{"analisi": "6B NHANES", "budget": "regola diabete o ipertensione noti", "strategia": "regola",
             "testati": int(rule.sum()), "positivi": int((rule & (y == 1)).sum()), "diff": 0,
             "ic_basso": np.nan, "ic_alto": np.nan, "auroc": np.nan, "auroc_ic": "", "verita": "KDIGO"}]
    for nome, s in [("modello Jinuo applicato senza riaddestramento", s_tr),
                    ("modello riaddestrato su NHANES (riferimento)", s_re)]:
        a = (roc_auc_score(y, s), *boot_auc(y, s, rng))
        sel = p0.select_at_parity(s, zero, rule, tiebreak)
        for verita, yy in [("KDIGO", y), ("corretta per i muscoli", d["y_std"].to_numpy())]:
            c = p0.counts(sel, rule, yy)
            lo, hi = p0.boot_diff(sel, rule, yy, np.random.default_rng(SEED))
            rows.append({"analisi": "6B NHANES", "budget": "regola diabete o ipertensione noti", "strategia": nome,
                         "testati": c["testati"], "positivi": c["pos_modello"], "diff": c["pos_modello"] - c["pos_regola"],
                         "ic_basso": lo, "ic_alto": hi, "auroc": a[0], "auroc_ic": f"{a[1]:.3f}–{a[2]:.3f}",
                         "verita": verita, "pos_regola_stessa_verita": c["pos_regola"]})
    cal = {"rischio_medio_predetto_trasportato": float(s_tr.mean()), "prevalenza_osservata_nhanes": float(y.mean()),
           "prevalenza_jinuo_training": float(yj.mean()), "auroc_interna_jinuo_14var": auc_int[0],
           "auroc_interna_jinuo_ic": f"{auc_int[1]:.3f}–{auc_int[2]:.3f}"}
    return pd.DataFrame(rows), pd.DataFrame([cal])


# ------------------------------------------------------------------ figura
def figura(a, b, cal):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    col = {"regola": "#8a8a8a", "k-means": "#7aa6d8", "autoencoder": "#a9c4e6", "Random": "#2f6db5",
           "modello Jinuo": "#2f6db5", "modello riaddestrato": "#a9c4e6"}
    short = {"regola": "Regola", "k-means": "Non sup.\nk-means", "autoencoder": "Non sup.\nautoencoder",
             "Random": "Supervisionato\n(RF)", "modello Jinuo": "Modello Jinuo\n(senza riaddestr.)",
             "modello riaddestrato": "Riaddestrato\nsu NHANES"}

    def bars(ax, g, top):
        x, xt, lab = 0, [], []
        for bud, gb in g.groupby("budget", sort=False):
            start = x
            for _, r in gb.iterrows():
                key = next(k for k in col if r.strategia.startswith(k))
                ax.bar(x, r.positivi, 0.8, color=col[key])
                t = f"{r.positivi}" if r.strategia == "regola" else \
                    f"{r.positivi}\n{r['diff']:+d}\n({r.ic_basso:+.0f}/{r.ic_alto:+.0f})\nAUROC {r.auroc:.2f}"
                ax.text(x, r.positivi + top * 0.01, t, ha="center", va="bottom", fontsize=7.5)
                xt.append(x)
                lab.append(short[key])
                x += 1
            ax.text((start + x - 1) / 2, top * 1.45, f"{bud.split(' (')[0]}: {int(gb.testati.iloc[0])} testati",
                    ha="center", fontsize=9, color="#3a3a3a")
            x += 0.8
        ax.set_xticks(xt)
        ax.set_xticklabels(lab, fontsize=8)
        ax.set_ylim(0, top * 1.55)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    bars(axes[0], a, a.positivi.max())
    axes[0].set_title("A · Jinuo: l'AI senza etichette contro le linee guida", fontsize=10)
    axes[0].set_ylabel("Malati KDIGO trovati")
    bk = b[(b.strategia == "regola") | (b.verita == "KDIGO")]
    bars(axes[1], bk, bk.positivi.max())
    axes[1].set_title("B · NHANES: validazione esterna del modello Jinuo", fontsize=10)
    fig.text(0.5, 0.01, "Sopra le barre: malati trovati, differenza dalla regola con IC 95%, AUROC. "
             f"AUROC interna Jinuo con le 14 variabili comuni: {cal.auroc_interna_jinuo_14var[0]:.2f} "
             f"({cal.auroc_interna_jinuo_ic[0]}).", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig1_nonsup_e_validazione.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    a = analisi_6a()
    a.to_csv(OUT / "P6A_nonsupervisionato_vs_regola.csv", index=False)
    b, cal = analisi_6b()
    b.to_csv(OUT / "P6B_validazione_esterna.csv", index=False)
    cal.to_csv(OUT / "P6B_calibrazione_e_riferimenti.csv", index=False)
    figura(a, b, cal)
    shutil.copy(OUT / "fig1_nonsup_e_validazione.png", FIGDIR / "fig1_nonsup_e_validazione.png")
    pd.set_option("display.width", 250)
    print(a.round(3).to_string(index=False))
    print(b.round(3).to_string(index=False))
    print(cal.round(3).T.to_string())


if __name__ == "__main__":
    main()
