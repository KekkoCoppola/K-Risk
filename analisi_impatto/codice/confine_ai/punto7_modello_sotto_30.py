"""Punto 7: l'AI vede il rischio sotto la soglia KDIGO 30? PREDITTIVO.
Primario: NON supervisionato (k-means k=4 sul training Jinuo, 14 variabili comuni, nessuna etichetta),
applicato a NHANES senza riaddestramento. Secondario: RF supervisionato trasportabile del punto 6B.
Piano fissato prima: analisi_impatto/04_esperimenti/confine_ai/piano_punto7.md
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import punto4_nhanes as p4  # noqa: E402
import punto6_nonsup_e_validazione as p6  # noqa: E402

ROOT, SEED = p4.ROOT, p4.SEED
OUT = ROOT / "analisi_impatto/04_esperimenti/confine_ai/punto7"
FIGDIR = ROOT / "analytics/soglia_ai/punto7"
FASCE = [0, 5, 10, 15, 20, 25, 30, 60, 300, np.inf]
NS = "non supervisionato (k-means, nessuna etichetta)"
SUP = "supervisionato (Random Forest, confronto)"


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = r @ r / (len(y) - X.shape[1])
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))


def punteggi(d):
    Xj, yj, _ = p6.jinuo_comuni()
    Xn = p6.nhanes_comuni(d)
    med = Xj.median()
    A = Xj.fillna(med)
    mu, sd = A.mean(), A.std().replace(0, 1)
    Zj, Zn = ((A - mu) / sd).to_numpy(), ((Xn.fillna(med) - mu) / sd).to_numpy()
    km = KMeans(4, n_init=10, random_state=SEED).fit(Zj)
    big = int(np.bincount(km.labels_).argmax())
    ns = np.log(np.linalg.norm(Zn - km.cluster_centers_[big], axis=1) + 1e-6)
    p = np.clip(p6.rf().fit(Xj.to_numpy(), yj).predict_proba(Xn.to_numpy())[:, 1], 1e-4, 1 - 1e-4)
    return {NS: ns, SUP: np.log(p / (1 - p))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    d, _ = p4.load()
    sc = punteggi(d)
    for k, v in sc.items():
        d[k] = (v - v.mean()) / v.std()          # z su tutto NHANES
    negm = (d["URDACT"] < 30) & (d["eGFR"] >= 60)
    neg = d[negm].copy()
    lab = pd.cut(d["URDACT"], FASCE, right=False)

    grad, fasce, cox, quint = [], [], [], []
    for k in sc:
        b, se = ols(np.column_stack([neg["lacr"], neg["RIDAGEYR"], neg["female"].astype(float)]), neg[k].to_numpy())
        grad.append({"modello": k, "n_sani_kdigo": len(neg), "pendenza_z_per_raddoppio_acr": b[1] * np.log(2),
                     "ic_basso": (b[1] - 1.96 * se[1]) * np.log(2), "ic_alto": (b[1] + 1.96 * se[1]) * np.log(2)})
        for iv, g in d.groupby(lab, observed=True):
            m, e = g[k].mean(), 1.96 * g[k].std() / np.sqrt(len(g))
            fasce.append({"modello": k, "fascia_acr": f"{iv.left:g}–{iv.right:g}", "n": len(g), "z_medio": m,
                          "ic_basso": m - e, "ic_alto": m + e, "morti": int(g["morte"].sum())})
        z = (neg[k] - neg[k].mean()) / neg[k].std()
        for esito in ["morte", "morte_cv"]:
            t, e = neg["t"].to_numpy(), neg[esito].to_numpy()
            for nome, extra in [("età, sesso", []), ("età, sesso, log ACR", [neg["lacr"]])]:
                m = p4.cox(np.column_stack([z, neg["RIDAGEYR"], neg["female"].astype(float)] + extra), t, e)
                cox.append({"modello": k, "esito": esito, "aggiustamento": nome, "n": len(neg), "eventi": int(e.sum()),
                            "HR_per_DS": np.exp(m["b"][0]), "ic_basso": np.exp(m["b"][0] - 1.96 * m["se"][0]),
                            "ic_alto": np.exp(m["b"][0] + 1.96 * m["se"][0])})
        q = pd.qcut(neg[k], 5, labels=[f"Q{i}" for i in range(1, 6)])
        qq = neg.groupby(q, observed=True).agg(n=("morte", "size"), morti=("morte", "sum"),
                                               eta_mediana=("RIDAGEYR", "median")).reset_index()
        qq["mortalita_pct"] = qq["morti"] / qq["n"] * 100
        qq.insert(0, "modello", k)
        quint.append(qq)
    grad, fasce, cox, quint = pd.DataFrame(grad), pd.DataFrame(fasce), pd.DataFrame(cox), pd.concat(quint)
    grad.to_csv(OUT / "P7a_gradiente_sotto_30.csv", index=False)
    fasce.to_csv(OUT / "P7a_punteggio_per_fascia_acr.csv", index=False)
    cox.to_csv(OUT / "P7b_mortalita_sani_kdigo.csv", index=False)
    quint.to_csv(OUT / "P7b_quintili.csv", index=False)

    # ---- figura (primario: non supervisionato) ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    for k, col, off, mk in [(NS, "#2f6db5", -0.12, "o"), (SUP, "#9a9a9a", 0.12, "s")]:
        f = fasce[fasce.modello == k].reset_index(drop=True)
        x = np.arange(len(f)) + off
        ax.errorbar(x, f.z_medio, yerr=[f.z_medio - f.ic_basso, f.ic_alto - f.z_medio], fmt=mk, ms=7, color=col,
                    elinewidth=1, label=k)
    ax.axvline(5.5, color="#3a3a3a", ls="--", lw=1.5)
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.text(5.6, ax.get_ylim()[1] * 0.9, "KDIGO 30", fontsize=9)
    ax.set_xticks(np.arange(len(FASCE) - 1))
    ax.set_xticklabels(fasce[fasce.modello == NS].fascia_acr, fontsize=8)
    ax.set_xlabel("Fascia di ACR (mg/g)")
    ax.set_ylabel("Punteggio dell'AI (deviazioni standard)")
    g0 = grad[grad.modello == NS].iloc[0]
    ax.set_title(f"A · Il punteggio dell'AI sale già sotto 30\nnon supervisionato, sotto 30 a parità di età e sesso: "
                 f"{g0.pendenza_z_per_raddoppio_acr:+.3f} DS per raddoppio (IC {g0.ic_basso:+.3f}/{g0.ic_alto:+.3f})",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax2 = axes[1]
    qn = quint[quint.modello == NS].reset_index(drop=True)
    ax2.bar(np.arange(5), qn["mortalita_pct"], 0.7, color="#2f6db5")
    for i, r in qn.iterrows():
        ax2.text(i, r.mortalita_pct + 0.3, f"{int(r.morti)} morti\nsu {int(r.n)}", ha="center", fontsize=8)
    hr = cox[(cox.modello == NS) & (cox.esito == "morte") & (cox.aggiustamento == "età, sesso")].iloc[0]
    ax2.set_xticks(np.arange(5))
    ax2.set_xticklabels(["Q1\n(più vicini\nalla norma)", "Q2", "Q3", "Q4", "Q5\n(più lontani\ndalla norma)"], fontsize=8)
    ax2.set_ylabel("Morti nel follow-up (%)")
    ax2.set_ylim(0, qn["mortalita_pct"].max() * 1.3)
    ax2.set_title(f"B · \"Sani\" per KDIGO (ACR < 30, eGFR ≥ 60) per punteggio dell'AI non supervisionata\n"
                  f"HR per DS a parità di età e sesso: {hr.HR_per_DS:.2f} ({hr.ic_basso:.2f}–{hr.ic_alto:.2f})",
                  fontsize=9.5)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("NHANES 2005–2018: AI addestrata sui Jinuo e applicata senza riaddestramento", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig1_ai_sotto_30.png", dpi=200)
    plt.close(fig)
    shutil.copy(OUT / "fig1_ai_sotto_30.png", FIGDIR / "fig1_ai_sotto_30.png")

    pd.set_option("display.width", 250)
    print(grad.round(4).to_string(index=False))
    print(fasce.round(3).to_string(index=False))
    print(cox.round(3).to_string(index=False))
    print(quint.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
