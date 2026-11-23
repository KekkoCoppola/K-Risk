"""Punto 7c: cause di morte fra i "sani" KDIGO (ACR < 30, eGFR >= 60) in NHANES. PREDITTIVO.
Punteggio non supervisionato (primario) e RF (confronto) del punto 7; Cox per causa specifica,
altre cause censurate, aggiustato per età e sesso. Controllo negativo: incidenti.
Piano: analisi_impatto/04_esperimenti/confine_ai/piano_punto7.md, sezione 7c.
"""
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import punto4_nhanes as p4  # noqa: E402
import punto7_modello_sotto_30 as p7  # noqa: E402

OUT, FIGDIR = p7.OUT, p7.FIGDIR
CAUSE = {"Malattie renali (9)": [9], "Cuore (1)": [1], "Cerebrovascolari (5)": [5], "Diabete (7)": [7],
         "Tumori (2)": [2], "Incidenti (4) - controllo negativo": [4], "Tutte le cause": None}


def main():
    d, _ = p4.load()
    sc = p7.punteggi(d)
    neg = d[(d["URDACT"] < 30) & (d["eGFR"] >= 60)].copy()
    rows = []
    for k, v in sc.items():
        s = v[((d["URDACT"] < 30) & (d["eGFR"] >= 60)).to_numpy()]
        z = (s - s.mean()) / s.std()
        for causa, codici in CAUSE.items():
            e = neg["morte"].to_numpy() if codici is None else \
                (neg["mortstat"].eq(1) & neg["ucod"].isin(codici)).astype(int).to_numpy()
            m = p4.cox(np.column_stack([z, neg["RIDAGEYR"], neg["female"].astype(float)]), neg["t"].to_numpy(), e)
            rows.append({"modello": k, "causa": causa, "n": len(neg), "morti": int(e.sum()),
                         "HR_per_DS": np.exp(m["b"][0]), "ic_basso": np.exp(m["b"][0] - 1.96 * m["se"][0]),
                         "ic_alto": np.exp(m["b"][0] + 1.96 * m["se"][0]),
                         "solo_descrittivo": bool(e.sum() < 30)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "P7c_cause_di_morte.csv", index=False)

    # figura: forest del punteggio non supervisionato
    g = t[t.modello == p7.NS].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, r in g.iterrows():
        col = "#c46b3c" if "controllo" in r.causa else ("#9a9a9a" if r.solo_descrittivo else "#2f6db5")
        ax.plot([r.ic_basso, r.ic_alto], [i, i], color=col, lw=2)
        ax.plot(r.HR_per_DS, i, "o", color=col, ms=8, mec="white", mew=1.5)
        ax.text(max(g.ic_alto.max(), 2.2) * 1.05, i, f"{r.HR_per_DS:.2f} ({r.ic_basso:.2f}–{r.ic_alto:.2f}), {r.morti} morti",
                va="center", fontsize=8.5)
    ax.axvline(1, color="#3a3a3a", lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 6])
    ax.set_xticklabels(["0,5", "1", "2", "4", "6"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.causa, fontsize=9)
    ax.set_xlabel("HR per DS del punteggio dell'AI non supervisionata (età e sesso)")
    ax.set_title("\"Sani\" per KDIGO (ACR < 30, eGFR ≥ 60): di cosa muoiono le persone che l'AI segnala?\n"
                 "Arancio: controllo negativo (incidenti). Grigio: meno di 30 morti, solo descrittivo", fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cause_di_morte.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(OUT / "fig2_cause_di_morte.png", FIGDIR / "fig2_cause_di_morte.png")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
