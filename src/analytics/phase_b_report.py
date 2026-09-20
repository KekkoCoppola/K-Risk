"""Figure della Fase B: tecniche di bilanciamento contro "nessuna correzione" (previsioni
out-of-fold reali). Richiede le tabelle di `python -m src.models.evaluation_b`.
Colore = modello (come nella Fase A); marcatore vuoto = tecnica esplorativa (CTGAN).
Il test set non viene letto.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.analytics.phase_a_report import COLORS, MODELS, NAMES
from src.analytics.plotting import GRAY, SURFACE, TEXT_SECONDARY, save
from src.models.evaluation_b import CONFIRMATORY, NAIVE, OUTPUT as TABLES, TARGET

SECTION = "phase_b"
TECHNIQUES = ["class_weight", "level_weight", "undersampling", "oversampling", "smote_nc",
              "smote_nc_level", "ctgan", "ctgan_level"]
TECH_NAMES = {"none": "nessuna correzione", "class_weight": "pesi di classe",
              "level_weight": "pesi per livello 1:2:3", "undersampling": "undersampling",
              "oversampling": "oversampling", "smote_nc": "SMOTE-NC", "smote_nc_level": "SMOTE-NC per livello",
              "ctgan": "CTGAN (esplorativo)", "ctgan_level": "CTGAN per livello (esplorativo)"}
OFFSETS = dict(zip(MODELS, np.linspace(0.27, -0.27, len(MODELS))))


def load():
    tables = {path.stem: pd.read_csv(path) for path in TABLES.glob("*.csv")}
    if not tables:
        raise FileNotFoundError(f"{TABLES}: lanciare prima python -m src.models.evaluation_b")
    return tables


def techniques_axis(ax, techniques):
    ypos = {t: i for i, t in enumerate(reversed(techniques))}
    ax.set_yticks(list(ypos.values()), [TECH_NAMES[t] for t in ypos])
    ax.set_ylim(-0.6, len(techniques) - 0.4)
    for i in range(len(techniques) - 1):
        ax.axhline(i + 0.5, color=SURFACE, linewidth=6)  # separa le righe delle tecniche
    return ypos


def dots(ax, table, techniques, x, low=None, high=None):
    """Un punto per tecnica e modello (con intervallo se dato); vuoto se esplorativa."""
    ypos = techniques_axis(ax, techniques)
    for _, r in table[table["technique"].isin(techniques) & table["model"].isin(MODELS)].iterrows():
        y0 = ypos[r["technique"]] + OFFSETS[r["model"]]
        color = COLORS[r["model"]]
        if low is not None:
            ax.hlines(y0, r[low], r[high], color=color, linewidth=1.6)
        filled = r["technique"] in CONFIRMATORY or r["technique"] == "none"
        ax.plot(r[x], y0, "o", markersize=6.5, markeredgewidth=1.6, color=color,
                markerfacecolor=color if filled else SURFACE)
    ax.grid(axis="x")


def model_legend(fig, extra=()):
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[m], markersize=6.5, label=NAMES[m])
               for m in MODELS]
    handles.append(Line2D([], [], marker="o", linestyle="", color=TEXT_SECONDARY, markerfacecolor=SURFACE,
                          markersize=6.5, label="tecnica esplorativa"))
    handles += list(extra)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0))


def primary(tables):
    fig, ax = plt.subplots(figsize=(10, 7))
    dots(ax, tables["primary_comparison"], TECHNIQUES, "difference", "low", "high")
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel("differenza di sensibilità sui casi gravi rispetto a nessuna correzione (IC 95% Newcombe)")
    ax.set_title(f"Esito primario: casi gravi riconosciuti a sensibilità complessiva {TARGET:.2f}")
    model_legend(fig)
    save(fig, SECTION, "01_primary_endpoint", rect=(0, 0.06, 1, 1))


def discrimination(tables):
    comparison = tables["comparison_techniques"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, metric, label in [(axes[0], "pr_auc", "PR-AUC"), (axes[1], "auc", "AUC")]:
        dots(ax, comparison[comparison["metric"] == metric], TECHNIQUES, "difference", "low", "high")
        ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
        ax.set_xlabel(f"differenza di {label} rispetto a nessuna correzione (IC 95% Nadeau-Bengio)")
        ax.set_title(label)
    model_legend(fig)
    save(fig, SECTION, "02_discrimination_vs_none", rect=(0, 0.06, 1, 1))


def sensitivity_by_level(tables):
    q3 = tables["q3_sensitivity"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    for ax, level in zip(axes, ["moderato", "alto", "molto alto"]):
        rows = q3[q3["level"] == level]
        dots(ax, rows, ["none", *TECHNIQUES], "sensitivity", "low", "high")
        ax.axvline(TARGET, color=GRAY, linestyle="--", linewidth=1)
        ax.set_xlabel("sensibilità con IC 95% (Wilson)")
        ax.set_title(f"{level} (n = {int(rows['n'].iloc[0])})")
    model_legend(fig)
    save(fig, SECTION, "03_sensitivity_by_level", rect=(0, 0.06, 1, 1))


def calibration(tables):
    table = tables["calibration"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, metric, reference, label in [(axes[0], "intercept", 0, "intercetta"), (axes[1], "slope", 1, "pendenza")]:
        ypos = techniques_axis(ax, ["none", *TECHNIQUES])
        for _, r in table[table["model"].isin(MODELS)].iterrows():
            raw = r["probabilities"] == "grezza"
            ax.plot(r[metric], ypos[r["technique"]] + OFFSETS[r["model"]], "o" if raw else "D", markersize=6,
                    color=COLORS[r["model"]], markerfacecolor=SURFACE if raw else COLORS[r["model"]],
                    markeredgewidth=1.5)
        ax.axvline(reference, color=GRAY, linestyle="--", linewidth=1)
        ax.set_xlabel(f"{label} ({reference} = calibrata)")
        ax.set_title(f"Calibrazione: {label}")
        ax.grid(axis="x")
    handles = [Line2D([], [], marker="s", linestyle="", color=COLORS[m], label=NAMES[m]) for m in MODELS]
    handles += [Line2D([], [], marker="o", linestyle="", color=TEXT_SECONDARY, markerfacecolor=SURFACE, label="grezza"),
                Line2D([], [], marker="D", linestyle="", color=TEXT_SECONDARY, label="ricalibrata (Platt)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0))
    save(fig, SECTION, "04_calibration", rect=(0, 0.06, 1, 1))


def naive_vs_real(tables):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    dots(axes[0], tables["naive"], ["none", *TECHNIQUES], "recall")
    axes[0].set_xlabel(f"recall alla soglia {NAIVE} (valutazione ingenua)")
    axes[0].set_title(f"Soglia {NAIVE}: il miglioramento apparente")
    dots(axes[1], tables["q1_discrimination"], ["none", *TECHNIQUES], "specificity",
         "specificity_low", "specificity_high")
    axes[1].set_xlabel(f"specificità alla soglia con sensibilità {TARGET:.2f} (IC 95% Wilson)")
    axes[1].set_title("A parità di sensibilità: il guadagno reale")
    for ax in axes:
        ax.set_xlim(left=0)
    model_legend(fig)
    save(fig, SECTION, "05_naive_vs_real", rect=(0, 0.06, 1, 1))


def kappa(tables):
    fig, ax = plt.subplots(figsize=(10, 7))
    dots(ax, tables["q4_kappa"], ["none", *TECHNIQUES], "kappa", "low", "high")
    ax.set_xlabel("kappa pesato (lineare) fra fasce e livelli KDIGO, IC 95% bootstrap")
    ax.set_title("Concordanza delle fasce con i livelli KDIGO")
    model_legend(fig)
    save(fig, SECTION, "06_kappa", rect=(0, 0.06, 1, 1))


def run():
    tables = load()
    primary(tables)
    discrimination(tables)
    sensitivity_by_level(tables)
    calibration(tables)
    naive_vs_real(tables)
    kappa(tables)


if __name__ == "__main__":
    run()
