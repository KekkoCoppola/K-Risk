"""Figure della Fase C: sottogruppo diabetico (domanda 6).

Richiede le tabelle di `python -m src.models.phase_c` (analytics/phase_c/evaluation/).
Colore = gruppo; marcatore pieno = "nessuna correzione", vuoto = pesi per livello (descrittivo).
Nella figura 01 la PR-AUC ha una linea di riferimento alla prevalenza di ciascun gruppo: fra
gruppi con prevalenza diversa non e' confrontabile (Matos et al. 2026; Van Calster et al. 2025).
Il test set non viene letto.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.analytics.phase_a_report import MODELS, NAMES
from src.analytics.plotting import BLUE, GRAY, ORANGE, SURFACE, TEXT_SECONDARY, save
from src.models.phase_c import (OUTPUT as TABLES, PRIMARY, REFERENCE, SECONDARY, SEVERE_LABEL,
                                SUBGROUP, TECHNIQUES)

SECTION = "phase_c"
GROUPS = [SUBGROUP, REFERENCE, "tutti"]
COLORS = {SUBGROUP: ORANGE, REFERENCE: BLUE, "tutti": GRAY}
TECH_NAMES = {"none": "nessuna correzione", "level_weight": "pesi per livello 1:2:3 (descrittivo)"}
OFFSETS = dict(zip(GROUPS, np.linspace(0.22, -0.22, len(GROUPS))))
# due bracci, sfalsati per non sovrapporsi: pieno = principale, vuoto = descrittivo
NUDGE = {PRIMARY: 0.055, SECONDARY: -0.055}
YPOS = {model: i for i, model in enumerate(reversed(MODELS))}


def load():
    tables = {path.stem: pd.read_csv(path) for path in TABLES.glob("*.csv")}
    if not tables:
        raise FileNotFoundError(f"{TABLES}: lanciare prima python -m src.models.phase_c")
    return tables


def models_axis(ax):
    ax.set_yticks(list(YPOS.values()), [NAMES[model] for model in YPOS])
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    for i in range(len(MODELS) - 1):
        ax.axhline(i + 0.5, color=SURFACE, linewidth=6)  # separa le righe dei modelli


def position(row):
    return YPOS[row["model"]] + OFFSETS[row["group"]] + NUDGE[row["technique"]]


def dots(ax, table, value, low=None, high=None):
    """Un punto per modello, gruppo e tecnica; vuoto per il braccio descrittivo."""
    models_axis(ax)
    rows = table[table["model"].isin(MODELS) & table["group"].isin(GROUPS)]
    for _, r in rows.iterrows():
        y0, color = position(r), COLORS[r["group"]]
        if low is not None and np.isfinite(r[low]):
            ax.hlines(y0, r[low], r[high], color=color, linewidth=1.5)
        ax.plot(r[value], y0, "o", markersize=6, markeredgewidth=1.5, color=color,
                markerfacecolor=color if r["technique"] == PRIMARY else SURFACE)
    ax.grid(axis="x")


def legend(fig, extra=()):
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[g], markersize=6, label=g)
               for g in GROUPS]
    handles.append(Line2D([], [], marker="o", linestyle="", color=TEXT_SECONDARY,
                          markerfacecolor=SURFACE, markersize=6, markeredgewidth=1.5,
                          label=TECH_NAMES[TECHNIQUES[1]]))
    handles += list(extra)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0))


def figure_01_discrimination(tables):
    """AUC e PR-AUC per gruppo. La PR-AUC va letta rispetto alla propria prevalenza: la linea
    tratteggiata di ciascun gruppo e' il valore del classificatore di maggioranza."""
    table = tables["q1_discrimination"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    dots(axes[0], table, "auc", "auc_low", "auc_high")
    axes[0].axvline(0.5, color=TEXT_SECONDARY, linestyle=":", linewidth=1.2)
    axes[0].set_title("AUC: confrontabile fra gruppi")
    axes[0].set_xlabel("AUC, IC 95% di DeLong")

    dots(axes[1], table, "pr_auc", "pr_auc_low", "pr_auc_high")
    prevalence = table.groupby("group")["prevalence"].first()
    for group in GROUPS:
        axes[1].axvline(prevalence[group], color=COLORS[group], linestyle="--", linewidth=1.2)
    axes[1].set_title("PR-AUC: NON confrontabile fra gruppi")
    axes[1].set_xlabel("PR-AUC, IC 95% logit; tratteggio = prevalenza del gruppo")

    reference = Line2D([], [], linestyle="--", color=TEXT_SECONDARY, label="prevalenza del gruppo")
    legend(fig, extra=[reference])
    save(fig, SECTION, "01_discrimination_diabetici", rect=(0, 0.08, 1, 1))


def figure_02_severe_cases(tables):
    """Casi gravi riconosciuti alla soglia globale fissa, accanto alla quota di soggetti da
    testare: fra i diabetici la sensibilita' e' massima solo perche' il modello segnala quasi
    tutti, quindi non aggiunge informazione rispetto a "testare tutti"."""
    severe = tables["q3_sensitivity"].query("level == @SEVERE_LABEL")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

    dots(axes[0], severe, "sensitivity", "low", "high")
    axes[0].set_title("casi gravi riconosciuti (alto + molto alto)")
    axes[0].set_xlabel("sensibilita' sui casi gravi, IC 95% di Wilson")
    axes[0].set_xlim(0.6, 1.06)
    for _, r in severe[(severe["technique"] == PRIMARY) & severe["model"].isin(MODELS)].iterrows():
        axes[0].annotate(f"{r['detected']}/{r['n']}", (r["sensitivity"], position(r)),
                         xytext=(7, 0), textcoords="offset points", va="center", fontsize=8,
                         color=COLORS[r["group"]])

    dots(axes[1], tables["q1_operating_points"], "alert_rate")
    axes[1].set_title("quota di soggetti da testare, stessa soglia")
    axes[1].set_xlabel("quota di allerta")
    axes[1].set_xlim(0, 1.05)

    legend(fig)
    save(fig, SECTION, "02_severe_cases", rect=(0, 0.08, 1, 1))


def run():
    tables = load()
    figure_01_discrimination(tables)
    figure_02_severe_cases(tables)
    print(f"figure in {TABLES.parent}", flush=True)


if __name__ == "__main__":
    run()
