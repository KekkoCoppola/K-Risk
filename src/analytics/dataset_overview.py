"""Figure introduttive sul dataset completo."""
import matplotlib.pyplot as plt
import numpy as np

from src.analytics.plotting import (BLUE, GRAY, LIGHT_BLUE, ORANGE, RED, TEXT,
                                    TEXT_SECONDARY, YELLOW, label_bars, save)
from src.data.kidney import (ACR_CATEGORIES, GFR_CATEGORIES, LEVELS, RISK_MAP,
                             acr_category, egfr, gfr_category, target)
from src.data.load import DIABETES, load_eligible, load_raw

SECTION = "dataset"
UNKNOWN_CODE = 9
LEVEL_COLORS = {"basso": LIGHT_BLUE, "moderato": YELLOW, "alto": ORANGE, "molto alto": RED}


def kdigo_heatmap(df):
    rows = list(reversed(GFR_CATEGORIES))
    counts = np.zeros((len(rows), len(ACR_CATEGORIES)), dtype=int)
    colors = np.empty(counts.shape, dtype=object)
    g = gfr_category(df).astype(str)
    a = acr_category(df).astype(str)
    for i, gc in enumerate(rows):
        for j, ac in enumerate(ACR_CATEGORIES):
            counts[i, j] = int(((g == gc) & (a == ac)).sum())
            colors[i, j] = LEVEL_COLORS[RISK_MAP.get((gc, ac), "molto alto")]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.02), 0.96, 0.96,
                                       color=colors[i, j]))
            ax.text(j + 0.5, i + 0.5, f"{counts[i, j]:,}".replace(",", "."),
                    ha="center", va="center", fontsize=10, color=TEXT)
    ax.set_xlim(0, counts.shape[1])
    ax.set_ylim(counts.shape[0], 0)
    ax.set_xticks(np.arange(counts.shape[1]) + 0.5,
                  ["A1\n< 30", "A2\n30–300", "A3\n> 300"])
    ax.set_yticks(np.arange(counts.shape[0]) + 0.5,
                  ["G1\n≥ 90", "G2\n60–89", "G3a\n45–59", "G3b\n30–44",
                   "G4\n15–29", "G5\n< 15"])
    ax.set_xlabel("ACR (mg/g)")
    ax.set_ylabel("eGFR (ml/min/1,73 m²)")
    ax.set_title("Soggetti per cella della heatmap KDIGO")
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LEVEL_COLORS[level]) for level in LEVELS]
    ax.legend(handles, LEVELS, loc="center left", bbox_to_anchor=(1.02, 0.5),
              title="rischio KDIGO", title_fontsize=9)
    save(fig, SECTION, "01_kdigo_heatmap")


def target_by_diabetes(df):
    y = target(df) == 1
    dm = df[DIABETES] == 1

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))

    groups = ["non diabetici", "diabetici"]
    totals = [int((~dm).sum()), int(dm.sum())]
    positives = [int((y & ~dm).sum()), int((y & dm).sum())]
    prevalence = [100 * p / t for p, t in zip(positives, totals)]
    bars = left.bar(groups, prevalence, color=[BLUE, ORANGE], width=0.5)
    label_bars(left, bars,
               [f"{p:.1f}%\n{k}/{t}" for p, k, t in zip(prevalence, positives, totals)],
               inside=True)
    left.axhline(100 * y.mean(), color=GRAY, linestyle="--", linewidth=1,
                 label=f"intera coorte ({100 * y.mean():.2f}%)")
    left.set_ylim(0, max(prevalence) * 1.2)
    left.set_ylabel("prevalenza del target (%)")
    left.set_title("Marcatori di malattia renale")
    left.grid(axis="y")
    left.legend(loc="upper left")

    bars = right.barh(groups, positives, color=[BLUE, ORANGE], height=0.5)
    share = [100 * p / sum(positives) for p in positives]
    label_bars(right, bars, [f"{k}  ({s:.1f}%)" for k, s in zip(positives, share)],
               horizontal=True)
    right.set_xlim(0, max(positives) * 1.25)
    right.set_xlabel("soggetti positivi")
    right.set_title("Composizione dei casi positivi")
    right.grid(axis="x")

    save(fig, SECTION, "02_target_by_diabetes")


def missing_values(df, top=25):
    missing = (100 * df.isna().mean()).sort_values(ascending=False)
    missing = missing[missing > 0].head(top)[::-1]

    fig, ax = plt.subplots(figsize=(7, 0.28 * len(missing) + 1.2))
    bars = ax.barh(missing.index, missing.values, color=BLUE, height=0.7)
    label_bars(ax, bars, [f"{v:.1f}%" for v in missing.values], horizontal=True)
    ax.set_xlim(0, missing.max() * 1.15)
    ax.set_xlabel("valori mancanti (%)")
    ax.set_title(f"Variabili con più valori mancanti (prime {len(missing)})")
    ax.grid(axis="x")
    save(fig, SECTION, "03_missing_values")


def unknown_codes(df):
    """Variabili categoriche che usano 9 come 'sconosciuto'."""
    counts = {}
    for c in df.columns:
        values = set(df[c].dropna().unique())
        if UNKNOWN_CODE in values and values - {UNKNOWN_CODE} <= {0, 1, 2, 3}:
            counts[c] = int((df[c] == UNKNOWN_CODE).sum())
    counts = dict(sorted(counts.items(), key=lambda kv: kv[1]))

    fig, ax = plt.subplots(figsize=(7, 0.28 * len(counts) + 1.2))
    bars = ax.barh(list(counts), list(counts.values()), color=ORANGE, height=0.7)
    label_bars(ax, bars, [str(v) for v in counts.values()], horizontal=True)
    ax.set_xlim(0, max(counts.values()) * 1.15)
    ax.set_xlabel(f"occorrenze del codice {UNKNOWN_CODE} (sconosciuto)")
    ax.set_title(f"Variabili categoriche con codice {UNKNOWN_CODE}")
    ax.grid(axis="x")
    save(fig, SECTION, "04_unknown_code_9")


def age_by_target(df):
    y = target(df) == 1
    bins = np.arange(15, 96, 5)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mask, color, label in [(~y, BLUE, "negativi"), (y, ORANGE, "positivi")]:
        ax.hist(df.loc[mask, "Age"], bins=bins, density=True, histtype="step",
                linewidth=2, color=color, label=f"{label} (n={int(mask.sum())})")
    ax.set_xlabel("età (anni)")
    ax.set_ylabel("densità")
    ax.set_title("Distribuzione dell'età per classe")
    ax.grid(axis="y")
    ax.legend()
    save(fig, SECTION, "05_age_by_target")


def egfr_comparison(df):
    """Confronto fra la colonna GFR del dataset e l'eGFR ricalcolato."""
    estimated = egfr(df)
    original = df["GFR"]
    mask = original.notna()
    limit = 320

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(original[mask], estimated[mask], s=6, alpha=0.25, color=BLUE,
               edgecolors="none")
    ax.plot([0, limit], [0, limit], color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax.axhline(60, color=ORANGE, linewidth=1.2, linestyle="--")
    ax.axvline(60, color=ORANGE, linewidth=1.2, linestyle="--")
    disagree = int(((original < 60) != (estimated < 60))[mask].sum())
    ax.annotate(f"soglia 60: {disagree} soggetti classificati in modo\ndiverso dalle due misure",
                (limit * 0.97, 72), ha="right", va="bottom", fontsize=8,
                color=TEXT_SECONDARY)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("GFR fornito dal dataset")
    ax.set_ylabel("eGFR ricalcolato (CKD-EPI 2021)")
    ax.set_title("La colonna GFR non segue le equazioni standard")
    save(fig, SECTION, "06_egfr_comparison")


def main():
    raw = load_raw()
    eligible = load_eligible()
    kdigo_heatmap(eligible)
    target_by_diabetes(eligible)
    missing_values(raw)
    unknown_codes(raw)
    age_by_target(eligible)
    egfr_comparison(eligible)


if __name__ == "__main__":
    main()
