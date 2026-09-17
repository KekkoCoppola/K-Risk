"""Figure introduttive sul dataset completo."""
import matplotlib.pyplot as plt
import numpy as np

from src.analytics.plotting import BLUE, GRAY, ORANGE, label_bars, save
from src.data.load import DIABETES, TARGET, binary_target, load_labeled, load_raw

SECTION = "dataset"
UNKNOWN_CODE = 9


def target_stages(df):
    counts = df[TARGET].value_counts(dropna=False)
    labels = ["0\nnessun danno", "3\n≥30 mg", "4\n≥300 mg", "5\nuremia", "mancante"]
    values = [counts.get(0, 0), counts.get(3, 0), counts.get(4, 0), counts.get(5, 0),
              int(df[TARGET].isna().sum())]
    colors = [BLUE, ORANGE, ORANGE, ORANGE, GRAY]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    label_bars(ax, bars, [f"{v:,}".replace(",", ".") for v in values])
    ax.set_yscale("log")
    ax.set_ylabel("soggetti (scala logaritmica)")
    ax.set_title(f"Distribuzione degli stadi {TARGET}")
    ax.grid(axis="y")
    save(fig, SECTION, "01_target_stages")


def target_by_diabetes(df):
    dn = binary_target(df) == 1
    dm = df[DIABETES] == 1

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))

    groups = ["non diabetici", "diabetici"]
    totals = [int((~dm).sum()), int(dm.sum())]
    positives = [int((dn & ~dm).sum()), int((dn & dm).sum())]
    prevalence = [100 * p / t for p, t in zip(positives, totals)]
    bars = left.bar(groups, prevalence, color=[BLUE, ORANGE], width=0.5)
    label_bars(left, bars, [f"{p:.1f}%\n{k}/{t}" for p, k, t in zip(prevalence, positives, totals)],
               inside=True)
    left.axhline(100 * dn.mean(), color=GRAY, linestyle="--", linewidth=1,
                 label=f"intera coorte ({100 * dn.mean():.2f}%)")
    left.set_ylim(0, max(prevalence) * 1.2)
    left.set_ylabel(f"prevalenza {TARGET} > 0 (%)")
    left.set_title("Prevalenza del danno renale")
    left.grid(axis="y")
    left.legend(loc="upper left")

    bars = right.barh(groups, positives, color=[BLUE, ORANGE], height=0.5)
    share = [100 * p / sum(positives) for p in positives]
    label_bars(right, bars, [f"{k}  ({s:.1f}%)" for k, s in zip(positives, share)], horizontal=True)
    right.set_xlim(0, max(positives) * 1.25)
    right.set_xlabel(f"soggetti con {TARGET} > 0")
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
    dn = binary_target(df) == 1
    bins = np.arange(15, 96, 5)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mask, color, label in [(~dn, BLUE, f"{TARGET} = 0"), (dn, ORANGE, f"{TARGET} > 0")]:
        ax.hist(df.loc[mask, "Age"], bins=bins, density=True, histtype="step",
                linewidth=2, color=color, label=f"{label} (n={int(mask.sum())})")
    ax.set_xlabel("età (anni)")
    ax.set_ylabel("densità")
    ax.set_title("Distribuzione dell'età per classe")
    ax.grid(axis="y")
    ax.legend()
    save(fig, SECTION, "05_age_by_target")


def main():
    raw = load_raw()
    labeled = load_labeled()
    target_stages(raw)
    target_by_diabetes(labeled)
    missing_values(raw)
    unknown_codes(raw)
    age_by_target(labeled)


if __name__ == "__main__":
    main()
