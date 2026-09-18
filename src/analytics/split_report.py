"""Figure di controllo sullo split train/test."""
import matplotlib.pyplot as plt
import numpy as np

from src.analytics.plotting import (BLUE, GRAY, ORANGE, TEXT_SECONDARY,
                                    label_bars, save)
from src.config import CONFIG
from src.data.kidney import LEVELS, kdigo_level, target
from src.data.load import DIABETES, load_eligible
from src.data.split import TEST_SIZE, balance_table, load_split, split_dataset

SECTION = "split"
MIN_EVENTS = 100
SMD_THRESHOLD = 0.1
SEVERE = "molto alto"


def prevalence(train, test):
    metrics = {
        "positivi": lambda d: target(d) == 1,
        "diabetici": lambda d: d[DIABETES] == 1,
        "diabetici positivi": lambda d: (target(d) == 1) & (d[DIABETES] == 1),
    }
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, d, color) in enumerate([("train", train, BLUE), ("test", test, ORANGE)]):
        masks = [f(d) for f in metrics.values()]
        values = [100 * m.mean() for m in masks]
        bars = ax.bar(x + (i - 0.5) * width, values, width - 0.02, color=color,
                      label=f"{name} (n={len(d)})")
        label_bars(ax, bars, [f"{v:.2f}%\n({int(m.sum())})" for v, m in zip(values, masks)])
    ax.set_xticks(x, list(metrics))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.set_ylabel("% sul totale del set")
    ax.set_title("Proporzioni in training e test set")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    save(fig, SECTION, "01_prevalence_train_test")


def smd_plot(train, test):
    table = balance_table(train, test, CONFIG["analytics"]["balance_columns"])
    table = table.sort_values()

    fig, ax = plt.subplots(figsize=(7, 0.3 * len(table) + 1.2))
    ax.hlines(table.index, 0, table.values, color=BLUE, linewidth=2)
    ax.plot(table.values, table.index, "o", color=BLUE, markersize=6)
    ax.axvline(SMD_THRESHOLD, color=ORANGE, linestyle="--", linewidth=1.5)
    ax.annotate(f"soglia {SMD_THRESHOLD}", (SMD_THRESHOLD, len(table) - 0.5),
                xytext=(-4, 0), textcoords="offset points", ha="right",
                fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, SMD_THRESHOLD * 1.3)
    ax.set_xlabel("differenza media standardizzata (SMD)")
    ax.set_title("Bilanciamento train vs test")
    ax.grid(axis="x")
    save(fig, SECTION, "02_smd_balance")


def kdigo_levels(train, test):
    counts = {name: kdigo_level(d).value_counts().reindex(LEVELS)
              for name, d in [("train", train), ("test", test)]}
    share = [100 * counts["test"][level] / (counts["train"][level] + counts["test"][level])
             for level in LEVELS]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(LEVELS, share, color=BLUE, width=0.55)
    label_bars(ax, bars,
               [f"{s:.1f}%\ntrain {counts['train'][level]} / test {counts['test'][level]}"
                for s, level in zip(share, LEVELS)])
    ax.axhline(100 * TEST_SIZE, color=GRAY, linestyle="--", linewidth=1,
               label=f"quota attesa ({100 * TEST_SIZE:.0f}%)")
    ax.set_ylim(0, 45)
    ax.set_ylabel("% del livello finita nel test set")
    ax.set_title("Ripartizione dei livelli KDIGO fra training e test")
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    save(fig, SECTION, "03_kdigo_levels_train_test")


def simulate(df, runs, stratify):
    positives, severe = [], []
    y = target(df) == 1
    level = kdigo_level(df) == SEVERE
    for seed in range(runs):
        _, test = split_dataset(df, seed=seed, stratify=stratify)
        positives.append(int(y[test.index].sum()))
        severe.append(int(level[test.index].sum()))
    return np.array(positives), np.array(severe)


def stratification_simulation(df, test):
    runs = CONFIG["analytics"]["simulation_runs"]
    random_positives, _ = simulate(df, runs, stratify=None)
    _, target_only_severe = simulate(df, runs, stratify=["target"])
    chosen_positives = int((target(test) == 1).sum())
    chosen_severe = int((kdigo_level(test) == SEVERE).sum())

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    bins = np.arange(random_positives.min(), random_positives.max() + 2) - 0.5
    left.hist(random_positives, bins=bins, color=GRAY)
    left.axvline(MIN_EVENTS, color=ORANGE, linestyle="--", linewidth=1.5,
                 label=f"minimo consigliato ({MIN_EVENTS})")
    left.axvline(chosen_positives, color=BLUE, linewidth=2,
                 label=f"split stratificato ({chosen_positives})")
    below = 100 * (random_positives < MIN_EVENTS).mean()
    left.set_xlabel("positivi nel test set")
    left.set_ylabel("numero di split")
    left.set_title(f"Split casuali non stratificati ({runs})")
    left.legend(loc="upper right", title=f"{below:.1f}% degli split sotto soglia",
                title_fontsize=8)
    left.set_ylim(0, left.get_ylim()[1] * 1.4)
    left.grid(axis="y")

    bins = np.arange(target_only_severe.min(), target_only_severe.max() + 2) - 0.5
    right.hist(target_only_severe, bins=bins, color=GRAY)
    right.axvline(chosen_severe, color=BLUE, linewidth=2,
                  label=f"stratificato KDIGO x DM ({chosen_severe})")
    right.set_xlabel(f"casi '{SEVERE}' nel test set")
    right.set_ylabel("numero di split")
    right.set_title(f"Split stratificati solo sul target ({runs})")
    right.legend(loc="upper right")
    right.set_ylim(0, right.get_ylim()[1] * 1.25)
    right.grid(axis="y")

    save(fig, SECTION, "04_stratification_simulation")


def test_size_events(df):
    sizes = [0.20, 0.25, 0.30]
    y = target(df) == 1
    test_events, train_events = [], []
    for size in sizes:
        train, test = split_dataset(df, test_size=size)
        test_events.append(int(y[test.index].sum()))
        train_events.append(int(y[train.index].sum()))

    labels = [f"{round(100 * (1 - s))}/{round(100 * s)}" for s in sizes]
    x = np.arange(len(sizes))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x - width / 2, train_events, width - 0.02, color=BLUE, label="training")
    label_bars(ax, bars, train_events, inside=True)
    bars = ax.bar(x + width / 2, test_events, width - 0.02, color=ORANGE, label="test")
    label_bars(ax, bars, test_events, inside=True)
    ax.axhline(MIN_EVENTS, color=GRAY, linestyle="--", linewidth=1, zorder=0.5,
               label=f"minimo consigliato nel test ({MIN_EVENTS})")
    ax.set_xticks(x, labels)
    ax.get_xticklabels()[sizes.index(TEST_SIZE)].set_fontweight("bold")
    ax.set_xlabel("proporzione training/test")
    ax.set_ylabel("positivi")
    ax.set_ylim(0, max(train_events) * 1.15)
    ax.set_title("Positivi al variare della proporzione di split")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    save(fig, SECTION, "05_test_size_events")


def main():
    df = load_eligible()
    train, test = load_split()
    prevalence(train, test)
    smd_plot(train, test)
    kdigo_levels(train, test)
    stratification_simulation(df, test)
    test_size_events(df)


if __name__ == "__main__":
    main()
