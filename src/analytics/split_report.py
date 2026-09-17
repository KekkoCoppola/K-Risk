"""Figure di controllo sullo split train/test."""
import matplotlib.pyplot as plt
import numpy as np

from src.analytics.plotting import BLUE, GRAY, ORANGE, TEXT_SECONDARY, label_bars, save
from src.config import CONFIG
from src.data.load import DIABETES, TARGET, binary_target, load_labeled
from src.data.split import TEST_SIZE, balance_table, load_split, split_dataset

SECTION = "split"
MIN_EVENTS = 100
SMD_THRESHOLD = 0.1


def prevalence(train, test):
    metrics = {
        "DN > 0": lambda d: binary_target(d) == 1,
        "diabetici": lambda d: d[DIABETES] == 1,
        "diabetici con DN > 0": lambda d: (binary_target(d) == 1) & (d[DIABETES] == 1),
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


def simulate(df, runs, stratify):
    dn = binary_target(df) == 1
    dm = df[DIABETES] == 1
    events, diabetic_events = [], []
    for seed in range(runs):
        _, test = split_dataset(df, seed=seed, stratify=stratify)
        events.append(int(dn[test.index].sum()))
        diabetic_events.append(int((dn & dm)[test.index].sum()))
    return np.array(events), np.array(diabetic_events)


def stratification_simulation(df, test):
    runs = CONFIG["analytics"]["simulation_runs"]
    random_events, _ = simulate(df, runs, stratify=None)
    _, target_only = simulate(df, runs, stratify=[TARGET])
    chosen_events = int(binary_target(test).sum())
    chosen_diabetic = int(((binary_target(test) == 1) & (test[DIABETES] == 1)).sum())

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    bins = np.arange(random_events.min(), random_events.max() + 2) - 0.5
    left.hist(random_events, bins=bins, color=GRAY)
    left.axvline(MIN_EVENTS, color=ORANGE, linestyle="--", linewidth=1.5,
                 label=f"minimo consigliato ({MIN_EVENTS})")
    left.axvline(chosen_events, color=BLUE, linewidth=2,
                 label=f"split stratificato ({chosen_events})")
    below = 100 * (random_events < MIN_EVENTS).mean()
    left.set_xlabel("casi DN > 0 nel test set")
    left.set_ylabel("numero di split")
    left.set_title(f"Split casuali non stratificati ({runs})")
    left.legend(loc="upper right", title=f"{below:.1f}% degli split sotto soglia",
                title_fontsize=8)
    left.set_ylim(0, left.get_ylim()[1] * 1.4)
    left.grid(axis="y")

    bins = np.arange(target_only.min(), target_only.max() + 2) - 0.5
    right.hist(target_only, bins=bins, color=GRAY)
    right.axvline(chosen_diabetic, color=BLUE, linewidth=2,
                  label=f"stratificato DN x DM ({chosen_diabetic})")
    right.set_xlabel("diabetici con DN > 0 nel test set")
    right.set_ylabel("numero di split")
    right.set_title(f"Split stratificati solo su DN ({runs})")
    right.legend(loc="upper left")
    right.set_ylim(0, right.get_ylim()[1] * 1.2)
    right.grid(axis="y")

    save(fig, SECTION, "03_stratification_simulation")


def test_size_events(df):
    sizes = [0.20, 0.25, 0.30]
    dn = binary_target(df) == 1
    test_events, train_events = [], []
    for size in sizes:
        train, test = split_dataset(df, test_size=size)
        test_events.append(int(dn[test.index].sum()))
        train_events.append(int(dn[train.index].sum()))

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
    chosen = sizes.index(TEST_SIZE)
    ax.get_xticklabels()[chosen].set_fontweight("bold")
    ax.set_xlabel("proporzione training/test")
    ax.set_ylabel("casi DN > 0")
    ax.set_ylim(0, max(train_events) * 1.15)
    ax.set_title("Casi positivi al variare della proporzione di split")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    save(fig, SECTION, "04_test_size_events")


def main():
    df = load_labeled()
    train, test = load_split()
    prevalence(train, test)
    smd_plot(train, test)
    stratification_simulation(df, test)
    test_size_events(df)


if __name__ == "__main__":
    main()
