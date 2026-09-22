"""Figure della Fase D: ricerca del tetto di prestazione (analisi post-hoc, esplorativa).

Richiede le tabelle di `python -m src.models.phase_d` (analytics/phase_d/). La figura 03 si calcola
da train.csv: le giornate di raccolta del solo training. Il test set non viene letto.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.phase_a_report import COLORS, NAMES
from src.analytics.plotting import AQUA, BLUE, GRAY, ORANGE, TEXT_SECONDARY, save
from src.data.split import PROCESSED
from src.models.phase_d import COLUMNS, DIAGNOSTICS, OUTPUT as TABLES, SETTINGS, THRESHOLDS

SECTION = "phase_d"
CANDIDATE_NAMES = {
    "ensemble_mean": "ensemble (media dei 3)",
    "xgboost_native_nan": "XGBoost, NaN nativi",
    "xgboost_extended": "XGBoost, spazio allargato",
    "target_decomposition": "target scomposto",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "ebm": "EBM",
    "tabpfn": "TabPFN",
    "xgboost_no_pruner": "XGBoost senza pruner",
    "lr_all_scaled": "logistica, tutto standardizzato",
    "positive_control": "controllo positivo (+ albumina urinaria)",
}
LABELS = {**NAMES, **CANDIDATE_NAMES}


def load(name):
    path = TABLES / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}: lanciare prima python -m src.models.phase_d")
    return pd.read_csv(path)


def figure_01_candidates():
    """Sinistra: AUROC out-of-fold con IC di DeLong. Destra: differenza di AUROC contro il miglior
    riferimento con IC di Nadeau-Bengio e soglia di rilevanza della regola di decisione."""
    metrics, comparison = load("discrimination"), load("comparison")
    table = metrics[metrics["role"] != "diagnostica"].iloc[::-1].reset_index(drop=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 0.42 * len(table) + 1.8),
                                      gridspec_kw={"width_ratios": [1.1, 1]})
    for i, r in table.iterrows():
        color = BLUE if r["role"] == "riferimento" else ORANGE
        left.hlines(i, r["auc_low"], r["auc_high"], color=color, linewidth=1.5)
        left.plot(r["auc"], i, "o", color=color, markersize=6)
        left.annotate(f"{r['auc']:.3f}", (r["auc_high"], i), xytext=(4, 0), textcoords="offset points",
                      va="center", fontsize=8, color=TEXT_SECONDARY)
    left.set_yticks(range(len(table)), [LABELS.get(m, m) for m in table["model"]])
    control = metrics[metrics["model"] == "positive_control"]
    title = "AUROC out-of-fold"
    if len(control):
        title += f" (controllo positivo: {control['auc'].iloc[0]:.3f})"
    left.set_title(title)
    left.set_xlabel("AUROC (IC 95% DeLong); blu = Fase A, arancio = candidati")
    left.grid(axis="x")

    comparison = comparison.iloc[::-1].reset_index(drop=True)
    for i, r in comparison.iterrows():
        right.hlines(i, r["low"], r["high"], color=ORANGE, linewidth=1.5)
        right.plot(r["difference"], i, "o", color=ORANGE, markersize=6)
    right.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    right.axvline(SETTINGS["min_difference"], color=AQUA, linestyle=":", linewidth=1.5)
    right.annotate(f"soglia di rilevanza +{SETTINGS['min_difference']}",
                   (SETTINGS["min_difference"], -0.45), xytext=(4, 0),
                   textcoords="offset points", fontsize=8, color=AQUA)
    right.set_yticks(range(len(comparison)), [LABELS.get(m, m) for m in comparison["candidate"]])
    reference = comparison["reference"].iloc[0] if len(comparison) else ""
    right.set_title(f"Differenza contro {LABELS.get(reference, reference)}")
    right.set_xlabel("differenza di AUROC (IC 95% Nadeau-Bengio)")
    right.grid(axis="x")
    save(fig, SECTION, "01_candidates")


def figure_02_learning_curve():
    table = load("learning_curve")
    by_fold = table.groupby(["model", "fraction", "fold"])[["auc", "n_train"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model, g in by_fold.groupby("model"):
        summary = g.groupby("fraction").agg(auc=("auc", "mean"), sd=("auc", "std"), n=("n_train", "mean"))
        ax.errorbar(summary["n"], summary["auc"], yerr=summary["sd"], marker="o", capsize=3,
                    color=COLORS.get(model, GRAY), label=NAMES.get(model, model))
    ax.set_xlabel("soggetti nel training del fold")
    ax.set_ylabel("AUROC sul fold esterno (media ± DS fra i fold)")
    ax.set_title("Curva di apprendimento")
    ax.legend(loc="lower right")
    ax.grid()
    save(fig, SECTION, "02_learning_curve")


def figure_03_urine_creatinine():
    """Per giornata di raccolta (training): mediana della creatinina urinaria contro quota di
    albuminuria e mediana dell'albumina urinaria."""
    spec = DIAGNOSTICS["label_quality"]
    train = pd.read_csv(PROCESSED / "train.csv")
    albuminuria = train[COLUMNS["acr"]] >= THRESHOLDS["acr_threshold"]
    days = pd.DataFrame({"day": train[spec["day"]], "ucre": train[spec["urine_creatinine"]],
                         "umalb": train["UmALB"], "albuminuria": albuminuria}).groupby("day").agg(
        n=("ucre", "size"), ucre=("ucre", "median"), umalb=("umalb", "median"),
        share=("albuminuria", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, column, label, color in [(axes[0], "share", "quota con ACR ≥ 30 mg/g (%)", ORANGE),
                                     (axes[1], "umalb", "mediana dell'albumina urinaria (UmALB)", BLUE)]:
        values = 100 * days[column] if column == "share" else days[column]
        ax.scatter(days["ucre"], values, s=days["n"] / 2, color=color, alpha=0.7, edgecolor="white")
        rho, p = stats.spearmanr(days["ucre"], days[column])
        ax.set_title(f"rho di Spearman = {rho:.2f} (p = {p:.1g})")
        ax.set_xlabel("mediana della creatinina urinaria (UCRE) della giornata")
        ax.set_ylabel(label)
        ax.grid()
    fig.suptitle(f"{len(days)} giornate di raccolta del training (area del punto = soggetti)",
                 x=0.01, ha="left", fontsize=10, color=TEXT_SECONDARY)
    save(fig, SECTION, "03_urine_creatinine_by_day", rect=(0, 0, 1, 0.95))


def figure_04_label_quality():
    table = load("label_quality")
    table = table[table["model"].isin(NAMES)]
    analyses = ["componente: eGFR < 60", "componente: solo albuminuria", "tutte le giornate",
                "giornate UCRE basso", "altre giornate"]
    models = list(dict.fromkeys(table["model"]))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(models)
    for k, model in enumerate(models):
        g = table[table["model"] == model].set_index("analysis").reindex(analyses)
        x = np.arange(len(analyses)) + (k - (len(models) - 1) / 2) * width
        ax.bar(x, g["auc"], width, color=COLORS.get(model, GRAY), label=NAMES.get(model, model))
    ax.set_xticks(range(len(analyses)), [a.replace("componente: ", "") for a in analyses], rotation=15)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("AUROC out-of-fold")
    ax.set_title("AUROC per componente del target e per gruppo di giornate")
    ax.legend(loc="upper right", ncol=len(models))
    ax.grid(axis="y")
    save(fig, SECTION, "04_label_quality")


def run():
    figure_01_candidates()
    figure_02_learning_curve()
    figure_03_urine_creatinine()
    figure_04_label_quality()


if __name__ == "__main__":
    run()
