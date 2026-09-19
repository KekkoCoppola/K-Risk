"""Figure della valutazione della Fase A: previsioni out-of-fold del training, set main.

Richiede le tabelle di `python -m src.models.evaluation` (analytics/phase_a/evaluation/).
Il test set non viene letto. Colore fisso per modello in tutte le figure; il classificatore di
maggioranza compare solo come riferimento (grigio tratteggiato).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_curve

from src.analytics.plotting import (AQUA, BLUE, GRAY, ORANGE, RED, SURFACE, TEXT, TEXT_SECONDARY,
                                    YELLOW, save)
from src.config import CONFIG
from src.data.kidney import LEVELS, kdigo_level, target
from src.data.split import PROCESSED
from src.models.evaluation import OUTPUT as TABLES, PREDICTIONS, SETTINGS

SECTION = "phase_a"
SEED = CONFIG["seed"]
MODELS = ["lr_scored", "lr_penalized", "random_forest", "xgboost"]
NAMES = {"lr_scored": "logistica SCORED", "lr_penalized": "logistica penalizzata",
         "random_forest": "Random Forest", "xgboost": "XGBoost"}
COLORS = dict(zip(MODELS, [BLUE, ORANGE, AQUA, YELLOW]))
# rampa sequenziale blu (dal chiaro allo scuro) per le quote nelle heatmap
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blu", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
TARGET = SETTINGS["target_sensitivity"]
CONSEQUENCE = CONFIG["features"]["consequence"]
INTERPRETATION = TABLES.parent / "interpretation"


def load():
    tables = {path.stem: pd.read_csv(path) for path in TABLES.glob("*.csv")}
    if not tables:
        raise FileNotFoundError(f"{TABLES}: lanciare prima python -m src.models.evaluation")
    train = pd.read_csv(PROCESSED / "train.csv")
    return tables, target(train).to_numpy(), kdigo_level(train), pd.read_csv(PREDICTIONS)


def predictions(oof, model, feature_set="main"):
    """Previsioni out-of-fold di un modello, nell'ordine delle righe di train.csv."""
    g = oof[(oof["feature_set"] == feature_set) & (oof["model"] == model)]
    return g.set_index("row")["probability"].sort_index().to_numpy()


def main_rows(table):
    return table[table["feature_set"] == "main"]


def curves(tables, y, oof):
    q1 = main_rows(tables["q1_discrimination"]).set_index("model")
    prevalence = y.mean()
    fig, (roc, pr) = plt.subplots(1, 2, figsize=(11, 5))
    for model in MODELS:
        p = predictions(oof, model)
        fpr, tpr, _ = roc_curve(y, p)
        roc.plot(fpr, tpr, color=COLORS[model], linewidth=1.8,
                 label=f"{NAMES[model]}  AUC {q1.loc[model, 'auc']:.3f}")
        precision, recall, _ = precision_recall_curve(y, p)
        pr.plot(recall, precision, color=COLORS[model], linewidth=1.8,
                label=f"{NAMES[model]}  PR-AUC {q1.loc[model, 'pr_auc']:.3f}")
    roc.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="maggioranza  AUC 0.500")
    roc.axhline(TARGET, color=TEXT_SECONDARY, linestyle=":", linewidth=1)
    roc.annotate(f"sensibilità {TARGET:.2f} (soglia scelta)", (0, TARGET), xytext=(4, 4),
                 textcoords="offset points", ha="left", fontsize=8, color=TEXT_SECONDARY)
    roc.set(xlabel="1 - specificità", ylabel="sensibilità", xlim=(0, 1), ylim=(0, 1.01))
    roc.set_title("Curva ROC")
    pr.axhline(prevalence, color=GRAY, linestyle="--", linewidth=1,
               label=f"maggioranza = prevalenza {prevalence:.3f}")
    pr.set(xlabel="recall (sensibilità)", ylabel="precision", xlim=(0, 1), ylim=(0, 1.01))
    pr.set_title("Curva precision-recall")
    for ax in (roc, pr):
        ax.grid()
        ax.legend(loc="upper right" if ax is pr else "lower right")
    save(fig, SECTION, "01_roc_pr")


def discrimination(tables):
    q1 = tables["q1_discrimination"].set_index(["feature_set", "model"])
    prevalence = q1.loc[("main", "dummy"), "prevalence"]
    ypos = {model: i for i, model in enumerate(reversed(MODELS))}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, metric, label, reference, reference_label in [
            (axes[0], "auc", "AUC", 0.5, "maggioranza 0.5"),
            (axes[1], "pr_auc", "PR-AUC", prevalence, f"prevalenza {prevalence:.3f}")]:
        for feature_set, offset, filled in [("main", 0.15, True), ("no_consequence", -0.15, False)]:
            for model in MODELS:
                row = q1.loc[(feature_set, model)]
                y0 = ypos[model] + offset
                ax.hlines(y0, row[f"{metric}_low"], row[f"{metric}_high"], color=COLORS[model], linewidth=1.8)
                ax.plot(row[metric], y0, "o", markersize=7, markeredgewidth=1.8, color=COLORS[model],
                        markerfacecolor=COLORS[model] if filled else SURFACE)
                ax.annotate(f"{row[metric]:.3f}", (row[f"{metric}_high"], y0), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.axvline(reference, color=GRAY, linestyle="--", linewidth=1)
        ax.annotate(reference_label, (reference, len(MODELS) - 0.45), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xlabel(f"{label} con IC 95%")
        ax.set_title(label)
        ax.grid(axis="x")
    axes[0].set_xlim(0.45, 0.8)
    axes[1].set_xlim(0.05, 0.35)
    axes[0].set_yticks(list(ypos.values()), [NAMES[m] for m in ypos])
    axes[0].set_ylim(-0.6, len(MODELS) - 0.2)
    handles = [Line2D([], [], marker="o", linestyle="", color=TEXT_SECONDARY, markersize=7,
                      label="main (74 feature)"),
               Line2D([], [], marker="o", linestyle="", color=TEXT_SECONDARY, markersize=7,
                      markerfacecolor=SURFACE, markeredgewidth=1.8,
                      label="no_consequence (senza variabili-conseguenza)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
    save(fig, SECTION, "02_discrimination_ci", rect=(0, 0.08, 1, 1))


def operating_points(tables, y, oof):
    """Sensibilità contro quota di soggetti da testare: il costo di ogni soglia."""
    alert = main_rows(tables["q1_operating_points"])
    alert = alert[np.isclose(alert["target_sensitivity"], TARGET)].set_index("model")["alert_rate"]
    positives = y.sum()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model in MODELS:
        fpr, tpr, _ = roc_curve(y, predictions(oof, model))
        tested = (tpr * positives + fpr * (len(y) - positives)) / len(y)
        ax.plot(tpr, 100 * tested, color=COLORS[model], linewidth=1.8,
                label=f"{NAMES[model]}: {100 * alert[model]:.1f}% da testare a {TARGET:.2f}")
    ax.plot([0, 1], [0, 100], color=GRAY, linestyle="--", linewidth=1,
            label="senza modello (soggetti scelti a caso)")
    ax.axvline(TARGET, color=TEXT_SECONDARY, linestyle=":", linewidth=1)
    ax.set(xlim=(0.5, 1), ylim=(0, 100), xlabel="sensibilità (quota di positivi riconosciuti)",
           ylabel="% di soggetti da testare")
    ax.set_title("Quanti soggetti testare per ogni livello di sensibilità")
    ax.grid()
    ax.legend(loc="upper left")
    save(fig, SECTION, "03_operating_points")


def probability_by_level(tables, levels, oof):
    q2 = main_rows(tables["q2_levels"])
    trend = main_rows(tables["q2_trend"]).set_index("model")
    codes = np.asarray(pd.Categorical(levels, LEVELS).codes)
    counts = np.bincount(codes, minlength=len(LEVELS))
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        p = predictions(oof, model)
        ax.boxplot([p[codes == i] for i in range(len(LEVELS))], positions=range(len(LEVELS)),
                   widths=0.5, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": COLORS[model] + "40", "edgecolor": COLORS[model]},
                   whiskerprops={"color": COLORS[model]}, capprops={"color": COLORS[model]},
                   medianprops={"color": TEXT, "linewidth": 1.5})
        # livelli piccoli: ogni soggetto visibile
        for i in (2, 3):
            values = p[codes == i]
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(values)), values, s=10, color=TEXT_SECONDARY,
                       alpha=0.7, edgecolors="none", zorder=3)
        rows = q2[q2["model"] == model].set_index("level").loc[LEVELS]
        ax.errorbar(np.arange(len(LEVELS)) + 0.32, rows["mean_probability"],
                    yerr=[rows["mean_probability"] - rows["low"], rows["high"] - rows["mean_probability"]],
                    fmt="D", color=TEXT, markersize=5, elinewidth=1.5, capsize=0, zorder=4)
        t = trend.loc[model]
        ax.set_title(f"{NAMES[model]}: concordanza {t['concordance']:.2f} "
                     f"({t['concordance_low']:.2f}-{t['concordance_high']:.2f})")
        ax.set_xticks(range(len(LEVELS)), [f"{level}\nn={n}" for level, n in zip(LEVELS, counts)])
        ax.grid(axis="y")
    for ax in axes[:, 0]:
        ax.set_ylabel("probabilità stimata (out-of-fold)")
    handles = [Line2D([], [], color=TEXT, linewidth=1.5, label="mediana (scatola: quartili; baffi: 1,5 IQR)"),
               Line2D([], [], marker="D", color=TEXT, linestyle="", markersize=5,
                      label="media con IC 95% (bootstrap stratificato)"),
               Line2D([], [], marker="o", color=TEXT_SECONDARY, linestyle="", markersize=4,
                      label="singoli soggetti (alto, molto alto)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Probabilità stimata per livello KDIGO (concordanza = coppie ordinate come KDIGO)",
                 x=0.01, ha="left", fontweight="bold", color=TEXT)
    save(fig, SECTION, "04_probability_by_level", rect=(0, 0.04, 1, 1))


def sensitivity_by_level(tables):
    q3 = main_rows(tables["q3_sensitivity"])
    levels = LEVELS[1:]
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for j, model in enumerate(MODELS):
        rows = q3[q3["model"] == model].set_index("level").loc[levels]
        x = np.arange(len(levels)) + (j - (len(MODELS) - 1) / 2) * width
        ax.errorbar(x, rows["sensitivity"],
                    yerr=[rows["sensitivity"] - rows["low"], rows["high"] - rows["sensitivity"]],
                    fmt="o", color=COLORS[model], markersize=7, elinewidth=1.8, capsize=0,
                    label=NAMES[model])
        for xi, (_, r) in zip(x, rows.iterrows()):
            ax.annotate(f"{r['detected']}/{r['n']}", (xi, r["low"]), xytext=(0, -9),
                        textcoords="offset points", ha="center", fontsize=7, color=TEXT_SECONDARY)
    ax.axhline(TARGET, color=GRAY, linestyle="--", linewidth=1)
    ax.annotate(f"sensibilità complessiva fissata {TARGET:.2f}", (len(levels) - 0.5, TARGET),
                xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8, color=TEXT_SECONDARY)
    n = q3.drop_duplicates("level").set_index("level")["n"]
    ax.set_xticks(range(len(levels)), [f"{level}\nn={n[level]}" for level in levels])
    ax.set(ylim=(0.55, 1.02), ylabel="sensibilità con IC 95% (Wilson)")
    ax.set_title("Positivi riconosciuti per livello KDIGO alla soglia scelta")
    ax.grid(axis="y")
    ax.legend(loc="lower left", ncol=2)
    save(fig, SECTION, "05_sensitivity_by_level")


def bands_vs_levels(tables):
    bands = main_rows(tables["q4_bands"])
    kappa = main_rows(tables["q4_kappa"]).set_index("model")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    for ax, model in zip(axes.flat, MODELS):
        table = bands[bands["model"] == model].pivot(index="level", columns="band", values="n").loc[LEVELS]
        share = table.div(table.sum(axis=1), axis=0)
        ax.imshow(share.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(len(LEVELS)):
            for j in range(len(LEVELS)):
                value = share.iat[i, j]
                ax.text(j, i, f"{table.iat[i, j]}\n{100 * value:.0f}%", ha="center", va="center",
                        fontsize=8, color=SURFACE if value > 0.5 else TEXT)
            # diagonale: fascia e livello coincidono
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=TEXT, linewidth=1.5))
        k = kappa.loc[model]
        ax.set_title(f"{NAMES[model]}\nkappa pesato {k['kappa']:.2f} ({k['low']:.2f}-{k['high']:.2f})")
        ax.set_xticks(range(len(LEVELS)), [f"fascia {b}" for b in range(1, len(LEVELS) + 1)])
        ax.set_yticks(range(len(LEVELS)), LEVELS)
        ax.set(xlabel="fascia del modello", ylabel="livello KDIGO")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Fasce del modello contro livelli KDIGO (colore e % = quota del livello nella fascia; "
                 "riquadri = accordo)", x=0.01, ha="left", fontweight="bold", color=TEXT, fontsize=11)
    save(fig, SECTION, "06_bands_vs_levels")


def differences(ax, rows, labels, colors, metric_label):
    """Differenze con IC 95% di Nadeau & Bengio, una riga per confronto."""
    ypos = np.arange(len(rows))[::-1]
    for y0, (_, r), color in zip(ypos, rows.iterrows(), colors):
        ax.hlines(y0, r["low"], r["high"], color=color, linewidth=1.8)
        ax.plot(r["difference"], y0, "o", color=color, markersize=7)
        p = r["p_holm"] if "p_holm" in r else r["p_value"]
        ax.annotate(f"p = {p:.2f}", (r["high"], y0), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    limit = 1.5 * np.abs(rows[["low", "high"]].to_numpy()).max()
    ax.set_xlim(-limit, limit)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel(f"differenza di {metric_label} (IC 95% Nadeau-Bengio)")
    ax.grid(axis="x")


def model_comparison(tables):
    comp = tables["comparison_models"]
    comp = comp[(comp["feature_set"] == "main") & (comp["model_a"] != "dummy") & (comp["model_b"] != "dummy")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, metric, label in [(axes[0], "pr_auc", "PR-AUC"), (axes[1], "auc", "AUC")]:
        rows = comp[comp["metric"] == metric]
        labels = [f"{NAMES[a]} - {NAMES[b]}" for a, b in zip(rows["model_a"], rows["model_b"])]
        differences(ax, rows, labels, [TEXT_SECONDARY] * len(rows), label)
        ax.set_title(f"{label}{' (metrica primaria)' if metric == 'pr_auc' else ''}")
    fig.suptitle("Differenze fra modelli sui 5 fold esterni (p corretti con Holm sulle 10 coppie, maggioranza inclusa)", x=0.01, ha="left",
                 fontweight="bold", color=TEXT, fontsize=11)
    save(fig, SECTION, "07_model_comparison")


def sensitivity_analysis(tables):
    comp = tables["comparison_sets"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, metric, label in [(axes[0], "pr_auc", "PR-AUC"), (axes[1], "auc", "AUC")]:
        rows = comp[comp["metric"] == metric].set_index("model").loc[MODELS]
        differences(ax, rows, [NAMES[m] for m in MODELS], [COLORS[m] for m in MODELS], label)
        ax.set_title(f"{label}: no_consequence - main")
    fig.suptitle(f"Senza le variabili-conseguenza ({', '.join(CONSEQUENCE)})", x=0.01, ha="left",
                 fontweight="bold", color=TEXT, fontsize=11)
    save(fig, SECTION, "08_no_consequence")


def feature_label(feature, per=None):
    """Nome della variabile; * = variabile-conseguenza della malattia renale."""
    mark = " *" if feature in CONSEQUENCE else ""
    return f"{feature}{mark}" if per is None else f"{feature}{mark} (per {per})"


def odds_ratio_plot(odds):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [1, 1.25]})
    scored = odds[odds["model"] == "lr_scored"]
    main = scored[scored["feature_set"] == "main"]
    ypos = {f: i for i, f in enumerate(reversed(main["feature"].tolist()))}
    for feature_set, offset, filled in [("main", 0.12, True), ("no_consequence", -0.12, False)]:
        for _, r in scored[scored["feature_set"] == feature_set].iterrows():
            y0 = ypos[r["feature"]] + offset
            left.hlines(y0, r["low"], r["high"], color=BLUE, linewidth=1.8)
            left.plot(r["odds_ratio"], y0, "o", markersize=7, markeredgewidth=1.8, color=BLUE,
                      markerfacecolor=BLUE if filled else SURFACE)
            left.annotate(f"{r['odds_ratio']:.2f}", (r["high"], y0), xytext=(4, 0), textcoords="offset points",
                          va="center", fontsize=8, color=TEXT_SECONDARY)
    per = main.set_index("feature")["per"]
    left.set_yticks(list(ypos.values()), [feature_label(f, per[f]) for f in ypos])
    left.set_title("Logistica SCORED: odds ratio con IC 95% (Wald)")
    left.legend(handles=[Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7, label="main"),
                         Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=7,
                                markerfacecolor=SURFACE, markeredgewidth=1.8, label="no_consequence")],
                loc="lower right")

    penalized = odds[(odds["model"] == "lr_penalized") & (odds["feature_set"] == "main")].head(15)
    ypos = np.arange(len(penalized))[::-1]
    right.hlines(ypos, 1, penalized["odds_ratio"], color=ORANGE, linewidth=1.8)
    right.plot(penalized["odds_ratio"], ypos, "o", markersize=7, color=ORANGE)
    right.set_yticks(ypos, [feature_label(f, p) for f, p in zip(penalized["feature"], penalized["per"])])
    right.set_title("Logistica penalizzata: prime 15 variabili (senza IC)")
    for ax in (left, right):
        ax.set_xscale("log")
        ticks = [t for t in [0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4] if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.axvline(1, color=GRAY, linestyle="--", linewidth=1)
        ax.set_xlabel("odds ratio (scala logaritmica)")
        ax.grid(axis="x")
    fig.text(0.01, 0.01, "per 1 DS: numeriche standardizzate; per unità: da 0 a 1 (Gender: da maschio a femmina); "
             "* variabile-conseguenza", fontsize=8, color=TEXT_SECONDARY)
    save(fig, SECTION, "09_odds_ratios", rect=(0, 0.03, 1, 1))


def shap_importance_plot(importance):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, model, scale in [(axes[0], "random_forest", "probabilità"), (axes[1], "xgboost", "logit")]:
        rows = importance[(importance["model"] == model) & (importance["feature_set"] == "main")].head(15)
        ypos = np.arange(len(rows))[::-1]
        ax.barh(ypos, rows["mean_abs_shap"], color=COLORS[model], height=0.6)
        ax.set_yticks(ypos, [feature_label(f) for f in rows["feature"]])
        ax.set_xlabel(f"media del valore assoluto SHAP (scala {scale})")
        ax.set_title(f"{NAMES[model]}: prime 15 variabili")
        ax.grid(axis="x")
    fig.text(0.01, 0.01, "* variabile-conseguenza della malattia renale", fontsize=8, color=TEXT_SECONDARY)
    save(fig, SECTION, "10_shap_importance", rect=(0, 0.03, 1, 1))


def shap_beeswarm_plot(importance, folder):
    """Ogni punto è un soggetto: posizione = contributo alla previsione, colore = valore della
    variabile (percentile nel training)."""
    diverging = LinearSegmentedColormap.from_list("valore", [BLUE, GRAY, RED])
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    for ax, model, scale in [(axes[0], "random_forest", "probabilità"), (axes[1], "xgboost", "logit")]:
        with np.load(folder / f"shap_{model}_main.npz", allow_pickle=False) as data:
            values, X, columns = data["values"], data["X"], list(data["columns"])
        top = importance[(importance["model"] == model) & (importance["feature_set"] == "main")]["feature"].head(15)
        for row, feature in enumerate(reversed(top.tolist())):
            j = columns.index(feature)
            x = values[:, j]
            # scostamento verticale proporzionale alla densità dei punti (sciame)
            counts, edges = np.histogram(x, bins=60)
            density = counts[np.clip(np.digitize(x, edges) - 1, 0, len(counts) - 1)] / counts.max()
            y0 = row + rng.uniform(-1, 1, len(x)) * 0.38 * density
            points = ax.scatter(x, y0, c=stats.rankdata(X[:, j]) / len(x), cmap=diverging, vmin=0, vmax=1,
                                s=4, alpha=0.6, edgecolors="none", rasterized=True)
        ax.axvline(0, color=GRAY, linewidth=1)
        ax.set_yticks(range(len(top)), [feature_label(f) for f in reversed(top.tolist())])
        ax.set_xlabel(f"valore SHAP (scala {scale}): > 0 alza il rischio stimato")
        ax.set_title(NAMES[model])
        ax.grid(axis="x")
    bar = fig.colorbar(points, ax=axes, fraction=0.02, pad=0.02, ticks=[0, 1])
    bar.ax.set_yticklabels(["basso", "alto"])
    bar.set_label("valore della variabile (percentile)")
    fig.text(0.01, 0.01, "* variabile-conseguenza della malattia renale", fontsize=8, color=TEXT_SECONDARY)
    save(fig, SECTION, "11_shap_beeswarm", rect=(0, 0.03, 0.9, 1))


def run():
    tables, y, levels, oof = load()
    curves(tables, y, oof)
    discrimination(tables)
    operating_points(tables, y, oof)
    probability_by_level(tables, levels, oof)
    sensitivity_by_level(tables)
    bands_vs_levels(tables)
    model_comparison(tables)
    sensitivity_analysis(tables)
    if (INTERPRETATION / "odds_ratios.csv").exists():
        odds = pd.read_csv(INTERPRETATION / "odds_ratios.csv")
        importance = pd.read_csv(INTERPRETATION / "shap_importance.csv")
        odds_ratio_plot(odds)
        shap_importance_plot(importance)
        shap_beeswarm_plot(importance, INTERPRETATION)
    else:
        print(f"{INTERPRETATION}: lanciare python -m src.models.interpretation per le figure 09-11")


if __name__ == "__main__":
    run()
