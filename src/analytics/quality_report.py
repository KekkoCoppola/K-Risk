"""Figure del blocco qualita' e utilita' clinica (analisi post-hoc, esplorativa).

Richiede le tabelle di `python -m src.models.clinical_utility` (analytics/quality/evaluation/).
Colore = modello (come nelle figure della Fase A); grigio tratteggiato = "testare tutti".
MCC, F1 e accuratezza bilanciata non compaiono in nessuna figura (protocollo, punto 4).
Il test set non viene letto.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from src.analytics.phase_a_report import COLORS, MODELS, NAMES
from src.analytics.plotting import BLUE, GRAY, ORANGE, SURFACE, TEXT, TEXT_SECONDARY, save
from src.models.clinical_utility import COST, EXTERNAL, OPERATING, OUTPUT as TABLES, TOTAL

SECTION = "quality"
SUBGROUPS = ["diabetici", "non diabetici"]
GROUP_COLORS = {"diabetici": ORANGE, "non diabetici": BLUE}
ALL_STYLE = {"color": GRAY, "linestyle": "--", "linewidth": 1.8}
NONE_STYLE = {"color": TEXT_SECONDARY, "linestyle": ":", "linewidth": 1.2}


def load(name):
    path = TABLES / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}: lanciare prima python -m src.models.clinical_utility")
    return pd.read_csv(path)


def percent_axis(ax, axis="x"):
    formatter = plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)


def threshold_axis(ax, thresholds):
    """Asse delle soglie: tacche ogni 2 punti percentuali, cosi' nessuna etichetta e' arrotondata."""
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    percent_axis(ax)
    ax.set_xlim(thresholds.min(), thresholds.max())


def italian(value, decimals=0):
    """Numero con la virgola decimale e il punto delle migliaia."""
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def mark_operating(ax):
    """Soglie 5% e 7%, usate come esempi da Bragg-Gresham et al. 2024."""
    for t in OPERATING:
        ax.axvline(t, color=TEXT_SECONDARY, linewidth=0.8, alpha=0.6)
        ax.annotate(f"{100 * t:.0f}%", (t, 1), xycoords=("data", "axes fraction"),
                    xytext=(2, -2), textcoords="offset points", va="top", fontsize=8,
                    color=TEXT_SECONDARY)


def model_legend(fig, extra=(), ncol=None):
    handles = [Line2D([], [], color=COLORS[m], linewidth=2, label=NAMES[m]) for m in MODELS]
    handles += list(extra)
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles),
               bbox_to_anchor=(0.5, 0))


def reference_handles():
    return [Line2D([], [], label="testare tutti", **ALL_STYLE),
            Line2D([], [], label="non testare nessuno", **NONE_STYLE)]


def decision_axes(ax, curve, group, ylim):
    """Net benefit ogni 100 persone per i 4 modelli, "testare tutti" e "non testare nessuno"."""
    g = curve[curve["group"] == group]
    for model in MODELS:
        m = g[g["model"] == model].sort_values("threshold")
        ax.plot(m["threshold"], 100 * m["net_benefit"], color=COLORS[model], linewidth=2)
    ref = g[g["model"] == MODELS[0]].sort_values("threshold")
    ax.plot(ref["threshold"], 100 * ref["net_benefit_all"], **ALL_STYLE)
    ax.axhline(0, **NONE_STYLE)
    mark_operating(ax)
    threshold_axis(ax, ref["threshold"])
    ax.set_ylim(*ylim)
    ax.set_xlabel("soglia di probabilità t (esami inutili accettati per caso = 1/t − 1)")
    ax.set_ylabel("net benefit: casi trovati netti ogni 100")
    ax.grid(axis="y")


def figure_01_decision_curve(curve):
    """Sinistra: decision curve su tutto il training. Destra: la stessa informazione come esami
    inutili evitati ogni 100 persone rispetto a "testare tutti" (Vickers et al. 2019)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    total = curve[curve["group"] == TOTAL]
    decision_axes(left, curve, TOTAL, ylim=(-1.0, 100 * total["net_benefit_all"].max() * 1.08))
    left.set_title(f"decision curve, tutto il training (n = {italian(total['n'].iloc[0])})")

    for model in MODELS:
        m = total[total["model"] == model].sort_values("threshold")
        right.plot(m["threshold"], m["avoided_per_100"], color=COLORS[model], linewidth=2)
    right.axhline(0, **ALL_STYLE)
    mark_operating(right)
    threshold_axis(right, total["threshold"])
    right.set_xlabel("soglia di probabilità t")
    right.set_ylabel("esami inutili evitati ogni 100 persone")
    right.set_title("rispetto a \"testare tutti\" (sopra 0 = il modello è meglio)")
    right.grid(axis="y")
    model_legend(fig, extra=reference_handles())
    save(fig, SECTION, "01_decision_curve", rect=(0, 0.07, 1, 1))


def figure_02_decision_curve_subgroups(curve):
    """Decision curve dentro ciascun gruppo, contro "testare tutti" dello stesso gruppo. Le scale
    sono diverse di proposito: il net benefit dipende dalla prevalenza e non si confronta fra
    gruppi (Matos et al. 2026)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, group in zip(axes, SUBGROUPS):
        g = curve[curve["group"] == group]
        top = 100 * g["net_benefit_all"].max()
        decision_axes(ax, curve, group, ylim=(-0.08 * top, top * 1.08))
        ax.set_title(f"{group} (n = {italian(g['n'].iloc[0])}, "
                     f"prevalenza {italian(100 * g['prevalence'].iloc[0], 1)}%)")
    model_legend(fig, extra=reference_handles())
    save(fig, SECTION, "02_decision_curve_subgroups", rect=(0, 0.07, 1, 1))


def figure_03_costs(costs):
    """Sinistra: costo del test ACR per caso trovato. Destra: costo di ogni caso in piu' che
    "testare tutti" trova rispetto al modello. Solo il costo del test: non e' costo-efficacia."""
    total = costs[costs["group"] == TOTAL]
    everyone = total["cost_per_case_all"].iloc[0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    for model in MODELS:
        m = total[total["model"] == model].sort_values("threshold")
        left.plot(m["threshold"], m["cost_per_case"], color=COLORS[model], linewidth=2)
        right.plot(m["threshold"], m["incremental_cost_per_case"], color=COLORS[model], linewidth=2)
    left.axhline(everyone, **ALL_STYLE)
    left.annotate(f"testare tutti: ${italian(everyone)}",
                  (total["threshold"].max(), everyone), xytext=(-4, 4), textcoords="offset points",
                  ha="right", fontsize=8, color=TEXT_SECONDARY)
    left.set_ylim(0, everyone * 1.15)
    left.set_title(f"costo per caso trovato (${COST['central']} a test ACR)")
    left.set_ylabel("dollari per caso trovato")
    # a soglie molto basse il modello segnala quasi tutti e il rapporto incrementale e' instabile
    right.set_ylim(0, 3000)
    right.annotate("instabile a soglie basse: il modello\nsegnala quasi tutti, i casi in più sono 0-3",
                   (0.99, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=8,
                   color=TEXT_SECONDARY)
    right.set_title("costo di ogni caso in più trovato testando tutti")
    right.set_ylabel("dollari per caso aggiuntivo")
    for ax in (left, right):
        mark_operating(ax)
        threshold_axis(ax, total["threshold"])
        ax.set_xlabel("soglia di probabilità t")
        ax.grid(axis="y")
    model_legend(fig, extra=[Line2D([], [], label="testare tutti", **ALL_STYLE)])
    save(fig, SECTION, "03_costs", rect=(0, 0.07, 1, 1))


def calibration_text(row):
    def ci(name):
        return (f"{row[name]:.2f} ({row[f'{name}_low_simple']:.2f} – "
                f"{row[f'{name}_high_simple']:.2f})")
    return f"intercetta {ci('intercept')}\npendenza {ci('slope')}\nO:E {ci('oe_ratio')}".replace(".", ",")


def histogram_axis(ax, hist, xmax, label=True):
    """Distribuzione delle probabilita' per esito: positivi sopra, negativi sotto (in quota)."""
    for outcome, sign, color in [(1, 1, TEXT), (0, -1, GRAY)]:
        h = hist[hist["outcome"] == outcome]
        ax.bar(h["bin_from"], sign * h["n"] / h["n"].sum(), width=h["bin_to"] - h["bin_from"],
               align="edge", color=color, linewidth=0)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.6)
    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    percent_axis(ax)
    if label:
        ax.set_xlabel("probabilità stimata")


def figure_04_calibration(curves, bins, hist, table):
    """Curva flessibile (spline cubica ristretta, banda al 95%), decili con IC di Wilson e, sotto,
    la distribuzione delle probabilita' per esito. Tutto il training; IC da bootstrap semplice."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), gridspec_kw={"height_ratios": [4, 1, 4, 1]})
    xmax = min(1.0, curves[curves["group"] == TOTAL]["predicted"].max() * 1.1)
    for i, model in enumerate(MODELS):
        row, col = 2 * (i // 2), i % 2
        ax, low = axes[row, col], axes[row + 1, col]
        c = curves[(curves["group"] == TOTAL) & (curves["model"] == model)]
        b = bins[(bins["group"] == TOTAL) & (bins["model"] == model)]
        ax.plot([0, xmax], [0, xmax], color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
        ax.fill_between(c["predicted"], c["low"], c["high"], color=COLORS[model], alpha=0.18,
                        linewidth=0)
        ax.plot(c["predicted"], c["observed"], color=COLORS[model], linewidth=2)
        ax.errorbar(b["predicted"], b["observed"],
                    yerr=[b["observed"] - b["low"], b["high"] - b["observed"]],
                    fmt="o", markersize=4, color=TEXT, elinewidth=0.8, capsize=0)
        r = table[(table["group"] == TOTAL) & (table["model"] == model)].iloc[0]
        ax.text(0.03, 0.97, calibration_text(r), transform=ax.transAxes, va="top", fontsize=8,
                color=TEXT_SECONDARY)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, xmax)
        percent_axis(ax, "y")
        ax.set_title(NAMES[model])
        ax.set_ylabel("proporzione osservata")
        ax.grid(True)
        ax.tick_params(labelbottom=False)
        histogram_axis(low, hist[(hist["group"] == TOTAL) & (hist["model"] == model)], xmax,
                       label=row == 2)
    handles = [Line2D([], [], color=TEXT_SECONDARY, linewidth=2, label="curva flessibile (banda 95%)"),
               Line2D([], [], marker="o", linestyle="", color=TEXT, markersize=4,
                      label="decili (IC di Wilson)"),
               Line2D([], [], color=TEXT_SECONDARY, linestyle=":", label="calibrazione perfetta"),
               Line2D([], [], color=TEXT, linewidth=6, label="positivi (sopra)"),
               Line2D([], [], color=GRAY, linewidth=6, label="negativi (sotto)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0))
    save(fig, SECTION, "04_calibration", rect=(0, 0.04, 1, 1))


def figure_05_calibration_subgroups(curves, table):
    """Curve flessibili dentro diabetici e non diabetici, per modello. Fra i diabetici 68 eventi,
    sotto i 200 suggeriti da Van Calster et al. 2019: banda larga, dichiarato nel protocollo."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharex=True, sharey=True)
    xmax = min(1.0, curves[curves["group"].isin(SUBGROUPS)]["predicted"].max() * 1.05)
    for ax, model in zip(axes.flat, MODELS):
        ax.plot([0, xmax], [0, xmax], color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
        lines = []
        for group in SUBGROUPS:
            c = curves[(curves["group"] == group) & (curves["model"] == model)]
            ax.fill_between(c["predicted"], c["low"], c["high"], color=GROUP_COLORS[group],
                            alpha=0.15, linewidth=0)
            ax.plot(c["predicted"], c["observed"], color=GROUP_COLORS[group], linewidth=2)
            r = table[(table["group"] == group) & (table["model"] == model)].iloc[0]
            lines.append(f"{group}: O:E {r['oe_ratio']:.2f} ({r['oe_ratio_low_simple']:.2f} – "
                         f"{r['oe_ratio_high_simple']:.2f})".replace(".", ","))
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8,
                color=TEXT_SECONDARY)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, xmax)
        percent_axis(ax)
        percent_axis(ax, "y")
        ax.set_title(NAMES[model])
        ax.grid(True)
    for ax in axes[1]:
        ax.set_xlabel("probabilità stimata")
    for ax in axes[:, 0]:
        ax.set_ylabel("proporzione osservata")
    handles = [Line2D([], [], color=GROUP_COLORS[g], linewidth=2, label=g) for g in SUBGROUPS]
    handles.append(Line2D([], [], color=TEXT_SECONDARY, linestyle=":", label="calibrazione perfetta"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    save(fig, SECTION, "05_calibration_subgroups", rect=(0, 0.04, 1, 1))


def figure_06_bragg_gresham(curves, comparison):
    """Quota da esaminare contro quota di casi trovati. Destra: il disegno piu' vicino a
    Bragg-Gresham et al. 2024 (non diabetici, sola albuminuria), con i loro due punti operativi.
    Il loro modello usa eGFR < 60 fra i predittori: e' un ordine di grandezza, non un testa a testa."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    titles = {"tesi": "bersaglio della tesi, tutto il training",
              "bragg_gresham": "non diabetici, sola albuminuria (ACR ≥ 30)"}
    for ax, setting in zip(axes, ["tesi", "bragg_gresham"]):
        ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
        for model in MODELS:
            c = curves[(curves["setting"] == setting) & (curves["model"] == model)]
            ax.plot(c["screened"], c["detected"], color=COLORS[model], linewidth=2)
            points = comparison[(comparison["setting"] == setting) & (comparison["model"] == model)
                                & comparison["rule"].str.startswith("soglia")]
            ax.plot(points["screened"], points["detected"], "o", color=COLORS[model], markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.axhline(EXTERNAL["sensitivity"], color=TEXT_SECONDARY, linewidth=0.8, alpha=0.6)
        ax.annotate(f"{100 * EXTERNAL['sensitivity']:.0f}% dei casi", (1, EXTERNAL["sensitivity"]),
                    xycoords=("axes fraction", "data"), xytext=(-2, -3), textcoords="offset points",
                    ha="right", va="top", fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(titles[setting])
        ax.set_xlabel("quota di soggetti esaminati")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        percent_axis(ax)
        percent_axis(ax, "y")
        ax.grid(True)
    axes[0].set_ylabel("quota di casi trovati")
    right = axes[1]
    for r in EXTERNAL["reference"]:
        label = f"Bragg-Gresham {100 * r['threshold']:.0f}%"
        if r["screened"] is not None:
            right.plot(r["screened"], r["detected"], "D", color=TEXT, markersize=7)
            right.annotate(label, (r["screened"], r["detected"]), xytext=(-8, 0),
                           textcoords="offset points", ha="right", va="center", fontsize=8,
                           color=TEXT)
        else:
            # "just under half": quota esatta non riportata nel testo, disegnata come intervallo
            right.hlines(r["detected"], 0.45, 0.50, color=TEXT, linewidth=3)
            right.annotate(f"{label}\n(\"poco meno di metà\" esaminati)", (0.475, r["detected"]),
                           xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                           fontsize=8, color=TEXT)
    extra = [Line2D([], [], marker="o", linestyle="", color=TEXT_SECONDARY, markersize=6,
                    label="modelli alle soglie 5% e 7%"),
             Line2D([], [], marker="D", linestyle="", color=TEXT, markersize=6,
                    label="Bragg-Gresham et al. 2024"),
             Line2D([], [], color=TEXT_SECONDARY, linestyle=":", label="esame a caso")]
    model_legend(fig, extra=extra, ncol=4)
    save(fig, SECTION, "06_bragg_gresham", rect=(0, 0.1, 1, 1))


def run():
    curve = load("decision_curve")
    figure_01_decision_curve(curve)
    figure_02_decision_curve_subgroups(curve)
    figure_03_costs(load("costs"))
    calibration = load("calibration")
    curves = load("calibration_curves")
    figure_04_calibration(curves, load("calibration_bins"), load("probability_histogram"),
                          calibration)
    figure_05_calibration_subgroups(curves, calibration)
    figure_06_bragg_gresham(load("screening_curves"), load("external_comparison"))
    print(f"figure in {TABLES.parent}", flush=True)


if __name__ == "__main__":
    run()
