"""Figure della fase E: richieste dei relatori (analisi post-hoc, esplorativa).

Richiede le tabelle di `python -m src.models.phase_e --evaluate` (analytics/phase_e/). Il test set
non viene letto.
"""
import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.phase_a_report import NAMES
from src.analytics.phase_d_report import CANDIDATE_NAMES
from src.analytics.plotting import AQUA, BLUE, GRAY, ORANGE, TEXT_SECONDARY, YELLOW, save
from src.models.phase_e import EVALUATION, OUTPUT as TABLES

SECTION = "phase_e"
LABELS = {**NAMES, **CANDIDATE_NAMES,
          "extra_trees": "Extra Trees", "hist_gb": "gradient boosting a istogrammi", "svm_rbf": "SVM (kernel RBF)",
          "knn": "k vicini", "naive_bayes": "Naive Bayes", "lda": "analisi discriminante lineare",
          "qda": "analisi discriminante quadratica", "mlp": "rete neurale (MLP)", "adaboost": "AdaBoost",
          "spline_lr": "logistica su spline (GAM)", "lr_interactions": "logistica con interazioni",
          "combinato_lgbm": "ACR + eGFR continui (LightGBM)",
          "combinato_lgbm_etero": "ACR + eGFR continui, scala per soggetto",
          "combinato_mlp": "ACR + eGFR continui (rete multi-compito)",
          "margine_kdigo": "margine KDIGO unico (LightGBM)", "stacking": "stacking (meta-modello)",
          "rank_mean": "media dei ranghi"}
ROLE_COLORS = {"riferimento (fase A)": BLUE, "già calcolato (fase D)": GRAY, "zoo nuovo": ORANGE,
               "ACR + eGFR continui": AQUA, "combinazione di modelli": YELLOW}


def load(name):
    path = TABLES / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}: lanciare prima python -m src.models.phase_e --evaluate")
    return pd.read_csv(path)


def legend(ax, roles):
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=ROLE_COLORS[r], label=r) for r in roles]
    ax.legend(handles=handles, loc="upper left")


def figure_01_auroc():
    """AUROC out-of-fold (probabilità ricalibrate con cross-fitting) con IC di DeLong, tutti i modelli
    dal più basso al più alto; linea verticale al riferimento della Fase A."""
    metrics = load("metrics").sort_values("auc").reset_index(drop=True)
    reference = load("comparison")["reference"].iloc[0]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(metrics) + 1.6))
    for i, r in metrics.iterrows():
        color = ROLE_COLORS[r["role"]]
        ax.hlines(i, r["auc_low"], r["auc_high"], color=color, linewidth=1.5)
        ax.plot(r["auc"], i, "o", color=color, markersize=6)
        ax.annotate(f"{r['auc']:.3f}", (r["auc_high"], i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.axvline(metrics.set_index("model").loc[reference, "auc"], color=BLUE, linestyle=":", linewidth=1)
    ax.axvline(0.5, color=GRAY, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(metrics)), [LABELS.get(m, m) for m in metrics["model"]])
    ax.set_xlabel("AUROC out-of-fold (IC 95% DeLong)")
    ax.set_title(f"{len(metrics)} modelli sullo stesso target e sugli stessi fold")
    ax.grid(axis="x")
    legend(ax, [r for r in ROLE_COLORS if r in set(metrics["role"])])
    save(fig, SECTION, "01_auroc")


def figure_02_differences():
    """Differenza di AUROC e di PR-AUC contro il riferimento della Fase A, IC di Nadeau-Bengio, con la
    soglia di rilevanza della regola. Stesso ordine dei candidati nei due pannelli."""
    comparison = load("comparison")
    reference = comparison["reference"].iloc[0]
    order = list(comparison[comparison["metric"] == "auc"].sort_values("difference")["candidate"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.36 * len(order) + 1.8), sharey=True)
    for ax, metric, title in zip(axes, ["auc", "pr_auc"], ["AUROC", "PR-AUC"]):
        block = comparison[comparison["metric"] == metric].set_index("candidate").loc[order].reset_index()
        for i, r in block.iterrows():
            color = AQUA if r["improves"] else ORANGE
            ax.hlines(i, r["low"], r["high"], color=color, linewidth=1.5)
            ax.plot(r["difference"], i, "o", color=color, markersize=6)
        ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
        ax.axvline(EVALUATION["min_difference"], color=AQUA, linestyle=":", linewidth=1.5)
        ax.set_title(f"{title}: differenza contro {LABELS.get(reference, reference)}")
        ax.set_xlabel("differenza (IC 95%); punteggiata = +0,01")
        ax.grid(axis="x")
    axes[0].set_yticks(range(len(order)), [LABELS.get(m, m) for m in order])
    save(fig, SECTION, "02_differences")


def figure_03_panel():
    """Metriche senza soglia e con soglia (sensibilità 0,90 stimata sugli altri fold)."""
    metrics = load("metrics").sort_values("auc").reset_index(drop=True)
    panels = [("pr_auc", "PR-AUC"), ("brier_scaled", "Brier scalato"),
              ("sens90_specificity", "specificità (sens. 0,90)"), ("sens90_npv", "VPN (sens. 0,90)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 0.3 * len(metrics) + 1.6), sharey=True)
    for ax, (column, title) in zip(axes, panels):
        for i, r in metrics.iterrows():
            ax.plot(r[column], i, "o", color=ROLE_COLORS[r["role"]], markersize=5)
        ax.set_title(title)
        ax.grid(axis="x")
    axes[0].set_yticks(range(len(metrics)), [LABELS.get(m, m) for m in metrics["model"]])
    save(fig, SECTION, "03_panel")


def figure_04_components():
    """AUROC della componente eGFR < 60 e della sola albuminuria (ognuna contro i negativi)."""
    metrics = load("metrics").sort_values("auc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(metrics) + 1.6))
    for i, r in metrics.iterrows():
        ax.hlines(i, r["auc_albuminuria_only"], r["auc_egfr"], color=GRAY, linewidth=1)
        ax.plot(r["auc_albuminuria_only"], i, "o", color=ORANGE, markersize=5)
        ax.plot(r["auc_egfr"], i, "o", color=BLUE, markersize=5)
    ax.set_yticks(range(len(metrics)), [LABELS.get(m, m) for m in metrics["model"]])
    ax.set_xlabel("AUROC contro i negativi: arancio = sola albuminuria, blu = eGFR < 60")
    ax.set_title("Dove sta il limite: le due componenti del target")
    ax.grid(axis="x")
    save(fig, SECTION, "04_components")


TEST = TABLES.parent / "test" / "run_1"
THESIS_MODELS = ["lr_scored", "lr_penalized", "random_forest", "xgboost"]
THESIS_COLORS = dict(zip(THESIS_MODELS, [BLUE, ORANGE, AQUA, YELLOW]))


def test_table(name):
    """Tabella del test set del 22/09/2026 (solo lettura del file salvato, nessun nuovo calcolo sul test)."""
    table = pd.read_csv(TEST / f"{name}.csv")
    return table[(table["technique"] == "none") & (table["group"] == "tutti") & table["model"].isin(THESIS_MODELS)]


def test_counts():
    """Veri/falsi positivi e negativi del test, ricostruiti da recall, quota di allerta, n e positivi."""
    discrimination = pd.read_csv(TEST / "q1_discrimination.csv").query("technique == 'none' and group == 'tutti'")
    n, positives = int(discrimination["n"].iloc[0]), int(discrimination["positives"].iloc[0])
    rows = {}
    for _, r in test_table("q1_operating_points").iterrows():
        tp = round(r["recall"] * positives)
        fp = round(r["alert_rate"] * n) - tp
        rows[r["model"]] = {"tp": tp, "fn": positives - tp, "fp": fp, "tn": n - positives - fp}
    return rows, n, positives


def figure_05_test_levels():
    """Test set: casi riconosciuti per livello KDIGO alla soglia fissata sul training (sensibilità 0,90),
    IC di Wilson; la banda grigia è la quota di soggetti segnalati, cioè la sensibilità attesa a caso."""
    table = test_table("q3_sensitivity")
    levels = ["moderato", "alto", "molto alto", "gravi (alto + molto alto)"]
    alert = test_table("q1_operating_points").set_index("model")["alert_rate"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhspan(alert.min(), alert.max(), color=GRAY, alpha=0.18, lw=0)
    band = plt.matplotlib.patches.Patch(color=GRAY, alpha=0.18,
                                        label=f"a caso: quota segnalata {100 * alert.min():.0f}-{100 * alert.max():.0f}%")
    width = 0.8 / len(THESIS_MODELS)
    for k, model in enumerate(THESIS_MODELS):
        g = table[table["model"] == model].set_index("level").reindex(levels)
        x = [i + (k - (len(THESIS_MODELS) - 1) / 2) * width for i in range(len(levels))]
        ax.errorbar(x, g["sensitivity"], yerr=[g["sensitivity"] - g["low"], g["high"] - g["sensitivity"]],
                    fmt="o", color=THESIS_COLORS[model], markersize=6, capsize=0, lw=1.5, label=NAMES[model])
        for xi, (_, r) in zip(x, g.iterrows()):
            ax.annotate(f"{int(r['detected'])}/{int(r['n'])}", (xi, r["low"]), xytext=(0, -10),
                        textcoords="offset points", ha="center", fontsize=7, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(levels)), ["moderato", "alto", "molto alto", "gravi\n(alto + molto alto)"])
    ax.set_ylim(0.4, 1.04)
    ax.set_ylabel("sensibilità (IC 95% Wilson)")
    ax.set_title("Test set: casi riconosciuti per livello KDIGO (soglia fissata sul training)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, band], loc="lower left", ncol=2)
    ax.grid(axis="y")
    save(fig, SECTION, "05_test_levels")


def figure_06_cv_vs_test():
    """AUROC in validazione incrociata (training) e sul test set, con IC di DeLong. Per XGBoost la
    validazione incrociata è quella con max_depth 1-12, lo stesso modello portato al test."""
    phase_a = TABLES.parent / "phase_a"
    cv = pd.read_csv(phase_a / "evaluation" / "q1_discrimination.csv").query("feature_set == 'main'").set_index("model")
    cv_xgb = pd.read_csv(phase_a / "sensitivity" / "depth_1_12" / "evaluation" / "q1_discrimination.csv")
    cv.loc["xgboost"] = cv_xgb.query("feature_set == 'main' and model == 'xgboost'").set_index("model").loc["xgboost"]
    test = test_table("q1_discrimination").set_index("model")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, model in enumerate(THESIS_MODELS):
        for dy, source, marker in [(-0.15, cv, "o"), (0.15, test, "s")]:
            r = source.loc[model]
            ax.hlines(i + dy, r["auc_low"], r["auc_high"], color=THESIS_COLORS[model], lw=1.5)
            ax.plot(r["auc"], i + dy, marker, color=THESIS_COLORS[model], markersize=6,
                    markerfacecolor="white" if marker == "s" else THESIS_COLORS[model])
            ax.annotate(f"{r['auc']:.3f}", (r["auc_high"], i + dy), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(THESIS_MODELS)), [NAMES[m] for m in THESIS_MODELS])
    ax.invert_yaxis()
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=GRAY, label="validazione incrociata (pieno)"),
               plt.Line2D([], [], marker="s", linestyle="", color=GRAY, markerfacecolor="white",
                          label="test set (vuoto)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_xlabel("AUROC (IC 95% DeLong)")
    ax.set_title("AUROC: validazione incrociata contro test set")
    ax.grid(axis="x")
    save(fig, SECTION, "06_cv_vs_test", rect=(0, 0.08, 1, 1))


def figure_07_test_operating():
    """Test set, soglia fissata sul training: sensibilità, specificità, VPN, VPP e quota segnalata con IC
    di Wilson; linee di riferimento senza modello (VPN = 1 - prevalenza, VPP = prevalenza)."""
    import src.models.evaluation as ev
    counts, n, positives = test_counts()
    prevalence = positives / n
    panels = [("sensibilità", lambda c: (c["tp"], c["tp"] + c["fn"]), None),
              ("specificità", lambda c: (c["tn"], c["tn"] + c["fp"]), None),
              ("VPN", lambda c: (c["tn"], c["tn"] + c["fn"]), 1 - prevalence),
              ("VPP", lambda c: (c["tp"], c["tp"] + c["fp"]), prevalence),
              ("quota segnalata", lambda c: (c["tp"] + c["fp"], n), None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.2), sharey=True)
    for ax, (title, fraction, reference) in zip(axes, panels):
        for i, model in enumerate(THESIS_MODELS):
            value, low, high = ev.wilson(*fraction(counts[model]))
            ax.hlines(i, low, high, color=THESIS_COLORS[model], lw=1.5)
            ax.plot(value, i, "o", color=THESIS_COLORS[model], markersize=6)
            ax.annotate(f"{value:.3f}", (value, i), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7.5, color=TEXT_SECONDARY)
        if reference is not None:
            ax.axvline(reference, color=GRAY, linestyle="--", lw=1)
            ax.text(reference, -0.78, " senza modello", ha="left", va="top", fontsize=7, color=TEXT_SECONDARY)
        ax.set_title(title)
        ax.grid(axis="x")
    axes[0].set_yticks(range(len(THESIS_MODELS)), [NAMES[m] for m in THESIS_MODELS])
    # asse invertito con spazio in alto per la scritta "senza modello"
    axes[0].set_ylim(len(THESIS_MODELS) - 0.5, -0.8)
    fig.suptitle("Test set: il punto operativo fissato sul training (sensibilità 0,90), IC 95% Wilson",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    save(fig, SECTION, "07_test_operating")


def figure_08_why_not_080(model="random_forest"):
    """L'AUROC del target composito è la media delle AUROC dei due gruppi di positivi (eGFR < 60; sola
    albuminuria) contro i negativi, pesata per quanti positivi ha ciascun gruppo: le larghezze delle
    barre sono le quote dei positivi, le altezze le AUROC. Uguaglianza esatta, controllata qui."""
    from src.data.kidney import target
    from src.data.split import PROCESSED
    from src.models.phase_d import components
    train = pd.read_csv(PROCESSED / "train.csv")
    y = target(train).to_numpy()
    albuminuria, low_egfr = components(train)
    n_egfr = int(low_egfr.sum())
    n_alb = int(((albuminuria == 1) & (low_egfr == 0)).sum())
    assert n_egfr + n_alb == y.sum()
    r = load("metrics").set_index("model").loc[model]
    weighted = (n_egfr * r["auc_egfr"] + n_alb * r["auc_albuminuria_only"]) / (n_egfr + n_alb)
    assert abs(weighted - r["auc"]) < 1e-9, (weighted, r["auc"])
    needed = (0.80 * (n_egfr + n_alb) - n_egfr * r["auc_egfr"]) / n_alb
    share = n_egfr / (n_egfr + n_alb)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(0, r["auc_egfr"] - 0.5, width=share, bottom=0.5, align="edge", color=BLUE, edgecolor="white", lw=2)
    ax.bar(share, r["auc_albuminuria_only"] - 0.5, width=1 - share, bottom=0.5, align="edge", color=ORANGE,
           edgecolor="white", lw=2)
    ax.text(share / 2, r["auc_egfr"] + 0.008, f"eGFR < 60\n{n_egfr} positivi\nAUROC {r['auc_egfr']:.3f}".replace(".", ","),
            ha="center", va="bottom", fontsize=10)
    ax.text(share + (1 - share) / 2, r["auc_albuminuria_only"] - 0.02,
            f"sola albuminuria (ACR ≥ 30, eGFR normale)\n{n_alb} positivi · AUROC {r['auc_albuminuria_only']:.3f}".replace(".", ","),
            ha="center", va="top", fontsize=10, color="white", fontweight="bold")
    ax.axhline(weighted, color=TEXT_SECONDARY, lw=2)
    ax.text(0.99, weighted + 0.006, f"AUROC complessiva = media pesata = {weighted:.3f}".replace(".", ","),
            ha="right", va="bottom", fontsize=10, fontweight="bold", transform=ax.get_yaxis_transform())
    ax.axhline(0.80, color=GRAY, lw=1.2, linestyle="--")
    ax.text(0.99, 0.806, f"per arrivare a 0,80 servirebbe AUROC {needed:.2f} sulla sola albuminuria".replace(".", ","),
            ha="right", va="bottom", fontsize=9, color=TEXT_SECONDARY, transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 0.9)
    ax.set_xticks([0, share, 1], ["0%", f"{100 * share:.0f}%", "100%"])
    ax.set_xlabel(f"quota dei positivi del training (larghezza delle barre) · {NAMES.get(model, model)}, "
                  "validazione incrociata")
    ax.set_ylabel("AUROC contro i negativi")
    ax.set_title("Perché unire eGFR e albuminuria non porta a 0,80")
    save(fig, SECTION, "08_why_not_080")


def figure_09_hundred_people(model="random_forest"):
    """100 persone del test set alla soglia della tesi (sensibilità 0,90 fissata sul training): conteggi
    del test riportati a 100 con il metodo dei resti maggiori; i valori esatti sono nella didascalia."""
    import numpy as np
    counts, n, positives = test_counts()
    c = counts[model]
    exact = {k: 100 * v / n for k, v in c.items()}
    per100 = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - int(exact[k]), reverse=True)[:100 - sum(per100.values())]:
        per100[k] += 1
    groups = [("tp", "malato trovato", ORANGE, ORANGE), ("fp", "sano segnalato (falso allarme)", BLUE, BLUE),
              ("fn", "malato perso", "white", ORANGE), ("tn", "sano rassicurato", "white", BLUE)]
    fig = plt.figure(figsize=(11, 5.4))
    ax = fig.add_axes([0.02, 0.1, 0.45, 0.78])
    i = 0
    for key, _, face, edge in groups:
        for _ in range(per100[key]):
            row, col = divmod(i, 10)
            ax.scatter(col, -row - (0.6 if i >= per100["tp"] + per100["fp"] else 0), s=260, facecolor=face,
                       edgecolor=edge, linewidth=2)
            i += 1
    flagged = per100["tp"] + per100["fp"]
    # riga tratteggiata fra l'ultima fila dei segnalati e la prima dei non segnalati (spostata di 0,6)
    last_flagged_row = (flagged - 1) // 10
    cut = -last_flagged_row - 0.8
    ax.plot([-0.5, 9.5], [cut, cut], color=TEXT_SECONDARY, lw=1, linestyle="--")
    ax.text(9.9, cut + 0.15, f"{flagged} segnalati", ha="left", va="bottom", fontsize=9, color=TEXT_SECONDARY)
    ax.text(9.9, cut - 0.15, f"{100 - flagged} non segnalati", ha="left", va="top", fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlim(-0.7, 12.8)
    ax.set_ylim(-10.4, 0.7)
    ax.axis("off")
    tp, fp, fn, tn = (per100[k] for k in ("tp", "fp", "fn", "tn"))
    lines = [
        ("Sensibilità (recall)", f"{tp} malati trovati su {tp + fn}", c["tp"] / (c["tp"] + c["fn"])),
        ("Specificità", f"{tn} sani rassicurati su {tn + fp}", c["tn"] / (c["tn"] + c["fp"])),
        ("VPP (precision)", f"{tp} malati fra {tp + fp} segnalati", c["tp"] / (c["tp"] + c["fp"])),
        ("VPN", f"{tn} sani fra {tn + fn} non segnalati", c["tn"] / (c["tn"] + c["fn"])),
    ]
    tx = fig.add_axes([0.5, 0.1, 0.48, 0.78])
    tx.axis("off")
    for j, (_, label, face, edge) in enumerate(groups):
        tx.scatter(0.02, 0.95 - j * 0.08, s=160, facecolor=face, edgecolor=edge, linewidth=2)
        tx.text(0.06, 0.95 - j * 0.08, f"{label}: {per100[_]}", va="center", fontsize=10)
    for j, (name, text, value) in enumerate(lines):
        y = 0.55 - j * 0.14
        tx.text(0.0, y, name, fontsize=11, fontweight="bold", va="center")
        tx.text(0.0, y - 0.055, f"{text} (valore esatto {100 * value:.1f}%)".replace(".", ","), fontsize=10,
                va="center", color=TEXT_SECONDARY)
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    fig.suptitle(f"100 persone del test set alla soglia della tesi ({NAMES.get(model, model)})", fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.03, f"Conteggi del test ({n} soggetti, {positives} positivi) riportati a 100. "
             "Arancio = malato (marcatori KDIGO), blu = sano; pieno = segnalato dal modello, vuoto = non segnalato.",
             fontsize=8.5, color=TEXT_SECONDARY)
    folder = TABLES
    fig.savefig(folder / "09_hundred_people.png", dpi=300)
    plt.close(fig)


def run():
    figure_08_why_not_080()
    figure_09_hundred_people()
    figure_01_auroc()
    figure_02_differences()
    figure_03_panel()
    figure_04_components()
    figure_05_test_levels()
    figure_06_cv_vs_test()
    figure_07_test_operating()


if __name__ == "__main__":
    run()
