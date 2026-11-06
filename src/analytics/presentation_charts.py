"""Grafici per la presentazione ai relatori (01/10/2026). Analisi descrittiva post-hoc: legge solo le
previsioni out-of-fold già salvate (fasi A, B, D, E) e train.csv; nessun modello riaddestrato, test set mai letto.

01_tutte_le_varianti: AUROC (media dei 5 fold esterni) di ogni modello completo, per metodologia, con i
controlli diagnostici come riferimento. Composito e sola albuminuria (ACR >= 30 ed eGFR >= 60 contro i negativi).
02_valori_liberi: ACR ed eGFR veri contro previsti dai regressori senza soglia (combinato_lgbm, Fase E).

    python -m src.analytics.presentation_charts
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analytics.plotting import AQUA, BLUE, GRAY, GRID as GRID_LINE, ORANGE, OUTPUT, TEXT, TEXT_SECONDARY, save
from src.data.kidney import COLUMNS, egfr
from src.data.split import PROCESSED
from src.models.phase_d import components

SECTION = "presentation"
OUTER = 5
ANALYTICS = Path(OUTPUT)

# metodologie (righe del grafico), dall'alto verso il basso
GROUPS = ["Fase A: modelli della tesi", "Fase B: bilanciamento (8 tecniche × 4 modelli)",
          "Fase D: candidati del tetto", "Fase E: famiglie nuove", "Fase E: ACR + eGFR senza soglia",
          "Fase E: combinazioni di modelli"]
PHASE_D = ["ensemble_mean", "catboost", "lightgbm", "ebm", "tabpfn", "xgboost_native_nan", "xgboost_extended",
           "xgboost_no_pruner", "lr_all_scaled", "target_decomposition", "tri_ensemble_top21",
           "continuous_target/combinato"]
PHASE_E_ZOO = ["extra_trees", "hist_gb", "knn", "naive_bayes", "lda", "qda", "svm_rbf", "adaboost", "mlp",
               "spline_lr", "lr_interactions"]
PHASE_E_CONTINUOUS = ["combinato_lgbm", "combinato_lgbm_etero", "combinato_mlp", "margine_kdigo"]
PHASE_E_COMBINED = ["stacking", "rank_mean"]
CONTROLS = {"positive_control": "controllo positivo\n(albumina urinaria fra le variabili)",
            "semi_synthetic_control/z80": "controllo semisintetico 0,80",
            "semi_synthetic_control/z75": "controllo semisintetico 0,75"}
NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost (3–12)", "lr_penalized": "logistica penalizzata",
         "lr_scored": "logistica SCORED", "tabpfn": "TabPFN", "stacking": "stacking",
         "combinato_lgbm": "LightGBM senza soglia", "spline_lr": "logistica su spline",
         "svm_rbf": "SVM", "lightgbm": "LightGBM", "extra_trees": "Extra Trees",
         "combinato_lgbm_etero": "LightGBM senza soglia, scala per soggetto"}
TECHNIQUES = {"level_weight": "pesi per livello", "class_weight": "pesi di classe", "undersampling": "undersampling",
              "oversampling": "oversampling", "smote_nc": "SMOTE-NC", "smote_nc_level": "SMOTE-NC per livello",
              "ctgan": "CTGAN", "ctgan_level": "CTGAN per livello"}
CONTROL_LABELS = {"positive_control": "controllo positivo", "semi_synthetic_control/z80": "semisintetico 0,80",
                  "semi_synthetic_control/z75": "semisintetico 0,75"}


def pretty(name):
    model, _, technique = name.partition(" | ")
    text = NAMES.get(model, model)
    return f"{text}, {TECHNIQUES[technique]}" if technique else text


def labels():
    train = pd.read_csv(PROCESSED / "train.csv")
    albuminuria, low_egfr = components(train)
    y = ((albuminuria == 1) | (low_egfr == 1)).astype(int)
    only = (y == 0) | ((albuminuria == 1) & (low_egfr == 0))
    return train, y, only


def fold_mean_auc(s, fold, mask, y):
    return float(np.mean([roc_auc_score(y[mask & (fold == k)], s[mask & (fold == k)]) for k in range(OUTER)]))


def from_records(path, n, key=None):
    s, fold = None, np.full(n, -1)
    for k in range(OUTER):
        record = json.loads((path / f"outer{k}.json").read_text(encoding="utf-8"))
        values = np.asarray(record[key] if key else record.get("score", record.get("probability")), dtype=float)
        if s is None:
            s = np.full((n, *values.shape[1:]), np.nan)
        s[record["rows"]] = values
        fold[record["rows"]] = k
    assert not np.isnan(s).any(), path
    return s, fold


def from_long(df, n):
    s, fold = np.full(n, np.nan), np.full(n, -1)
    s[df["row"].to_numpy()] = df["probability"].to_numpy()
    fold[df["row"].to_numpy()] = df["fold"].str[-1].astype(int).to_numpy()
    assert not np.isnan(s).any()
    return s, fold


def variants():
    """Una riga per modello completo (nessun punteggio parziale, nessun duplicato fra fasi)."""
    train, y, only = labels()
    n, everyone = len(y), np.ones(len(y), bool)
    rows = []

    def add(group, name, s, fold):
        rows.append({"gruppo": group, "modello": name, "auc_composito": fold_mean_auc(s, fold, everyone, y),
                     "auc_sola_albuminuria": fold_mean_auc(s, fold, only, y)})

    a = pd.read_csv(ANALYTICS / "phase_a" / "oof_predictions.csv")
    for model, g in a[(a.feature_set == "main") & (a.model != "dummy")].groupby("model"):
        add(GROUPS[0], model, *from_long(g, n))
    b = pd.read_csv(ANALYTICS / "phase_b" / "oof_predictions.csv")
    b = b[(b.feature_set == "main") & (b.model != "dummy") & (b.technique != "none")]
    for (model, technique), g in b.groupby(["model", "technique"]):
        add(GROUPS[1], f"{model} | {technique}", *from_long(g, n))
    for group, names, base in [(GROUPS[2], PHASE_D, "phase_d"), (GROUPS[3], PHASE_E_ZOO, "phase_e"),
                               (GROUPS[4], PHASE_E_CONTINUOUS, "phase_e"), (GROUPS[5], PHASE_E_COMBINED, "phase_e")]:
        for name in names:
            add(group, name, *from_records(ANALYTICS / base / name, n))
    for name in CONTROLS:
        add("controllo", name, *from_records(ANALYTICS / "phase_d" / name, n))
    return pd.DataFrame(rows)


def figure_01(table):
    models = table[table.gruppo != "controllo"]
    controls = table[table.gruppo == "controllo"].set_index("modello")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    panels = [("auc_composito", "Bersaglio KDIGO completo"),
              ("auc_sola_albuminuria", "Sola albuminuria (ACR ≥ 30, eGFR ≥ 60)")]
    rng = np.random.default_rng(42)
    for ax, (column, title) in zip(axes, panels):
        for i, group in enumerate(GROUPS):
            values = models.loc[models.gruppo == group, column].to_numpy()
            jitter = rng.uniform(-0.18, 0.18, len(values))
            ax.scatter(values, i + jitter, s=34, color=BLUE, edgecolor="white", linewidth=1, zorder=3)
            best = models[models.gruppo == group].sort_values(column).iloc[-1]
            ax.annotate(f"{pretty(best.modello)} {best[column]:.3f}".replace(".", ","),
                        (best[column], i), xytext=(0, 13), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT_SECONDARY)
        for name in CONTROLS:
            x = controls.loc[name, column]
            ax.axvline(x, color=ORANGE if name == "positive_control" else GRAY, linestyle="--", linewidth=1.5,
                       zorder=2)
            ax.text(x + 0.006, len(GROUPS) - 0.6, f"{CONTROL_LABELS[name]} {x:.3f}".replace(".", ","),
                    rotation=90, fontsize=8, color=TEXT_SECONDARY, va="bottom", ha="left")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(len(GROUPS) - 0.5, -0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("AUROC (media dei 5 fold esterni)")
        ax.grid(axis="x")
    axes[0].set_yticks(range(len(GROUPS)), GROUPS)
    fig.suptitle(f"{len(models)} modelli, 6 metodologie: tutti si fermano nella stessa fascia\n"
                 "Le linee tratteggiate sono i controlli: con l'informazione renale la stessa pipeline arriva a 0,93",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    save(fig, SECTION, "01_tutte_le_varianti", rect=(0, 0, 1, 0.93))


def figure_02():
    train, y, only = labels()
    mu, _ = from_records(ANALYTICS / "phase_e" / "combinato_lgbm", len(y), key="mu")
    true_acr = train[COLUMNS["acr"]].to_numpy(float)
    pred_acr = np.exp(mu[:, 0])
    true_egfr = egfr(train).to_numpy()
    predicted = train.copy()
    predicted[COLUMNS["creatinine"]] = np.exp(mu[:, 1])
    pred_egfr = egfr(predicted).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    panels = [(axes[0], true_acr, pred_acr, 30, "ACR (mg/g)", True,
               f"ACR: il 90% delle previsioni sta fra {np.percentile(pred_acr, 5):.1f} e "
               f"{np.percentile(pred_acr, 95):.1f} mg/g"),
              (axes[1], true_egfr, pred_egfr, 60, "eGFR (ml/min/1,73 m²)", False,
               f"eGFR: previsti sotto 60 {int((pred_egfr < 60).sum())} su {len(pred_egfr)}, "
               f"veri sotto 60 {int((true_egfr < 60).sum())}")]
    for ax, true, pred, cut, unit, log, title in panels:
        ax.scatter(true, pred, s=6, color=BLUE, alpha=0.25, linewidth=0, zorder=3)
        low, high = (min(true.min(), pred.min()), max(true.max(), pred.max()))
        ax.plot([low, high], [low, high], color=GRAY, linewidth=1.2, zorder=2)
        ax.text(high, high, "  previsione = valore vero", fontsize=8, color=TEXT_SECONDARY, ha="right", va="bottom")
        for line in (ax.axvline, ax.axhline):
            line(cut, color=ORANGE, linestyle="--", linewidth=1.5, zorder=2)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(f"{unit}, valore vero")
        ax.set_ylabel(f"{unit}, previsto senza soglia")
        ax.set_title(title.replace(".", ","), fontsize=11)
        ax.grid()
    fig.suptitle("Senza soglie i modelli non stimano i valori: li schiacciano verso la media\n"
                 "LightGBM della Fase E, previsioni out-of-fold sul training; tratteggio arancione = soglia KDIGO",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    save(fig, SECTION, "02_valori_liberi", rect=(0, 0, 1, 0.92))


def random_forest_oof(n):
    a = pd.read_csv(ANALYTICS / "phase_a" / "oof_predictions.csv")
    return from_long(a[(a.feature_set == "main") & (a.model == "random_forest")], n)


def figure_03():
    """Random Forest della Fase A, previsioni OOF del training: cosa cambia al variare della soglia."""
    from src.data.kidney import kdigo_level
    train, y, _ = labels()
    p, _ = random_forest_oof(len(y))
    level = kdigo_level(train).astype(str).to_numpy()
    severe = np.isin(level, ["alto", "molto alto"])
    cuts = pd.read_csv(ANALYTICS / "phase_a" / "evaluation" / "cutpoints.csv")
    chosen = float(cuts.query("feature_set == 'main' and model == 'random_forest'").threshold.iloc[0])
    grid = np.geomspace(0.01, 0.6, 400)

    def rates(t):
        flag = p >= t
        tn, fn = np.sum(~flag & (y == 0)), np.sum(~flag & (y == 1))
        return {"positivi trovati": flag[y == 1].mean(), "casi gravi trovati": flag[severe].mean(),
                "molto alto trovati": flag[level == "molto alto"].mean(), "soggetti segnalati": flag.mean(),
                "VPP": (y[flag] == 1).mean() if flag.any() else np.nan,
                "VPN": tn / (tn + fn) if tn + fn else np.nan}

    curve = pd.DataFrame([rates(t) for t in grid], index=grid)
    marks = [(chosen, f"soglia scelta {chosen:.3f}\n(sensibilità 0,90)", ORANGE, "-"),
             (0.088, "0,088", GRAY, "--"), (0.5, "0,5 (predefinita)", GRAY, ":")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    panels = [(axes[0], ["positivi trovati", "casi gravi trovati", "molto alto trovati"], "Chi trova"),
              (axes[1], ["soggetti segnalati", "VPP", "VPN"], "Quanto costa e quanto ci si può fidare")]
    colors = [BLUE, ORANGE, AQUA]
    for ax, columns, title in panels:
        for column, color in zip(columns, colors):
            ax.plot(curve.index, curve[column] * 100, color=color, linewidth=2, label=column)
        height = 3 if ax is axes[0] else 48
        for t, text, color, style in marks:
            ax.axvline(t, color=color, linestyle=style, linewidth=1.5)
            values = rates(t)
            note = "\n".join(f"{c}: nessuno segnalato" if np.isnan(values[c]) else f"{c}: {values[c]:.0%}"
                             for c in columns)
            left = t == chosen
            ax.annotate(f"{text}\n{note}".replace(".", ","), (t, height), xytext=(-5 if left else 5, 0),
                        textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY, va="bottom",
                        ha="right" if left else "left")
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ["0,01", "0,02", "0,05", "0,1", "0,2", "0,5"])
        ax.set_ylim(0, 102)
        ax.set_xlabel("soglia di probabilità (scala logaritmica)")
        ax.set_ylabel("%")
        ax.set_title(title, fontsize=11)
        if ax is axes[0]:
            ax.legend(loc="upper right")
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(0.6, 0.86))
        ax.grid()
    fig.suptitle("Abbassare la soglia da 0,5 a quella scelta fa passare i casi gravi trovati da quasi nessuno "
                 "alla quasi totalità\nRandom Forest, previsioni out-of-fold del training (4.350 soggetti, "
                 f"{int(severe.sum())} casi gravi); la soglia è fissata qui e applicata invariata al test",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    save(fig, SECTION, "03_soglia", rect=(0, 0, 1, 0.9))
    return chosen, curve


def figure_04():
    """Informazione renale: il controllo positivo (albumina urinaria fra le variabili) e i semisintetici."""
    from sklearn.metrics import roc_curve
    _, y, _ = labels()
    n = len(y)
    phase_d = ANALYTICS / "phase_d"
    series = [("Random Forest (tesi)", random_forest_oof(n)[0], BLUE, "-"),
              ("TabPFN (migliore senza esami renali)", from_records(phase_d / "tabpfn", n)[0], AQUA, "-"),
              ("semisintetico 0,75", from_records(phase_d / "semi_synthetic_control/z75", n)[0], GRAY, ":"),
              ("semisintetico 0,80", from_records(phase_d / "semi_synthetic_control/z80", n)[0], GRAY, "--"),
              ("controllo positivo: + albumina urinaria", from_records(phase_d / "positive_control", n)[0],
               ORANGE, "-")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.1]})
    rows = []
    for name, s, color, style in series:
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        axes[0].plot(fpr, tpr, color=color, linestyle=style, linewidth=2,
                     label=f"{name}: {auc:.3f}".replace(".", ","))
        # soglia più alta che trova almeno il 90% dei malati
        threshold = np.sort(s[y == 1])[::-1][int(np.ceil(0.90 * (y == 1).sum())) - 1]
        rows.append((name, float(np.mean(s[y == 0] < threshold)), auc, color))
    axes[0].plot([0, 1], [0, 1], color=GRID_LINE, linewidth=1)
    axes[0].set_xlabel("1 − specificità (sani segnalati)")
    axes[0].set_ylabel("sensibilità (malati trovati)")
    axes[0].set_title("Curve ROC (AUROC out-of-fold aggregata)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid()
    values = [r[1] * 100 for r in rows]
    bars = axes[1].barh(range(len(rows)), values, color=[r[3] for r in rows], height=0.6)
    for bar, value in zip(bars, values):
        axes[1].annotate(f"{value:.0f} su 100", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                         xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT_SECONDARY)
    axes[1].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("sani esclusi correttamente ogni 100, trovando il 90% dei malati")
    axes[1].set_title("Quanti esami si evitano a parità di malati trovati", fontsize=11)
    axes[1].grid(axis="x")
    fig.suptitle("Con l'albumina urinaria fra le variabili la stessa pipeline arriva a 0,93: il limite è "
                 "l'informazione, non il metodo\nIl controllo positivo non è un modello utilizzabile: l'albumina "
                 "urinaria è il numeratore dell'ACR, cioè l'esame che K-Risk vuole evitare",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    save(fig, SECTION, "04_informazione_renale", rect=(0, 0, 1, 0.9))
    return rows


def figure_05(table):
    """Scheda dei migliori risultati, ogni numero letto dal suo file."""
    test = ANALYTICS / "test" / "run_1"
    q1 = pd.read_csv(test / "q1_discrimination.csv").query(
        "technique == 'fase_a' and model == 'random_forest' and group == 'tutti'").iloc[0]
    q3 = pd.read_csv(test / "q3_sensitivity.csv").query(
        "technique == 'fase_a' and model == 'random_forest' and group == 'tutti'").set_index("level")
    tp = pd.read_csv(test / "test_predictions.csv").query("technique == 'fase_a' and model == 'random_forest'")
    cuts = pd.read_csv(ANALYTICS / "phase_a" / "evaluation" / "cutpoints.csv")
    chosen = float(cuts.query("feature_set == 'main' and model == 'random_forest'").threshold.iloc[0])
    negative = tp.probability < chosen
    npv = (tp.target[negative] == 0).mean()
    baseline = (tp.target == 0).mean()
    e = pd.read_csv(ANALYTICS / "phase_e" / "metrics.csv").set_index("model")
    dc = pd.read_csv(ANALYTICS / "quality" / "evaluation" / "decision_curve.csv")
    avoided = dc[(dc.group == "tutti") & (dc.model == "random_forest")
                 & np.isclose(dc.threshold, 0.10)].avoided_per_100.iloc[0]
    best_alb = table[table.gruppo != "controllo"].sort_values("auc_sola_albuminuria").iloc[-1]
    severe = q3.loc["gravi (alto + molto alto)"]
    very = q3.loc["molto alto"]
    tiles = [
        (f"{q1.auc:.3f}", f"AUROC sul test\n(IC {q1.auc_low:.3f}–{q1.auc_high:.3f})",
         "Random Forest, test set, 1.451 soggetti"),
        (f"{int(severe.detected)} su {int(severe.n)}", "casi gravi trovati sul test",
         "Random Forest, soglia fissata sul training"),
        (f"{int(very.detected)} su {int(very.n)}", "rischio molto alto\ntrovati sul test",
         "Random Forest, soglia fissata sul training"),
        (f"{npv:.1%}", f"non segnalati davvero negativi\n(senza modello: {baseline:.1%})", "Random Forest, test set"),
        (f"{avoided:.1f}", "esami inutili evitati\nogni 100 persone (rischio 10%)",
         "decision curve, Random Forest, CV"),
        (f"{e.loc['tabpfn', 'auc_folds_mean']:.3f}", "AUROC migliore in CV\n(TabPFN, media dei fold)",
         "Fase E, 24 modelli"),
        (f"{best_alb.auc_sola_albuminuria:.3f}", "AUROC migliore sulla\nsola albuminuria",
         f"{pretty(best_alb.modello)}, media dei fold"),
        (f"{e.loc['tabpfn', 'auc_egfr']:.3f}", "AUROC su eGFR < 60", "TabPFN, Fase E"),
        (f"{e.loc['tabpfn', 'auc_molto_alto']:.3f}", "AUROC rischio molto alto\ncontro basso", "TabPFN, Fase E"),
        (f"{e.loc['tabpfn', 'top10_severe_recall']:.1%}", "dei casi gravi nel 10%\npiù a rischio", "TabPFN, Fase E"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, (value, label, source) in zip(axes.flat, tiles):
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes, facecolor="#f4f7fc",
                                   edgecolor="#d6e2f3", linewidth=1))
        ax.text(0.08, 0.68, value.replace(".", ","), transform=ax.transAxes, fontsize=24, fontweight="bold",
                color=TEXT, va="center")
        ax.text(0.08, 0.38, label.replace(".", ","), transform=ax.transAxes, fontsize=10, color=TEXT, va="center")
        ax.text(0.08, 0.1, source, transform=ax.transAxes, fontsize=7.5, color=TEXT_SECONDARY, va="center")
    fig.suptitle("K-Risk, i risultati migliori: riconosce bene i casi gravi e la funzione renale ridotta, "
                 "e serve soprattutto per escludere",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    save(fig, SECTION, "05_migliori_risultati", rect=(0, 0, 1, 0.93))
    return tiles


TECHNIQUE_KIND = {"class_weight": "pesi o campionamento", "level_weight": "pesi o campionamento",
                  "undersampling": "pesi o campionamento", "oversampling": "pesi o campionamento",
                  "smote_nc": "SMOTE-NC", "smote_nc_level": "SMOTE-NC", "ctgan": "CTGAN (esplorativa)",
                  "ctgan_level": "CTGAN (esplorativa)"}
MODEL_NAMES = {"lr_scored": "logistica SCORED", "lr_penalized": "logistica penalizzata",
               "random_forest": "Random Forest", "xgboost": "XGBoost (1–12)"}


def figure_07():
    """Fase B: alla soglia 0,5 il bilanciamento sembra alzare il recall; a parità di persone segnalate il
    modello senza bilanciamento trova gli stessi malati (il bilanciamento equivale a una soglia più bassa)."""
    from src.data.kidney import kdigo_level
    train, y, _ = labels()
    severe = np.isin(kdigo_level(train).astype(str).to_numpy(), ["alto", "molto alto"])
    b = pd.read_csv(ANALYTICS / "phase_b" / "oof_predictions.csv")
    b = b[(b.feature_set == "main") & (b.model != "dummy")]
    rows = []
    for model, gm in b.groupby("model"):
        base, _ = from_long(gm[gm.technique == "none"], len(y))
        rows.append({"modello": model, "tecnica": "none", "segnalati": float(np.mean(base >= 0.5)),
                     "recall_bilanciato": float(np.mean(base[y == 1] >= 0.5)), "recall_senza": np.nan,
                     "gravi_bilanciato": int(np.sum(base[severe] >= 0.5)), "gravi_senza": np.nan,
                     "soglia_equivalente": np.nan})
        for technique, g in gm[gm.technique != "none"].groupby("technique"):
            s, _ = from_long(g, len(y))
            flag = s >= 0.5
            if not flag.any():
                continue
            cut = np.quantile(base, 1 - flag.mean())        # stessa quota di segnalati, senza bilanciamento
            same = base >= cut
            rows.append({"modello": model, "tecnica": technique, "segnalati": float(flag.mean()),
                         "recall_bilanciato": float(flag[y == 1].mean()), "recall_senza": float(same[y == 1].mean()),
                         "gravi_bilanciato": int(flag[severe].sum()), "gravi_senza": int(same[severe].sum()),
                         "soglia_equivalente": float(cut)})
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    # sinistra: l'illusione, recall alla soglia 0,5 senza e con il bilanciamento che rende di più per modello
    order = list(MODEL_NAMES)
    none = table[table.tecnica == "none"].set_index("modello").recall_bilanciato
    best = table[table.tecnica != "none"].sort_values("recall_bilanciato").groupby("modello").tail(1).set_index("modello")
    x = np.arange(len(order))
    bars_none = axes[0].bar(x - 0.2, [none[m] * 100 for m in order], width=0.38, color=GRAY, label="senza bilanciamento")
    bars_best = axes[0].bar(x + 0.2, [best.loc[m, "recall_bilanciato"] * 100 for m in order], width=0.38, color=ORANGE,
                            label="con il bilanciamento che rende di più")
    for bars, texts in [(bars_none, [f"{none[m]:.1%}" for m in order]),
                        (bars_best, [f"{best.loc[m, 'recall_bilanciato']:.1%}\n{TECHNIQUES[best.loc[m, 'tecnica']]}"
                                     for m in order])]:
        for bar, text in zip(bars, texts):
            axes[0].annotate(text.replace(".", ","), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5,
                             color=TEXT_SECONDARY)
    axes[0].set_xticks(x, [MODEL_NAMES[m] for m in order], fontsize=8.5)
    axes[0].set_ylim(0, 80)
    axes[0].set_ylabel("malati trovati alla soglia 0,5 (%)")
    axes[0].set_title("Quello che sembra: alla soglia 0,5", fontsize=11)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y")
    # destra: la realtà, stessa quota di segnalati
    kinds = {"pesi o campionamento": BLUE, "SMOTE-NC": ORANGE, "CTGAN (esplorativa)": GRAY}
    balanced = table[table.tecnica != "none"]
    for kind, color in kinds.items():
        part = balanced[balanced.tecnica.map(TECHNIQUE_KIND) == kind]
        axes[1].scatter(part.recall_senza * 100, part.recall_bilanciato * 100, s=42, color=color,
                        edgecolor="white", linewidth=1, label=f"{kind} ({len(part)})", zorder=3)
    axes[1].plot([0, 70], [0, 70], color=GRAY, linewidth=1.2, zorder=2)
    axes[1].text(68, 66, "stessi malati trovati", fontsize=8, color=TEXT_SECONDARY, ha="right", va="top", rotation=38)
    star = balanced.sort_values("recall_bilanciato").iloc[-1]
    axes[1].annotate(f"{MODEL_NAMES[star.modello]}, {TECHNIQUES[star.tecnica]}:\n"
                     f"{star.recall_bilanciato:.1%} con il bilanciamento a 0,5\n"
                     f"{star.recall_senza:.1%} senza, alla soglia {star.soglia_equivalente:.3f}".replace(".", ","),
                     (star.recall_senza * 100, star.recall_bilanciato * 100), xytext=(2, 38),
                     textcoords="data", fontsize=8, color=TEXT_SECONDARY,
                     arrowprops={"arrowstyle": "-", "color": GRAY})
    differences = (balanced.recall_bilanciato - balanced.recall_senza) * 100
    axes[1].set_xlim(0, 70)
    axes[1].set_ylim(0, 70)
    axes[1].set_xlabel("malati trovati SENZA bilanciamento, segnalando le stesse persone (%)")
    axes[1].set_ylabel("malati trovati CON bilanciamento, soglia 0,5 (%)")
    axes[1].set_title(f"Quello che è: a parità di segnalati, differenza da {differences.min():+.1f} a "
                      f"{differences.max():+.1f} punti".replace(".", ","), fontsize=11)
    axes[1].legend(loc="upper left", title="tecnica", title_fontsize=8)
    axes[1].grid()
    fig.suptitle("Il bilanciamento non trova più malati: equivale ad abbassare la soglia\n"
                 "Fase B, 4 modelli × 8 tecniche, previsioni out-of-fold del training (4.350 soggetti, 425 positivi)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    save(fig, SECTION, "07_bilanciamento_apparente", rect=(0, 0, 1, 0.9))
    table.to_csv(ANALYTICS / SECTION / "bilanciamento_stessa_quota.csv", index=False)
    return table


GROUP_NAMES = {"renal_exams": "esami renali (urea)", "undocumented": "non documentate", "sparse": "quasi vuote (> 50%)",
               "knowledge_survey": "questionario", "derived": "derivate da variabili usate",
               "duplicate_measures": "doppioni", "linear_combinations": "combinazioni lineari",
               "comorbidities": "comorbidità (codice 9)", "unstable_indices": "indice instabile",
               "high_missing": "mancanti > 15%"}


def univariate_auc(x, y, mask):
    """AUROC di una sola colonna, verso indifferente (max(AUC, 1 - AUC)), sui soggetti senza mancanti."""
    x = pd.Series(np.asarray(x, dtype=float))
    m = mask & x.notna().to_numpy()
    if x[m].nunique() < 2 or len(np.unique(y[m])) < 2:
        return np.nan
    a = roc_auc_score(y[m], x[m])
    return max(a, 1 - a)


def figure_08(table):
    """Colonne escluse (tranne leakage e amministrative): AUROC di ciascuna da sola; esame delle urine."""
    from src.config import CONFIG
    train, y, only = labels()
    features = CONFIG["features"]
    everyone = np.ones(len(y), bool)
    rows = []
    for group, columns in features["exclude"].items():
        if group in ("leakage", "admin"):
            continue
        for c in columns:
            if c not in train:
                continue
            x = pd.to_numeric(train[c], errors="coerce")
            if c in features["unknown_columns"]:
                x = x.where(x != features["unknown_code"])
            rows.append({"gruppo": GROUP_NAMES[group], "colonna": c, "mancanti": float(x.isna().mean()),
                         "auc_composito": univariate_auc(x, y, everyone),
                         "auc_sola_albuminuria": univariate_auc(x, y, only)})
    excluded = pd.DataFrame(rows)
    age = univariate_auc(train["Age"], y, everyone)
    ratio = 176.8 * train["UmALB"] / train["UCRE"]
    rebuilt = float(np.mean((ratio >= 30) == (train[COLUMNS["acr"]] >= 30)))
    best_blood = table[table.gruppo != "controllo"].auc_sola_albuminuria.max()
    urine = [("migliore con il solo sangue (stacking)", best_blood, BLUE),
             ("creatinina urinaria da sola", univariate_auc(train["UCRE"], y, only), GRAY),
             ("albumina urinaria da sola", univariate_auc(train["UmALB"], y, only), ORANGE),
             ("albumina / creatinina urinaria", univariate_auc(ratio, y, only), ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
    top = excluded.sort_values("auc_composito", ascending=False).head(15).iloc[::-1]
    colors = [GRAY if m > 0.5 else BLUE for m in top.mancanti]
    bars = axes[0].barh(range(len(top)), top.auc_composito, color=colors, height=0.65)
    for bar, (_, r) in zip(bars, top.iterrows()):
        axes[0].annotate(f"{r.auc_composito:.3f} · {r.gruppo} · mancanti {r.mancanti:.1%}".replace(".", ","),
                         (bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=7.5, color=TEXT_SECONDARY)
    axes[0].axvline(age, color=ORANGE, linestyle="--", linewidth=1.5)
    axes[0].text(age, len(top) - 0.4, f" età, la migliore fra quelle usate: {age:.3f}".replace(".", ","),
                 fontsize=8, color=TEXT_SECONDARY, va="bottom")
    axes[0].set_yticks(range(len(top)), top.colonna)
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_xlabel("AUROC della colonna da sola (bersaglio KDIGO, training)")
    axes[0].set_title(f"Le 15 più forti fra le {len(excluded)} colonne escluse (grigio: oltre metà vuota)", fontsize=11)
    axes[0].grid(axis="x")
    bars = axes[1].barh(range(len(urine)), [u[1] for u in urine], color=[u[2] for u in urine], height=0.6)
    for bar, (_, value, _) in zip(bars, urine):
        axes[1].annotate(f"{value:.3f}".replace(".", ","), (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                         xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT_SECONDARY)
    axes[1].set_yticks(range(len(urine)), [u[0] for u in urine])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.5, 1.08)
    axes[1].set_xlabel("AUROC sulla sola albuminuria (ACR ≥ 30, eGFR ≥ 60)")
    axes[1].set_title(f"Urine: etichetta ricostruita al {rebuilt:.0%}", fontsize=11)
    axes[1].grid(axis="x")
    fig.suptitle("Le colonne escluse non portano informazione; l'esame delle urine contiene la risposta (leakage)\n"
                 "Albumina e creatinina urinarie sono numeratore e denominatore dell'ACR: usarle vuol dire "
                 "calcolare il bersaglio, non prevederlo",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    save(fig, SECTION, "08_colonne_escluse_e_urine", rect=(0, 0, 1, 0.9))
    excluded.to_csv(ANALYTICS / SECTION / "colonne_escluse.csv", index=False)
    return excluded, age, urine, rebuilt


def main():
    table = variants()
    folder = ANALYTICS / SECTION
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "tutte_le_varianti.csv", index=False)
    figure_01(table)
    figure_02()
    chosen, curve = figure_03()
    print("soglia scelta", round(chosen, 4))
    for t in (chosen, 0.088, 0.5):
        i = int(np.argmin(np.abs(curve.index - t)))
        print(t, curve.iloc[i].round(3).to_dict())
    for name, specificity, auc, _ in figure_04():
        print(name, "AUROC", round(auc, 3), "sani esclusi a sens 0,90", round(specificity, 3))
    for value, label, source in figure_05(table):
        print(value, "|", label.replace("\n", " "), "|", source)
    balance = figure_07()
    balanced = balance[balance.tecnica != "none"]
    difference = (balanced.recall_bilanciato - balanced.recall_senza) * 100
    print("bilanciamento: combinazioni", len(balanced), "differenza recall (punti) min", round(difference.min(), 1),
          "max", round(difference.max(), 1), "media", round(difference.mean(), 1))
    print(balance[balance.tecnica == "none"][["modello", "recall_bilanciato"]].round(3).to_string(index=False))
    excluded, age, urine, rebuilt = figure_08(table)
    print("colonne escluse", len(excluded), "| età", round(age, 3), "| etichetta ricostruita", rebuilt)
    print(excluded.sort_values("auc_composito", ascending=False).head(6).round(3).to_string(index=False))
    for name, value, _ in urine:
        print(name, round(value, 3))
    summary = table[table.gruppo != "controllo"].groupby("gruppo")[["auc_composito", "auc_sola_albuminuria"]]
    print(summary.agg(["count", "min", "max"]).round(3).to_string())
    print(table[table.gruppo == "controllo"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
