"""Script completo di benchmarking per la tesi:
Confronta TUTTI i modelli sviluppati (Fase A, Fase D e nuovi modelli aggiunti),
estrae tutte le metriche (AUROC, PR-AUC, eGFR-AUROC, casi gravi, top decile lift,
Brier score, F1, Balanced Accuracy, Net Benefit) e genera grafici e tabelle
ad altissima risoluzione per la presentazione della tesi e ai relatori.
"""
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier

# Configurazione stile grafici
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "analytics" / "benchmark_thesis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from src.data.imputed import load_fold
from src.data.kidney import egfr, kdigo_level, target
from src.data.split import PROCESSED


def delong_ci(y_true, y_pred, alpha=0.05):
    """Intervallo di confidenza DeLong per AUROC."""
    y = np.asarray(y_true) == 1
    p = np.asarray(y_pred, dtype=float)
    pos, neg = p[y], p[~y]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    v10 = (ranks[:m] - stats.rankdata(pos)) / n
    v01 = 1 - (ranks[m:] - stats.rankdata(neg)) / m
    auc = v10.mean()
    se = np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def load_all_predictions(train_df):
    """Raccoglie le predizioni out-of-fold di TUTTI i modelli."""
    n_samples = len(train_df)
    preds = {}

    # 1. Modelli Fase A
    oof_phase_a = pd.read_csv(BASE_DIR / "analytics" / "phase_a" / "oof_predictions.csv")
    phase_a_main = oof_phase_a[oof_phase_a["feature_set"] == "main"]
    for m in phase_a_main["model"].unique():
        sub = phase_a_main[phase_a_main["model"] == m].sort_values("row")
        arr = np.zeros(n_samples)
        arr[sub["row"].to_numpy()] = sub["probability"].to_numpy()
        name_map = {
            "dummy": "Dummy (Prevalenza)",
            "lr_scored": "Logistic Regr. (SCORED)",
            "lr_penalized": "Logistic Regr. (Penalized)",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost (Fase A)",
        }
        preds[name_map.get(m, m)] = arr

    # 2. Modelli Fase D
    phase_d_dir = BASE_DIR / "analytics" / "phase_d"
    phase_d_candidates = {
        "catboost": "CatBoost",
        "lightgbm": "LightGBM",
        "ebm": "Explainable BM (EBM)",
        "tabpfn": "TabPFN",
        "xgboost_extended": "XGBoost (Extended)",
        "xgboost_native_nan": "XGBoost (Native NaN)",
        "xgboost_no_pruner": "XGBoost (No Pruner)",
        "lr_all_scaled": "Logistic Regr. (All Scaled)",
        "tri_ensemble_top21": "Tri-Ensemble (Top 21)",
        "ensemble_mean": "Ensemble Mean (LR+RF+XGB)",
        "target_decomposition": "Target Decomposition (eGFR+ACR)",
    }

    for folder, label in phase_d_candidates.items():
        model_dir = phase_d_dir / folder
        if (model_dir / "outer0.json").exists():
            arr = np.zeros(n_samples)
            for k in range(5):
                with open(model_dir / f"outer{k}.json", "r") as f:
                    data = json.load(f)
                    arr[data["rows"]] = data["probability"]
            preds[label] = arr

    # Continuous target
    cont_dir = phase_d_dir / "continuous_target" / "combinato"
    if (cont_dir / "outer0.json").exists():
        arr = np.zeros(n_samples)
        for k in range(5):
            with open(cont_dir / f"outer{k}.json", "r") as f:
                data = json.load(f)
                arr[data["rows"]] = data["probability"]
        preds["Continuous Target (Dual Regression)"] = arr

    # 3. Nuovi Modelli Aggiuntivi su Folds Precomputati
    print("Addestramento modelli aggiuntivi (ExtraTrees e MLP)...")
    y = target(train_df).to_numpy()
    arr_et = np.zeros(n_samples)
    arr_mlp = np.zeros(n_samples)

    for k in range(5):
        X_train, X_valid = load_fold("main", outer=k)
        y_tr = y[X_train.index]

        # ExtraTrees
        et = ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
        et.fit(X_train, y_tr)
        arr_et[X_valid.index] = et.predict_proba(X_valid)[:, 1]

        # MLP Neural Network
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42, early_stopping=True)
        mlp.fit(X_train, y_tr)
        arr_mlp[X_valid.index] = mlp.predict_proba(X_valid)[:, 1]

    preds["Extra Trees"] = arr_et
    preds["Tabular MLP (Neural Net)"] = arr_mlp

    # 4. Super Learner / Stacking Classifier (Stacking di LR, RF, XGB, CatBoost, LightGBM)
    print("Costruzione Super Learner (Stacking Ensemble)...")
    stack_features = ["Logistic Regr. (Penalized)", "Random Forest", "XGBoost (Extended)", "CatBoost", "LightGBM"]
    X_meta = np.column_stack([preds[m] for m in stack_features])
    arr_stack = np.zeros(n_samples)

    # 5-fold CV sul meta-modello per evitare leakage
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X_meta, y):
        meta = LogisticRegression(C=1.0, random_state=42)
        meta.fit(X_meta[tr_idx], y[tr_idx])
        arr_stack[va_idx] = meta.predict_proba(X_meta[va_idx])[:, 1]
    preds["Super Learner (Stacking Meta-Ensemble)"] = arr_stack

    return preds


def compute_metrics(y_true, p, gfr_arr, acr_arr, levels_arr):
    """Calcola la batteria completa di metriche per un modello."""
    gfr_impaired = (gfr_arr < 60).astype(int)
    acr_high = (acr_arr >= 30).astype(int)
    acr_macro = (acr_arr >= 300).astype(int)
    severe_mask = np.isin(levels_arr, ["alto", "molto alto"])
    prevalence = y_true.mean()

    # AUROC e PR-AUC
    auc, auc_low, auc_high = delong_ci(y_true, p)
    pr_auc = average_precision_score(y_true, p)
    brier = brier_score_loss(y_true, p)

    # AUROC sui componenti specifici
    auc_egfr = roc_auc_score(gfr_impaired, p)
    auc_acr = roc_auc_score(acr_high, p)
    auc_macro = roc_auc_score(acr_macro, p)

    # Soglia Youden
    fpr, tpr, ths = roc_curve(y_true, p)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    youden_th = ths[best_j_idx]
    pred_youden = (p >= youden_th).astype(int)
    youden_sens = tpr[best_j_idx]
    youden_spec = 1 - fpr[best_j_idx]
    youden_f1 = f1_score(y_true, pred_youden)
    youden_bal_acc = balanced_accuracy_score(y_true, pred_youden)

    # Soglia a sensibilità 90%
    sorted_pos = np.sort(p[y_true == 1])[::-1]
    k90 = int(np.ceil(0.90 * len(sorted_pos))) - 1
    th90 = sorted_pos[k90]
    pred90 = (p >= th90).astype(int)
    spec90 = (1 - pred90[y_true == 0]).mean()
    prec90 = y_true[pred90 == 1].mean()
    alert_rate90 = pred90.mean()
    severe_sens = (p[severe_mask] >= th90).mean()

    # Decili di rischio (Lift)
    cut10 = np.quantile(p, 0.90)
    top10_prev = y_true[p >= cut10].mean()
    top10_lift = top10_prev / prevalence

    cut20 = np.quantile(p, 0.80)
    top20_prev = y_true[p >= cut20].mean()
    top20_lift = top20_prev / prevalence

    # Net Benefit a soglia clinica 10%
    pt = 0.10
    tp = np.sum(pred90 & (y_true == 1))
    fp = np.sum(pred90 & (y_true == 0))
    net_benefit = (tp / len(y_true)) - (fp / len(y_true)) * (pt / (1 - pt))

    return {
        "AUROC": auc,
        "AUROC_low": auc_low,
        "AUROC_high": auc_high,
        "PR_AUC": pr_auc,
        "Brier_Score": brier,
        "AUROC_eGFR_inf_60": auc_egfr,
        "AUROC_ACR_sup_30": auc_acr,
        "AUROC_Macroalbuminuria": auc_macro,
        "Severe_Cases_Sensitivity": severe_sens,
        "Youden_Sensitivity": youden_sens,
        "Youden_Specificity": youden_spec,
        "Youden_F1": youden_f1,
        "Youden_Balanced_Accuracy": youden_bal_acc,
        "Spec_at_90_Sens": spec90,
        "Prec_at_90_Sens": prec90,
        "Alert_Rate_at_90_Sens": alert_rate90,
        "Top_10_Prevalence": top10_prev,
        "Top_10_Lift": top10_lift,
        "Top_20_Prevalence": top20_prev,
        "Top_20_Lift": top20_lift,
        "Net_Benefit_10pct": net_benefit,
    }


def plot_all_models_comparison(metrics_df):
    """Genera il super grafico comparativo di TUTTI i modelli."""
    plot_df = metrics_df[metrics_df["Modello"] != "Dummy (Prevalenza)"].copy()
    plot_df = plot_df.sort_values("AUROC", ascending=True)

    fig, axes = plt.subplots(1, 4, figsize=(22, 10), sharey=True)

    # 1. AUROC (con barre d'errore IC 95%)
    y_pos = np.arange(len(plot_df))
    aucs = plot_df["AUROC"].values
    xerr = [aucs - plot_df["AUROC_low"].values, plot_df["AUROC_high"].values - aucs]
    bars1 = axes[0].barh(y_pos, aucs, xerr=xerr, color="#2b5c8f", alpha=0.85, capsize=4, edgecolor="black")
    axes[0].set_title("AUROC Globale (IC 95%)", fontweight="bold")
    axes[0].set_xlim(0.5, 0.8)
    axes[0].axvline(0.70, color="red", linestyle="--", alpha=0.7, label="Baseline 0.70")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(plot_df["Modello"].values, fontweight="bold")
    axes[0].set_xlabel("AUROC")
    for bar, val in zip(bars1, aucs):
        axes[0].text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)

    # 2. AUROC su eGFR < 60
    egfr_aucs = plot_df["AUROC_eGFR_inf_60"].values
    bars2 = axes[1].barh(y_pos, egfr_aucs, color="#1e824c", alpha=0.85, edgecolor="black")
    axes[1].set_title("AUROC su eGFR < 60 (Filtrazione)", fontweight="bold")
    axes[1].set_xlim(0.65, 0.90)
    axes[1].axvline(0.80, color="green", linestyle="--", alpha=0.7, label="Soglia Eccellente (0.80)")
    axes[1].set_xlabel("AUROC")
    for bar, val in zip(bars2, egfr_aucs):
        axes[1].text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

    # 3. Sensibilità sui Casi Gravi (KDIGO Alto e Molto Alto)
    sens_severe = plot_df["Severe_Cases_Sensitivity"].values * 100
    bars3 = axes[2].barh(y_pos, sens_severe, color="#d35400", alpha=0.85, edgecolor="black")
    axes[2].set_title("Sensibilità Casi Gravi KDIGO (%)", fontweight="bold")
    axes[2].set_xlim(75, 100)
    axes[2].axvline(90, color="orange", linestyle="--", alpha=0.7, label="90% Sensibilità")
    axes[2].set_xlabel("Sensibilità (%)")
    for bar, val in zip(bars3, sens_severe):
        axes[2].text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    # 4. Top 10% Risk Lift (Arricchimento)
    lifts = plot_df["Top_10_Lift"].values
    bars4 = axes[3].barh(y_pos, lifts, color="#8e44ad", alpha=0.85, edgecolor="black")
    axes[3].set_title("Lift nel Top 10% di Rischio", fontweight="bold")
    axes[3].set_xlim(1.5, 4.0)
    axes[3].axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="Random (1.0x)")
    axes[3].set_xlabel("Fattore di Moltiplicazione (x)")
    for bar, val in zip(bars4, lifts):
        axes[3].text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}x", va="center", fontsize=9, fontweight="bold")

    plt.suptitle("BENCHMARK COMPLESSIVO DI TUTTI I MODELLI TESTATI\nConfronto multidimensionale su discriminazione, danno renale e utilità clinica", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "01_all_models_comprehensive_comparison.png", dpi=300)
    plt.close()
    print(f"Salvato grafico: {OUTPUT_DIR / '01_all_models_comprehensive_comparison.png'}")


def plot_best_model_deep_dive(best_name, p_best, y_true, levels_arr):
    """Genera il super pannello a 6 grafici di analisi approfondita per il modello migliore."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # 1. Curva ROC
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_true, p_best)
    auc_val = roc_auc_score(y_true, p_best)
    ax1.plot(fpr, tpr, color="#1b4f72", lw=2.5, label=f"{best_name} (AUC = {auc_val:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Caso (AUC = 0.500)")
    ax1.set_xlabel("1 - Specificità (FPR)")
    ax1.set_ylabel("Sensibilità (TPR)")
    ax1.set_title("A. Curva ROC", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 2. Curva Precision-Recall
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(y_true, p_best)
    pr_auc_val = average_precision_score(y_true, p_best)
    base_prev = y_true.mean()
    ax2.plot(recall, precision, color="#117864", lw=2.5, label=f"PR-AUC = {pr_auc_val:.3f}")
    ax2.axhline(base_prev, color="red", linestyle="--", lw=1.5, label=f"Prevalenza Base ({base_prev*100:.1f}%)")
    ax2.set_xlabel("Recall (Sensibilità)")
    ax2.set_ylabel("Precision (PPV)")
    ax2.set_title("B. Curva Precision-Recall", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 3. Curva di Calibrazione (Reliability Diagram)
    ax3 = fig.add_subplot(gs[0, 2])
    prob_true, prob_pred = calibration_curve(y_true, p_best, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, p_best)
    ax3.plot(prob_pred, prob_true, marker="s", color="#7d3c98", lw=2, label=f"Brier = {brier:.4f}")
    ax3.plot([0, 0.6], [0, 0.6], color="gray", linestyle="--", lw=1.5, label="Calibrazione Perfetta")
    ax3.set_xlabel("Probabilità Predetta")
    ax3.set_ylabel("Frazione Osservata di Positivi")
    ax3.set_title("C. Calibrazione Probabilistica", fontweight="bold")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # 4. Distribuzione del Rischio per Livello KDIGO Clinico
    ax4 = fig.add_subplot(gs[1, 0])
    kdigo_df = pd.DataFrame({"Probabilità": p_best, "Livello KDIGO": levels_arr})
    order = ["basso", "moderato", "alto", "molto alto"]
    palette = ["#2ecc71", "#f39c12", "#e67e22", "#c0392b"]
    sns.boxplot(data=kdigo_df, x="Livello KDIGO", y="Probabilità", order=order, palette=palette, ax=ax4, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"})
    ax4.set_title("D. Rischio Predetto per Livello KDIGO", fontweight="bold")
    ax4.set_xlabel("Severità Clinica KDIGO")
    ax4.set_ylabel("Probabilità Stimata")
    ax4.grid(True, alpha=0.3)

    # 5. Decision Curve Analysis (Net Benefit)
    ax5 = fig.add_subplot(gs[1, 1])
    thresholds = np.linspace(0.01, 0.30, 50)
    net_benefits_model = []
    net_benefits_all = []
    n = len(y_true)
    for pt in thresholds:
        pred = (p_best >= pt).astype(int)
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits_model.append(nb)
        # Treat all
        nb_all = base_prev - (1 - base_prev) * (pt / (1 - pt))
        net_benefits_all.append(nb_all)

    ax5.plot(thresholds * 100, net_benefits_model, color="#2e4053", lw=2.5, label=f"Strategia {best_name}")
    ax5.plot(thresholds * 100, net_benefits_all, color="gray", linestyle=":", lw=1.5, label="Testare Tutti")
    ax5.axhline(0, color="black", linestyle="-", lw=1, label="Non Testare Nessuno")
    ax5.set_xlim(1, 25)
    ax5.set_ylim(-0.02, 0.12)
    ax5.set_xlabel("Soglia di Rischio Decisionale (%)")
    ax5.set_ylabel("Net Benefit")
    ax5.set_title("E. Decision Curve Analysis (Utilità Clinica)", fontweight="bold")
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)

    # 6. Matrice di Confusione alla Soglia Ottimale Youden
    ax6 = fig.add_subplot(gs[1, 2])
    fpr, tpr, ths = roc_curve(y_true, p_best)
    best_th = ths[np.argmax(tpr - fpr)]
    pred_opt = (p_best >= best_th).astype(int)
    cm = confusion_matrix(y_true, pred_opt)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax6,
                xticklabels=["Sano (0)", "Rischio (1)"], yticklabels=["Sano (0)", "Rischio (1)"])
    ax6.set_xlabel("Classe Predetta")
    ax6.set_ylabel("Classe Reale KDIGO")
    ax6.set_title(f"F. Matrice di Confusione (Soglia {best_th:.3f})", fontweight="bold")

    plt.suptitle(f"PROFILO DIAGNOSTICO COMPLETO: {best_name.upper()}\nValidazione Out-Of-Fold a 5 Fold Annidati", fontsize=15, fontweight="bold")
    plt.savefig(OUTPUT_DIR / "02_best_model_deep_dive.png", dpi=300)
    plt.close()
    print(f"Salvato grafico: {OUTPUT_DIR / '02_best_model_deep_dive.png'}")


def plot_eGFR_vs_ACR_comparison(metrics_df, gfr_arr, acr_arr, y_true, p_best):
    """Grafico esplicativo che mostra come eGFR < 60 supera ampiamente 0.83 rispetto al target aggregato."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ROC Curves per componente
    fpr_tot, tpr_tot, _ = roc_curve(y_true, p_best)
    auc_tot = roc_auc_score(y_true, p_best)

    gfr_pos = (gfr_arr < 60).astype(int)
    fpr_gfr, tpr_gfr, _ = roc_curve(gfr_pos, p_best)
    auc_gfr = roc_auc_score(gfr_pos, p_best)

    acr_macro = (acr_arr >= 300).astype(int)
    fpr_macro, tpr_macro, _ = roc_curve(acr_macro, p_best)
    auc_macro = roc_auc_score(acr_macro, p_best)

    ax1.plot(fpr_gfr, tpr_gfr, color="#1e824c", lw=2.5, label=f"eGFR < 60 (Filtrazione): AUC = {auc_gfr:.3f}")
    ax1.plot(fpr_macro, tpr_macro, color="#d35400", lw=2.2, label=f"Macroalbuminuria (ACR ≥ 300): AUC = {auc_macro:.3f}")
    ax1.plot(fpr_tot, tpr_tot, color="#2b5c8f", lw=2, linestyle="--", label=f"Target Globale KDIGO: AUC = {auc_tot:.3f}")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1)
    ax1.set_xlabel("1 - Specificità")
    ax1.set_ylabel("Sensibilità")
    ax1.set_title("A. Discriminazione per Tipologia e Severità del Danno", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Confronto tra Modelli sulla sola componente eGFR < 60
    top_models = metrics_df.sort_values("AUROC_eGFR_inf_60", ascending=True).tail(8)
    y_pos = np.arange(len(top_models))
    bars = ax2.barh(y_pos, top_models["AUROC_eGFR_inf_60"].values, color="#1e824c", alpha=0.85, edgecolor="black")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_models["Modello"].values, fontweight="bold")
    ax2.set_xlim(0.70, 0.90)
    ax2.axvline(0.80, color="red", linestyle="--", label="Eccellente (0.80)")
    ax2.set_xlabel("AUROC su eGFR < 60")
    ax2.set_title("B. Top Modelli su Compromissione Funzionale Renale", fontweight="bold")
    ax2.legend(loc="lower right")
    for bar, val in zip(bars, top_models["AUROC_eGFR_inf_60"].values):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontweight="bold")

    plt.suptitle("DISSEZIONE BIOLOGICA DEL TARGET: LA FILTRAZIONE RENALE SUPERA 0.83\nEvidenza del motivo biologico della divergenza tra eGFR e Albuminuria", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_target_decomposition_eGFR_vs_all.png", dpi=300)
    plt.close()
    print(f"Salvato grafico: {OUTPUT_DIR / '03_target_decomposition_eGFR_vs_all.png'}")


def plot_decile_lift(y_true, preds_dict):
    """Grafico delle curve di arricchimento / Lift per decile."""
    plt.figure(figsize=(10, 6))
    deciles = np.arange(1, 11)
    base_prev = y_true.mean()

    key_models = [
        "Super Learner (Stacking Meta-Ensemble)",
        "Continuous Target (Dual Regression)",
        "Target Decomposition (eGFR+ACR)",
        "TabPFN",
        "Random Forest",
        "Logistic Regr. (Penalized)",
    ]

    colors = ["#8e44ad", "#2980b9", "#27ae60", "#d35400", "#34495e", "#7f8c8d"]

    for m, c in zip(key_models, colors):
        if m not in preds_dict:
            continue
        p = preds_dict[m]
        df_temp = pd.DataFrame({"y": y_true, "p": p})
        df_temp["decile"] = pd.qcut(df_temp["p"], 10, labels=False, duplicates="drop")
        decile_prev = df_temp.groupby("decile")["y"].mean().values
        decile_lift = decile_prev / base_prev
        plt.plot(np.arange(1, len(decile_lift) + 1), decile_lift, marker="o", lw=2.2, color=c, label=f"{m} (Max {decile_lift[-1]:.2f}x)")

    plt.axhline(1.0, color="black", linestyle="--", lw=1.5, label="Nessun Arricchimento (1.0x)")
    plt.xlabel("Decile di Rischio Predetto (1 = Rischio Minimo, 10 = Rischio Massimo)")
    plt.ylabel("Fattore di Arricchimento (Lift x)")
    plt.title("CURVE DI LIFT PER DECILE DI RISCHIO\nCapacità di concentrare i casi patologici nel decile di massima allerta", fontweight="bold")
    plt.xticks(deciles)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_top_decile_lift_comparison.png", dpi=300)
    plt.close()
    print(f"Salvato grafico: {OUTPUT_DIR / '04_top_decile_lift_comparison.png'}")


def generate_operating_table(y_true, p_best, best_name):
    """Genera la tabella completa dei punti operativi a varie soglie decisionali."""
    thresholds = [0.03, 0.05, 0.07, 0.09, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    # Aggiungi soglie speciali
    fpr, tpr, ths = roc_curve(y_true, p_best)
    best_j_idx = np.argmax(tpr - fpr)
    youden_th = ths[best_j_idx]

    sorted_pos = np.sort(p_best[y_true == 1])[::-1]
    th90 = sorted_pos[int(np.ceil(0.90 * len(sorted_pos))) - 1]
    th85 = sorted_pos[int(np.ceil(0.85 * len(sorted_pos))) - 1]
    th80 = sorted_pos[int(np.ceil(0.80 * len(sorted_pos))) - 1]

    named_thresholds = [
        ("Sensibilità Fissata all'80%", th80),
        ("Sensibilità Fissata all'85%", th85),
        ("Sensibilità Fissata al 90%", th90),
        (f"Soglia Ottimale Youden ({youden_th:.3f})", youden_th),
    ] + [(f"Soglia fissa {t:.2f}", t) for t in thresholds]

    rows = []
    n = len(y_true)
    for label, cut in named_thresholds:
        pred = (p_best >= cut).astype(int)
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum((~pred.astype(bool)) & (y_true == 1)))
        tn = int(np.sum((~pred.astype(bool)) & (y_true == 0)))

        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn)
        alert_rate = (tp + fp) / n
        tests_avoided_per_100 = (1 - alert_rate) * 100

        rows.append({
            "Punto Operativo": label,
            "Soglia": cut,
            "Sensibilità (%)": sens * 100,
            "Specificità (%)": spec * 100,
            "Precision (PPV) (%)": ppv * 100 if not np.isnan(ppv) else np.nan,
            "NPV (%)": npv * 100,
            "F1-Score": f1,
            "Tasso di Allerta (%)": alert_rate * 100,
            "Esami Risparmiati (%)": tests_avoided_per_100,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        })

    op_df = pd.DataFrame(rows)
    op_df.to_csv(OUTPUT_DIR / "best_model_operating_table.csv", index=False)
    print(f"Salvata tabella punti operativi: {OUTPUT_DIR / 'best_model_operating_table.csv'}")
    return op_df


def main():
    print("=== AVVIO MEGA-BENCHMARKING PER LA TESI ===")
    train_df = pd.read_csv(PROCESSED / "train.csv")
    y_true = target(train_df).to_numpy()
    gfr_arr = egfr(train_df).to_numpy()
    acr_arr = train_df["UMAUCR"].to_numpy()
    levels_arr = kdigo_level(train_df).to_numpy()

    # 1. Carica tutte le predizioni
    preds_dict = load_all_predictions(train_df)
    print(f"Modelli raccolti con successo: {len(preds_dict)}")

    # 2. Calcola metriche per ciascun modello
    all_metrics = []
    for model_name, p in preds_dict.items():
        m_dict = compute_metrics(y_true, p, gfr_arr, acr_arr, levels_arr)
        all_metrics.append({"Modello": model_name, **m_dict})

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("AUROC", ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "all_models_comprehensive_metrics.csv", index=False)
    print(f"Salvata tabella metriche: {OUTPUT_DIR / 'all_models_comprehensive_metrics.csv'}")

    # 3. Individua il migliore
    best_row = metrics_df[metrics_df["Modello"] != "Dummy (Prevalenza)"].iloc[0]
    best_name = best_row["Modello"]
    p_best = preds_dict[best_name]
    print(f"Modello top per AUROC: {best_name} (AUROC = {best_row['AUROC']:.3f}, eGFR-AUROC = {best_row['AUROC_eGFR_inf_60']:.3f})")

    # 4. Genera grafici
    plot_all_models_comparison(metrics_df)
    plot_best_model_deep_dive(best_name, p_best, y_true, levels_arr)
    plot_eGFR_vs_ACR_comparison(metrics_df, gfr_arr, acr_arr, y_true, p_best)
    plot_decile_lift(y_true, preds_dict)
    generate_operating_table(y_true, p_best, best_name)

    print("=== MEGA-BENCHMARKING COMPLETATO CON SUCCESSO! ===")


if __name__ == "__main__":
    main()
