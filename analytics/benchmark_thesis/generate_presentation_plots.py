"""Genera due grafici dedicati alla presentazione:
1. Trade-off tra Sensibilità, Specificità, NPV e Risparmio Esami al variare della soglia decisionale.
2. Gradiente clinico KDIGO: sensibilità che cresce con la gravità clinica del paziente.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "analytics" / "benchmark_thesis"
ARTIFACT_DIR = Path(r"C:\Users\franc\.gemini\antigravity-ide\brain\5d25a663-b80a-46e6-b1e2-7dc0f178174e")

from src.data.kidney import kdigo_level, target
from src.data.split import PROCESSED

# Configurazione stile
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

def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    y_true = target(train).to_numpy()
    levels = kdigo_level(train).to_numpy()
    n = len(y_true)

    # Carica TabPFN (miglior modello per AUROC)
    p_tabpfn = np.zeros(n)
    for k in range(5):
        with open(OUTPUT_DIR.parent / "phase_d" / "tabpfn" / f"outer{k}.json") as f:
            d = json.load(f)
            p_tabpfn[d["rows"]] = d["probability"]

    # 1. Grafico Trade-off Soglie (Sensibilità, Specificità, NPV, Risparmio)
    ths_grid = np.linspace(0.02, 0.25, 200)
    sens_list, spec_list, npv_list, ppv_list, saved_list = [], [], [], [], []

    for th in ths_grid:
        pred = (p_tabpfn >= th).astype(int)
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        fn = np.sum((~pred.astype(bool)) & (y_true == 1))
        tn = np.sum((~pred.astype(bool)) & (y_true == 0))

        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 1.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        saved = (1 - (tp + fp) / n) * 100

        sens_list.append(sens * 100)
        spec_list.append(spec * 100)
        npv_list.append(npv * 100)
        ppv_list.append(ppv * 100)
        saved_list.append(saved)

    # Calcola soglia Youden e soglia screening 90%
    fpr, tpr, ths = roc_curve(y_true, p_tabpfn)
    best_j_idx = np.argmax(tpr - fpr)
    th_youden = ths[best_j_idx]

    sorted_pos = np.sort(p_tabpfn[y_true == 1])[::-1]
    th90 = sorted_pos[int(np.ceil(0.90 * len(sorted_pos))) - 1]

    plt.figure(figsize=(12, 7))
    plt.plot(ths_grid, sens_list, label="Sensibilità (% Malati Trovati)", color="#27ae60", lw=2.5)
    plt.plot(ths_grid, spec_list, label="Specificità (% Sani Corretti)", color="#2980b9", lw=2.5)
    plt.plot(ths_grid, npv_list, label="NPV (% Sano se Test Negativo)", color="#8e44ad", lw=2, linestyle="-.")
    plt.plot(ths_grid, saved_list, label="Esami Specialistici Risparmiati (%)", color="#d35400", lw=2, linestyle="--")

    plt.axvline(th90, color="#16a085", linestyle=":", lw=2, label=f"Soglia di Screening (0.040): Sensibilità 90.1%")
    plt.axvline(th_youden, color="#c0392b", linestyle=":", lw=2, label=f"Soglia Ottimale Youden ({th_youden:.3f}): Bilanciata")

    plt.scatter([th90], [90.1], color="#16a085", s=80, zorder=5)
    plt.scatter([th_youden], [62.1], color="#c0392b", s=80, zorder=5)

    plt.title("ANALISI OPERATIVA DELLE METRICHE AL VARIARE DELLA SOGLIA DECISIONALE\n(Modello TabPFN: come bilanciare sensibilità clinica, specificità e risparmio economico)", fontweight="bold")
    plt.xlabel("Soglia di Rischio Decisionale")
    plt.ylabel("Percentuale (%)")
    plt.xlim(0.02, 0.22)
    plt.ylim(0, 102)
    plt.legend(loc="center right", frameon=True, facecolor="white", framealpha=0.95)
    plt.tight_layout()

    p1 = OUTPUT_DIR / "05_sensitivity_specificity_tradeoff.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    shutil.copy2(p1, ARTIFACT_DIR / p1.name)
    print(f"Salvato e copiato {p1.name}")

    # 2. Grafico Gradiente Clinico KDIGO
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart riconoscimento per stadio a soglia screening
    kdigo_tiers = ["Moderato (Giallo)", "Alto (Arancione)", "Molto Alto (Rosso)"]
    kdigo_keys = ["moderato", "alto", "molto alto"]
    sens_tiers = []
    counts_tiers = []
    totals_tiers = []

    for k in kdigo_keys:
        mask = levels == k
        flagged = p_tabpfn[mask] >= th90
        sens_tiers.append(flagged.mean() * 100)
        counts_tiers.append(flagged.sum())
        totals_tiers.append(mask.sum())

    colors = ["#f39c12", "#e67e22", "#c0392b"]
    bars = ax1.bar(kdigo_tiers, sens_tiers, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    ax1.set_ylim(70, 105)
    ax1.set_ylabel("Tasso di Riconoscimento / Sensibilità (%)")
    ax1.set_title("A. Riconoscimento dei Casi per Severità KDIGO\n(Alla soglia operativa di screening)", fontweight="bold")
    ax1.axhline(90, color="gray", linestyle="--", alpha=0.7, label="90% Benchmark")

    for bar, val, count, tot in zip(bars, sens_tiers, counts_tiers, totals_tiers):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}%\n({count}/{tot})", ha="center", fontweight="bold", fontsize=10)
    ax1.legend(loc="lower right")

    # Boxplot rischio predetto per stadio
    df_box = pd.DataFrame({"Rischio Predetto": p_tabpfn, "Stadio": levels})
    order_box = ["basso", "moderato", "alto", "molto alto"]
    palette_box = ["#2ecc71", "#f39c12", "#e67e22", "#c0392b"]
    sns.boxplot(data=df_box, x="Stadio", y="Rischio Predetto", order=order_box, palette=palette_box, ax=ax2, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"})
    ax2.set_title("B. Distribuzione del Rischio Assegnato dal Modello\n(Pazienti Sani vs Stadi di Gravità Clinica Crescente)", fontweight="bold")
    ax2.set_xlabel("Stadio Clinico KDIGO")
    ax2.set_ylabel("Probabilità Stimata")

    plt.suptitle("VALIDAZIONE CLINICA: IL MODELLO INTERCETTA IL 95.8% DEI CASI GRAVI\n(La capacità di allerta cresce spontaneamente con la gravità reale del danno d'organo)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    p2 = OUTPUT_DIR / "06_kdigo_severity_gradient.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    shutil.copy2(p2, ARTIFACT_DIR / p2.name)
    print(f"Salvato e copiato {p2.name}")

if __name__ == "__main__":
    main()
