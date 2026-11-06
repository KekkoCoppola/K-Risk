"""Genera il grafico comparativo: Tutte le metriche CON SOGLIA e SENZA SOGLIA.
Confronto tra approccio con soglia (Random Forest, LightGBM) e senza soglia (Combinato LGBM, Margine KDIGO, Stacking).
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Carica metriche
METRICS_PATH = Path("analytics/phase_e/metrics.csv")
df = pd.read_csv(METRICS_PATH).set_index("model")

# Selezione modelli rappresentativi
SELECTED_MODELS = [
    ("random_forest", "Random Forest (Con Soglia)", "#2563eb", "soglia"),
    ("lightgbm", "LightGBM (Con Soglia)", "#60a5fa", "soglia"),
    ("combinato_lgbm", "Combinato LGBM (Senza Soglia)", "#059669", "continua"),
    ("margine_kdigo", "Margine KDIGO (Senza Soglia)", "#34d399", "continua"),
    ("stacking", "Stacking Meta-Modello", "#d97706", "ensemble"),
]

# Configurazione stile tipografico
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#475569",
    "axes.titlecolor": "#0f172a",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.color": "#64748b",
    "ytick.color": "#0f172a",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "grid.color": "#f1f5f9",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(16, 12), dpi=300)

# Layout a griglia 2x2 con margini generosi per le etichette
gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32, left=0.20, right=0.96, top=0.89, bottom=0.08)

# Intestazione globale
fig.text(
    0.5, 0.96,
    "CONFRONTO METODOLOGICO COMPLETO: VALUTAZIONE SENZA SOGLIA vs CON SOGLIA",
    fontsize=13, fontweight="bold", color="#0f172a", ha="center"
)
fig.text(
    0.5, 0.935,
    "Modelli addestrati su Target Binario KDIGO (Con Soglia) vs Modelli a Regressione Continua non vincolata (Senza Soglia)",
    fontsize=10.5, color="#475569", ha="center"
)

# -------------------------------------------------------------------------------------------------
# PANNELLO 1: METRICHE GLOBALI SENZA SOGLIA (Threshold-Free)
# -------------------------------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("1. METRICHE GLOBALI SENZA SOGLIA (Threshold-Free)\nOrdinamento del rischio su tutta la distribuzione", fontsize=10.5, pad=10)

tf_metrics = [
    ("auc", "AUROC (DeLong)"),
    ("pr_auc", "PR-AUC (baseline 0.098)"),
    ("gini", "Indice di Gini (2*AUC - 1)"),
    ("ks", "Statistica KS"),
    ("kdigo_concordance", "Concordanza 4 Livelli KDIGO"),
]

y_pos = np.arange(len(tf_metrics))
bar_height = 0.15
offsets = np.linspace(-bar_height * 2, bar_height * 2, len(SELECTED_MODELS))

for idx, (m_key, m_name, color, _) in enumerate(SELECTED_MODELS):
    row = df.loc[m_key]
    vals = [row[col] for col, _ in tf_metrics]
    bars = ax1.barh(y_pos + offsets[idx], vals, height=bar_height, color=color, label=m_name, alpha=0.9)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.008, bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                 va="center", ha="left", fontsize=7.2, color="#1e293b", fontweight="bold")

ax1.set_yticks(y_pos)
ax1.set_yticklabels([m[1] for m in tf_metrics], fontweight="semibold")
ax1.set_xlim(0, 0.88)
ax1.grid(axis="x", linestyle="--", alpha=0.7)
ax1.legend(loc="lower right", fontsize=8, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

# -------------------------------------------------------------------------------------------------
# PANNELLO 2: DECOMPOSIZIONE SENZA SOGLIA - Dove risiede il tetto informativo
# -------------------------------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("2. DISCRIMINAZIONE PER COMPONENTE (AUROC)\nTettonica informativa: eGFR vs Sola Albuminuria", fontsize=10.5, pad=10)

comp_metrics = [
    ("auc_egfr", "eGFR < 60 (Filtrazione)"),
    ("auc", "Target Composito KDIGO"),
    ("auc_albuminuria_only", "Sola Albuminuria (uACR ≥ 30)\n(91% dei positivi di screening)"),
]

y_comp = np.arange(len(comp_metrics))
for idx, (m_key, m_name, color, _) in enumerate(SELECTED_MODELS):
    row = df.loc[m_key]
    vals = [row[col] for col, _ in comp_metrics]
    bars = ax2.barh(y_comp + offsets[idx], vals, height=bar_height, color=color, alpha=0.9)
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 0.008, bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                 va="center", ha="left", fontsize=7.2, color="#1e293b", fontweight="bold")

ax2.axvline(0.70, color="#94a3b8", linestyle=":", linewidth=1.4, label="Plateau Albuminuria (~0.70)")
ax2.axvline(0.85, color="#64748b", linestyle=":", linewidth=1.4, label="Capacità Filtrato (~0.85)")
ax2.set_yticks(y_comp)
ax2.set_yticklabels([c[1] for c in comp_metrics], fontweight="semibold")
ax2.set_xlim(0.55, 0.96)
ax2.grid(axis="x", linestyle="--", alpha=0.7)
ax2.legend(loc="lower right", fontsize=8, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

# -------------------------------------------------------------------------------------------------
# PANNELLO 3: METRICHE CON SOGLIA (Punto Operativo di Triage a Sensibilità 90%)
# -------------------------------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title("3. METRICHE CON SOGLIA (Operating Point: Sens. ≥ 90%)\nRegola di triage clinico cautelativo di primo livello", fontsize=10.5, pad=10)

th_metrics = [
    ("sens90_sensitivity", "Sensibilità (Recall fissato)"),
    ("sens90_specificity", "Specificità"),
    ("sens90_npv", "Valore Predittivo Negativo (VPN)"),
    ("sens90_ppv", "Valore Predittivo Positivo (VPP)"),
    ("sens90_f2", "Score F2 (peso al recall)"),
]

y_th = np.arange(len(th_metrics))
for idx, (m_key, m_name, color, _) in enumerate(SELECTED_MODELS):
    row = df.loc[m_key]
    vals = [row[col] for col, _ in th_metrics]
    bars = ax3.barh(y_th + offsets[idx], vals, height=bar_height, color=color, alpha=0.9)
    for bar in bars:
        w = bar.get_width()
        txt = f"{w*100:.1f}%" if w > 0.05 and w <= 1.0 else f"{w:.3f}"
        ax3.text(w + 0.012, bar.get_y() + bar.get_height() / 2, txt,
                 va="center", ha="left", fontsize=7.2, color="#1e293b", fontweight="bold")

ax3.set_yticks(y_th)
ax3.set_yticklabels([t[1] for t in th_metrics], fontweight="semibold")
ax3.set_xlim(0, 1.18)
ax3.grid(axis="x", linestyle="--", alpha=0.7)

# -------------------------------------------------------------------------------------------------
# PANNELLO 4: SICUREZZA SUI CASI GRAVI & EFFICIENZA (CON SOGLIA)
# -------------------------------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title("4. SICUREZZA CLINICA & UTILITÀ (CON SOGLIA)\nIntercettazione casi critici e test specialistici evitati", fontsize=10.5, pad=10)

sec_metrics = [
    ("sens90_severe_sensitivity", "Recall Casi Gravi (Alto + Molto Alto)", "%"),
    ("sens90_sensitivity_molto_alto", "Recall Casi Molto Alto (Stadio critico)", "%"),
    ("sens90_tests_avoided_per_100", "Esami specialistici evitati (su 100)", "n"),
    ("sens90_alert_rate", "Quota Allerta (% Inviati al 2° livello)", "%"),
]

y_sec = np.arange(len(sec_metrics))
for idx, (m_key, m_name, color, _) in enumerate(SELECTED_MODELS):
    row = df.loc[m_key]
    vals = []
    for col, _, kind in sec_metrics:
        v = row[col]
        vals.append(v / 100.0 if kind == "n" else v)
    bars = ax4.barh(y_sec + offsets[idx], vals, height=bar_height, color=color, alpha=0.9)
    for b_idx, bar in enumerate(bars):
        w = bar.get_width()
        col, _, kind = sec_metrics[b_idx]
        raw_val = row[col]
        txt = f"{raw_val:.1f} su 100" if kind == "n" else f"{raw_val*100:.1f}%"
        ax4.text(w + 0.012, bar.get_y() + bar.get_height() / 2, txt,
                 va="center", ha="left", fontsize=7.2, color="#1e293b", fontweight="bold")

ax4.set_yticks(y_sec)
ax4.set_yticklabels([s[1] for s in sec_metrics], fontweight="semibold")
ax4.set_xlim(0, 1.25)
ax4.grid(axis="x", linestyle="--", alpha=0.7)

# Nota esplicativa in basso
fig.text(
    0.20, 0.02,
    "Takeaway clinico: Rimuovere la soglia (regressione continua senza vincoli KDIGO) porta l'AUROC a 0.711 vs 0.705 con soglia (Δ = +0.0056, p = 0.57, non significativo).\n"
    "Spiegazione biologica: L'eGFR dal sangue è altamente discriminabile (0.82-0.85), mentre l'albuminuria isolata ha un tetto biologico a ~0.686 per tutti i modelli.\n"
    "A soglia clinica cautelativa (Sensibilità 90%), entrambi gli approcci intercettano 90-95% dei casi critici evitando 23-26 esami specialistici inutili ogni 100 persone.",
    fontsize=8.5, color="#334155", style="italic", linespacing=1.4
)

# Salva immagine nel repo
out_repo = Path("analytics/phase_e/10_con_e_senza_soglia.png")

fig.savefig(out_repo, dpi=300)
plt.close(fig)

print("Immagine rigenerata con successo!")
