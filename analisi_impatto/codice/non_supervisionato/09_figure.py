"""Figure del filone "non supervisionato" (parte A, split G, split T).

Ogni figura salva accanto un CSV con i numeri disegnati (solo aggregati, nessuna riga di
paziente). Nessuna nuova stima: si leggono i CSV già prodotti e i punteggi salvati.

Uscite: `analisi_impatto/04_esperimenti/non_supervisionato/figure/`.
"""
import importlib.util
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import prep  # noqa: E402

ROOT = prep.ROOT
from src.data import kidney  # noqa: E402
from src.models.evaluation import delong  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load("parteA", "01_parteA.py")
B = _load("parteB", "03_parteB.py")
KPrototypes = B.KPrototypes  # il pickle di T cerca la classe in __main__

EXP = ROOT / "analisi_impatto/04_esperimenti/non_supervisionato"
OUT = EXP / "figure"
# palette di riferimento (skill dataviz, references/palette.md), ordine fisso
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8f8e8a"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True})
NOMI = {"autoencoder": "autoencoder", "kmeans_k4": "k-means, k = 4", "kmeans_gap": "k-means, gap",
        "kmeans_k2": "k-means, k = 2", "mahalanobis": "Mahalanobis", "gmm_k4": "mistura gaussiana, k = 4",
        "gmm_k2": "mistura gaussiana, k = 2", "gmm_bic": "mistura gaussiana, BIC"}


def num(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def save(fig, name, data):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}.png")


def fig1_parte_a():
    d = pd.read_csv(EXP / "parteA/A2_conteggi_e_bootstrap.csv")
    rows = []
    for reg in ["R-a_DM", "R-b_DM"]:
        g = d[d.regola == reg]
        rows.append({"regola": reg, "chi_sceglie": "regola del medico", "gruppo": "regola",
                     "testati": int(g.testati.iat[0]), "positivi": int(g.pos_regola.iat[0])})
        for p, lab, grp in [("solo_eta", "solo età (i più anziani)", "baseline semplice"),
                            ("clinico_logistica", "logistica con 6 variabili del clinico", "baseline semplice"),
                            ("random_forest", "Random Forest (Fase A)", "modello AI"),
                            ("tabpfn", "TabPFN (Fase D)", "modello AI")]:
            r = g[g.punteggio == p].iloc[0]
            rows.append({"regola": reg, "chi_sceglie": lab, "gruppo": grp, "testati": int(r.testati),
                         "positivi": int(r.pos_punteggio), "diff_contro_regola": int(r["diff"]),
                         "ic_basso": r.ic_basso, "ic_alto": r.ic_alto})
    data = pd.DataFrame(rows)
    col = {"regola": GRAY, "baseline semplice": ORANGE, "modello AI": BLUE}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, reg, title in zip(axes, ["R-a_DM", "R-b_DM"],
                              ["Regola «diabete o ipertensione nota»", "Regola larga (R-b)"]):
        g = data[data.regola == reg].iloc[::-1]
        ax.barh(g.chi_sceglie, g.positivi, color=[col[x] for x in g.gruppo], height=0.6, edgecolor=SURF, linewidth=2)
        for yv, v in enumerate(g.positivi):
            ax.text(v + 2, yv, str(v), va="center", color=INK, fontsize=9)
        ax.axvline(g[g.gruppo == "regola"].positivi.iat[0], color=INK2, lw=1, ls="--")
        ax.set_title(f"{title}: {g.testati.iat[0]} testati su 4.350", fontsize=10, loc="left")
        ax.set_xlabel("malati trovati (su 425)")
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in col.values()]
    fig.legend(handles, list(col), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Parte A [PREDITTIVO]: a parità di persone testate, quanti malati trova chi sceglie", x=0.01,
                 y=1.04, ha="left", fontsize=11)
    save(fig, "fig1_parteA_malati_trovati", data)


def fig2_g_auroc():
    b5 = pd.read_csv(EXP / "parteB/G/B5_confronto_kdigo.csv")
    b5 = b5[b5.configurazione != "gmm_bic"]  # identica a gmm_k2 (il BIC sceglie k = 2 in tutti i fold)
    rows = [{"modello": NOMI[r.configurazione], "gruppo": "non supervisionato su G (senza etichetta)",
             "auroc": r.auroc, "ic_basso": r.ic_basso, "ic_alto": r.ic_alto} for r in b5.itertuples()]
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    y = kidney.target(df).to_numpy()
    p, _ = A.load_scores("phase_a/main/random_forest", len(y))
    a, lo, hi = delong(y, p)
    rows.append({"modello": "Random Forest Fase A (74 colonne, senza esami renali)",
                 "gruppo": "supervisionato senza esami renali", "auroc": a, "ic_basso": lo, "ic_alto": hi})
    b7 = pd.read_csv(EXP / "parteB/G/B7_supervisionato_auroc.csv")
    lab = {"xgboost": "XGBoost su G", "random_forest": "Random Forest su G", "lr_penalized": "elastic net su G"}
    for r in b7.itertuples():
        rows.append({"modello": lab[r.modello], "gruppo": "supervisionato su G (con etichetta)",
                     "auroc": r.auroc, "ic_basso": r.ic_basso, "ic_alto": r.ic_alto})
    data = pd.DataFrame(rows).sort_values("auroc")
    col = {"non supervisionato su G (senza etichetta)": BLUE, "supervisionato senza esami renali": ORANGE,
           "supervisionato su G (con etichetta)": AQUA}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, r in enumerate(data.itertuples()):
        ax.plot([r.ic_basso, r.ic_alto], [i, i], color=col[r.gruppo], lw=2)
        ax.plot(r.auroc, i, "o", ms=8, color=col[r.gruppo], mec=SURF, mew=2)
        ax.text(r.ic_alto + 0.006, i, num(r.auroc), va="center", fontsize=9)
    ax.set_yticks(range(len(data)), data.modello)
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xlim(0.48, 1.04)
    ax.set_xlabel("AUROC contro KDIGO (IC 95% DeLong), 4.350 soggetti, 5 fold")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8) for c in col.values()]
    ax.legend(handles, list(col), loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Split G [ESPLICATIVO]: senza etichetta l'AI arriva a 0,71, come il supervisionato senza reni;\n"
                 "con l'etichetta, sugli stessi dati, ricostruisce KDIGO (0,998)", loc="left", fontsize=10)
    save(fig, "fig2_G_auroc_con_e_senza_etichetta", data)


def fig3_g_direzioni():
    dep = pd.read_csv(EXP / "xai/G/C1_direzione_shap.csv")
    dep = dep[dep.configurazione != "gmm_bic"]
    data = (dep.groupby(["variabile", "configurazione"]).spearman_valore_shap
            .agg(media="mean", minimo="min", massimo="max").reset_index())
    data = data[data.variabile.isin(["UmALB", "UCRE", "SCRE"])]
    verso = {"UmALB": ("albumina urinaria", "il clinico: più alta = più malato (+)"),
             "UCRE": ("creatinina urinaria", "nel rapporto ACR sta al denominatore (−)"),
             "SCRE": ("creatinina nel sangue", "il clinico: più alta = più malato (+)")}
    order = ["autoencoder", "kmeans_k4", "kmeans_gap", "kmeans_k2", "mahalanobis", "gmm_k4", "gmm_k2"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, v in zip(axes, ["UmALB", "UCRE", "SCRE"]):
        g = data[data.variabile == v].set_index("configurazione").loc[order[::-1]]
        for i, (_, r) in enumerate(g.iterrows()):
            colr = BLUE if (r.media > 0) == (v != "UCRE") else ORANGE
            ax.plot([r.minimo, r.massimo], [i, i], color=colr, lw=2)
            ax.plot(r.media, i, "o", ms=8, color=colr, mec=SURF, mew=2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(range(len(g)), [NOMI[c] for c in g.index])
        ax.set_xlim(-1, 1)
        ax.set_title(f"{verso[v][0]} ({v})\n{verso[v][1]}", fontsize=9.5, loc="left")
        ax.set_xlabel("Spearman fra valore e contributo SHAP")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8) for c in (BLUE, ORANGE)]
    fig.legend(handles, ["stesso verso del clinico", "verso opposto"], loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Split G [ESPLICATIVO, lettura post-hoc]: in che verso l'AI senza etichetta usa gli esami renali "
                 "(punto = media, barra = min–max sui 5 fold)", x=0.01, y=1.08, ha="left", fontsize=10.5)
    save(fig, "fig3_G_verso_degli_esami_renali", data)


EDGES = [0, 2, 4, 5, 7, 10, 13, 17, 22, 26, 30, 34, 40, 60, 100, 300, np.inf]


def fig4_acr():
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    acr = pd.to_numeric(df["UMAUCR"], errors="coerce").to_numpy()
    labels = [f"{int(a)}–{int(b)}" if np.isfinite(b) else f"≥ {int(a)}" for a, b in zip(EDGES[:-1], EDGES[1:])]
    band = pd.cut(acr, EDGES, right=False, labels=labels)
    rows = []
    for split in ["G", "T"]:
        z = np.load(prep.CACHE / f"{split}_out" / "punteggi_oof.npz")
        for cfg in ["kmeans_k4", "autoencoder"]:
            s = z[f"s_{cfg}"].astype(float).copy()
            for k in range(5):
                r = z["fold"] == k
                s[r] = (s[r] - s[r].mean()) / (s[r].std() or 1.0)
            g = pd.DataFrame({"fascia": band, "z": s}).groupby("fascia", observed=False).z
            for lab, m, sd, n in zip(labels, g.mean(), g.std(), g.size()):
                half = 1.96 * sd / np.sqrt(n)  # IC 95% approssimato (normale) della media
                rows.append({"split": split, "configurazione": cfg, "fascia_acr_mg_g": lab, "n": int(n),
                             "punteggio_medio_z": m, "ic_basso": m - half, "ic_alto": m + half})
    data = pd.DataFrame(rows)
    ymax = float(data.ic_alto.max()) + 0.3
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for ax, split in zip(axes, ["G", "T"]):
        for cfg, c in [("kmeans_k4", BLUE), ("autoencoder", ORANGE)]:
            g = data[(data.split == split) & (data.configurazione == cfg)]
            xx = np.arange(len(g)) + (0.12 if cfg == "autoencoder" else -0.12)
            ax.vlines(xx, g.ic_basso, g.ic_alto, color=c, lw=1, alpha=0.6)
            ax.plot(xx, g.punteggio_medio_z, "-o", color=c, lw=2, ms=6, mec=SURF, mew=1.5, label=NOMI[cfg])
        xk = labels.index("30–34") - 0.5
        ax.axvline(xk, color=INK2, ls="--", lw=1)
        ax.text(xk + 0.15, ymax - 0.1, "soglia KDIGO: 30", fontsize=8.5, color=INK2, va="top")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(labels)), labels, rotation=55, fontsize=8)
        ax.set_xlabel("fascia di ACR (mg/g)")
        ax.set_title(f"Split {split}" + (" (78 colonne)" if split == "G" else " (tutte le colonne)"), loc="left",
                     fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("punteggio medio (z dentro il fold)")
    axes[0].set_ylim(float(data.ic_basso.min()) - 0.1, ymax)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("[ESPLICATIVO, post-hoc] Punteggio medio dell'AI senza etichetta per fascia di ACR (barre: IC 95% della media)",
                 x=0.01, ha="left", fontsize=10.5)
    save(fig, "fig4_punteggio_per_fascia_di_acr", data)


def fig5_t_colonna():
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    y = kidney.target(df).to_numpy()
    e = pd.to_numeric(df["EUGFR90UACR01"], errors="coerce").fillna(-1).to_numpy()
    name = {0: "0", 1: "1", -1: "vuota"}
    tab = pd.DataFrame([{"pannello": "a", "EUGFR90UACR01": name[v], "y": yy,
                         "soggetti": int(((e == v) & (y == yy)).sum())} for v in (0, 1, -1) for yy in (0, 1)])
    clus = []
    for k in range(5):
        pack = joblib.load(prep.CACHE / "T_out" / f"modelli_outer{k}.joblib")
        _, X = prep.load("T", f"outer{k}")
        lab = pack["models"]["kmeans"]["k4"].predict(X.to_numpy())
        yy, ee = y[X.index], e[X.index]
        best = max(np.unique(lab), key=lambda c: yy[lab == c].mean())
        c = lab == best
        clus.append({"pannello": "b", "fold": k, "soggetti_nel_cluster": int(c.sum()),
                     "malati_nel_cluster": int(yy[c].sum()),
                     "con_EUGFR90UACR01_1_nel_cluster": int((ee[c] == 1).sum()), "malati_nel_fold": int(yy.sum())})
    cl = pd.DataFrame(clus)
    data = pd.concat([tab, cl], ignore_index=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1.4]})
    x = np.arange(3)
    for j, (yy, c, nm) in enumerate([(0, GRAY, "sani (y = 0)"), (1, ORANGE, "malati (y = 1)")]):
        v = [tab[(tab.EUGFR90UACR01 == s) & (tab.y == yy)].soggetti.iat[0] for s in ("0", "1", "vuota")]
        bars = a1.bar(x + (j - 0.5) * 0.38, v, width=0.36, color=c, label=nm, edgecolor=SURF, linewidth=2)
        for b, val in zip(bars, v):
            a1.text(b.get_x() + b.get_width() / 2, val + 40, f"{val:,}".replace(",", "."), ha="center", fontsize=9)
    a1.set_xticks(x, ["colonna = 0", "colonna = 1", "colonna vuota"])
    a1.set_ylabel("soggetti del training")
    a1.legend(frameon=False, fontsize=9)
    a1.set_title("a) Il file contiene già la diagnosi: EUGFR90UACR01\n(«eGFR < 60 or UACR ≥ 30» nel dizionario)",
                 loc="left", fontsize=10)
    a1.grid(axis="x", visible=False)
    xs = np.arange(5)
    for j, (col_, c, nm) in enumerate([("soggetti_nel_cluster", GRAY, "soggetti nel gruppo"),
                                        ("con_EUGFR90UACR01_1_nel_cluster", BLUE, "di cui con la colonna = 1"),
                                        ("malati_nel_cluster", ORANGE, "di cui malati")]):
        bars = a2.bar(xs + (j - 1) * 0.27, cl[col_], width=0.25, color=c, label=nm, edgecolor=SURF, linewidth=2)
        for b, val in zip(bars, cl[col_]):
            a2.text(b.get_x() + b.get_width() / 2, val + 2, str(val), ha="center", fontsize=8)
    a2.set_xticks(xs, [f"fold {k}" for k in range(5)])
    a2.set_ylabel("soggetti")
    a2.legend(frameon=False, fontsize=9, loc="upper left")
    a2.set_title("b) Il gruppo «più malato» trovato da k-means (k = 4) senza etichetta, in ogni fold\n"
                 "(85 malati per fold)", loc="left", fontsize=10)
    a2.grid(axis="x", visible=False)
    fig.suptitle("Split T [ESPLICATIVO]: l'AI senza etichetta raggruppa chi ha la diagnosi già scritta nel file",
                 x=0.01, y=1.06, ha="left", fontsize=10.5)
    save(fig, "fig5_T_diagnosi_gia_scritta", data)


def fig6_t_ablazione():
    ab = pd.read_csv(EXP / "parteB/T/posthoc_ablazione_target.csv")
    b5g = pd.read_csv(EXP / "parteB/G/B5_confronto_kdigo.csv").set_index("configurazione")
    var = [("solo_colonne_del_target", "solo le 18 colonne renali\n(esami + diagnosi scritte)"),
           ("tutte", "tutte le 188\ncolonne"),
           ("senza_le_14_assenti_in_G", "senza le 14 diagnosi\nscritte (restano\ngli esami grezzi)"),
           ("senza_colonne_del_target", "senza nessuna\ndelle 18")]
    rows = []
    for cfg in ["kmeans_k4", "autoencoder"]:
        for v, lab in var:
            r = ab[(ab.configurazione == cfg) & (ab.variante == v)].iloc[0]
            rows.append({"configurazione": cfg, "variante": v, "etichetta": lab.replace("\n", " "),
                         "auroc_media_fold": r.auroc_media_fold, "auroc_min_fold": r.auroc_min_fold,
                         "auroc_max_fold": r.auroc_max_fold, "auroc_G_media_fold": b5g.loc[cfg, "auroc_media_fold"]})
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    cols = [AQUA, BLUE, ORANGE, GRAY]
    for ax, cfg in zip(axes, ["kmeans_k4", "autoencoder"]):
        g = data[data.configurazione == cfg]
        bars = ax.bar(range(4), g.auroc_media_fold, color=cols, width=0.6, edgecolor=SURF, linewidth=2)
        ax.errorbar(range(4), g.auroc_media_fold, yerr=[g.auroc_media_fold - g.auroc_min_fold,
                                                        g.auroc_max_fold - g.auroc_media_fold],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        for b, val in zip(bars, g.auroc_media_fold):
            ax.text(b.get_x() + b.get_width() / 2, 0.53, num(val), ha="center", fontsize=10, color="white",
                    fontweight="bold")
        gv = g.auroc_G_media_fold.iat[0]
        ax.axhline(gv, color=INK, ls="--", lw=1)
        ax.text(1.38, gv + 0.012, f"split G: {num(gv)}", ha="center", fontsize=8.5,
                bbox={"facecolor": SURF, "edgecolor": "none", "pad": 1})
        ax.set_xticks(range(4), [lab for _, lab in var], fontsize=8)
        ax.set_ylim(0.5, 1.05)
        ax.set_title(NOMI[cfg], loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("AUROC contro KDIGO\n(media dei 5 fold; barra = min–max)")
    fig.suptitle("Split T [ESPLICATIVO, ablazione post-hoc]: stessi modelli, punteggio ricalcolato guardando "
                 "solo alcune colonne", x=0.01, ha="left", fontsize=10.5)
    save(fig, "fig6_T_da_dove_viene_lo_0_84", data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_parte_a()
    fig2_g_auroc()
    fig3_g_direzioni()
    fig4_acr()
    fig5_t_colonna()
    fig6_t_ablazione()


if __name__ == "__main__":
    main()
