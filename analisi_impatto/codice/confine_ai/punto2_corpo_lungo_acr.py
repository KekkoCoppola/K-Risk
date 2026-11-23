"""Punto 2: il corpo cambia già prima di ACR 30?

Esperimento ESPLICATIVO, solo training. Piano fissato prima dei risultati:
analisi_impatto/04_esperimenti/confine_ai/piano_punto2.md

Per ogni esame non renale: residui su età (2° grado), sesso, età×sesso; poi
modello a bastone da hockey y = b0 + b1·x + b2·max(x - c, 0) con x = log(ACR),
c su griglia di 91 punti (5°–95° percentile), confronto con la retta via BIC,
IC di c con bootstrap (1.000 repliche, seed 42; 99,6% per le 12 primarie).
Uscite (solo aggregati): analisi_impatto/04_esperimenti/confine_ai/punto2/
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data import kidney  # noqa: E402

OUT = ROOT / "analisi_impatto/04_esperimenti/confine_ai/punto2"
FIG4 = ROOT / "analisi_impatto/04_esperimenti/non_supervisionato/figure/fig4_punteggio_per_fascia_di_acr.csv"

SEED = 42
B = 1000
NGRID = 91
KDIGO = 30.0
AI_ZONE = (13.0, 26.0)

PRIMARIE = ["Bpsys", "Bpdia", "FPG", "PG2h", "HbA1c", "SUA", "TG", "HDL",
            "waist1", "BMI", "FINS", "HGB"]
CONTROLLI = ["Height", "HbsAg"]
LOG_VARS = {"TG", "FINS", "PG2h", "FPG"}
SEGNO = {v: 1.0 for v in PRIMARIE}
SEGNO["HDL"] = -1.0
PROFILO = [v for v in PRIMARIE if v != "HGB"]
ETICHETTE = {
    "Bpsys": "Pressione sistolica", "Bpdia": "Pressione diastolica",
    "FPG": "Glicemia a digiuno (log)", "PG2h": "Glicemia a 2 ore (log)",
    "HbA1c": "HbA1c", "SUA": "Acido urico", "TG": "Trigliceridi (log)",
    "HDL": "Colesterolo HDL", "waist1": "Girovita", "BMI": "BMI",
    "FINS": "Insulina a digiuno (log)", "HGB": "Emoglobina",
    "Height": "Altezza (controllo)", "HbsAg": "Epatite B (controllo)",
    "profilo": "Profilo complessivo",
}
FASCE = [0, 2, 4, 5, 7, 10, 13, 17, 22, 26, 30, 34, 40, 60, 100, 300, np.inf]

# Colori: corpo (blu), AI (arancio), KDIGO (grigio scuro tratteggiato)
C_CORPO = "#2f6db5"
C_AI = "#d9822b"
C_KDIGO = "#3a3a3a"
C_PUNTI = "#5b5b5b"


def load():
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    n0 = len(df)
    df = df.assign(eGFR=kidney.egfr(df).to_numpy(),
                   female=kidney.is_female(df).astype(float).to_numpy())
    df = df[(df["eGFR"] >= 60) & (df["UMAUCR"] > 0)].copy()
    df["x"] = np.log(df["UMAUCR"])
    return df, n0


def adjust(df, var, extra=()):
    """Residui standardizzati di var su età², sesso, età×sesso (+ extra)."""
    y = df[var].astype(float)
    if var in LOG_VARS:
        y = np.log(y.where(y > 0))
    cols = ["Age", "female"] + list(extra)
    d = pd.concat([y.rename("y"), df[cols], df["x"]], axis=1).dropna()
    a = d["Age"].to_numpy()
    X = [np.ones(len(d)), a, a ** 2, d["female"].to_numpy(), a * d["female"].to_numpy()]
    X += [d[c].to_numpy() for c in extra]
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
    r = d["y"].to_numpy() - X @ beta
    return pd.Series((r - r.mean()) / r.std(), index=d.index)


def hockey(x, y, grid, w=None):
    """Bastone da hockey su griglia, pesi bootstrap w (B×n) o None.

    Ritorna (c_best per replica, SSE best, SSE retta)."""
    W = np.ones((1, len(x))) if w is None else w
    H = np.maximum(x[:, None] - grid[None, :], 0.0)          # n×G
    S1, Sx, Sxx = W.sum(1), W @ x, W @ (x * x)
    Sy, Sxy, Syy = W @ y, W @ (x * y), W @ (y * y)
    Sh, Sxh, Shh, Shy = W @ H, W @ (x[:, None] * H), W @ (H * H), W @ (H * y[:, None])
    nb, G = Sh.shape
    XtX = np.empty((nb, G, 3, 3))
    XtX[..., 0, 0] = S1[:, None]
    XtX[..., 0, 1] = XtX[..., 1, 0] = Sx[:, None]
    XtX[..., 1, 1] = Sxx[:, None]
    XtX[..., 0, 2] = XtX[..., 2, 0] = Sh
    XtX[..., 1, 2] = XtX[..., 2, 1] = Sxh
    XtX[..., 2, 2] = Shh
    Xty = np.stack([np.repeat(Sy[:, None], G, 1), np.repeat(Sxy[:, None], G, 1), Shy], -1)
    beta = np.linalg.solve(XtX + 1e-9 * np.eye(3), Xty[..., None])[..., 0]
    sse = Syy[:, None] - (beta * Xty).sum(-1)
    k = sse.argmin(1)
    det = S1 * Sxx - Sx ** 2
    b1 = (S1 * Sxy - Sx * Sy) / det
    b0 = (Sy - b1 * Sx) / S1
    sse_lin = Syy - b0 * Sy - b1 * Sxy
    return grid[k], sse[np.arange(nb), k], sse_lin, k, beta[np.arange(nb), k]


def analizza(df, y, nome, livello, rng, xcol="x"):
    x = df.loc[y.index, xcol].to_numpy()
    yv = y.to_numpy()
    lo, hi = np.percentile(x, [5, 95])
    grid = np.linspace(lo, hi, NGRID)
    c, sse, sse_lin, k, beta = hockey(x, yv, grid)
    n = len(x)
    bic_h = n * np.log(sse[0] / n) + 4 * np.log(n)
    bic_l = n * np.log(sse_lin[0] / n) + 2 * np.log(n)
    w = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(float)
    cb, *_ = hockey(x, yv, grid, w)
    a = (1 - livello) / 2
    ci = np.quantile(cb, [a, 1 - a])
    perc = lambda v: float((x <= v).mean() * 100)
    return {
        "variabile": nome, "n": n, "asse": xcol,
        "piega": float(np.exp(c[0])), "ic_basso": float(np.exp(ci[0])), "ic_alto": float(np.exp(ci[1])),
        "livello_ic": livello, "percentile_piega": perc(c[0]),
        "percentile_ic_basso": perc(ci[0]), "percentile_ic_alto": perc(ci[1]),
        "delta_bic": float(bic_l - bic_h), "al_bordo": bool(k[0] in (0, NGRID - 1)),
        "pendenza_prima": float(beta[0, 1]), "pendenza_dopo": float(beta[0, 1] + beta[0, 2]),
    }


def per_fascia(df, y, nome):
    acr = df.loc[y.index, "UMAUCR"]
    lab = pd.cut(acr, FASCE, right=False)
    rows = []
    for iv, g in y.groupby(lab, observed=True):
        m, s, n = g.mean(), g.std(), len(g)
        e = 1.96 * s / np.sqrt(n) if n > 1 else np.nan
        mid = np.sqrt(max(iv.left, 1) * iv.right) if np.isfinite(iv.right) else iv.left * 1.5
        rows.append({"variabile": nome, "fascia": f"{iv.left:g}–{iv.right:g}", "centro": mid,
                     "n": n, "media_z": m, "ic_basso": m - e, "ic_alto": m + e})
    return rows


def fit_curve(df, y, row):
    """Curva del bastone da hockey stimato, per il disegno."""
    x = df.loc[y.index, "x"].to_numpy()
    c = np.log(row["piega"])
    X = np.column_stack([np.ones(len(x)), x, np.maximum(x - c, 0)])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 200)
    return np.exp(xs), beta[0] + beta[1] * xs + beta[2] * np.maximum(xs - c, 0)


def decorate(ax, row=None):
    ax.set_xscale("log")
    ax.axvspan(*AI_ZONE, color=C_AI, alpha=0.12, lw=0)
    if row is not None:
        ax.axvspan(row["ic_basso"], row["ic_alto"], color=C_CORPO, alpha=0.12, lw=0)
        ax.axvline(row["piega"], color=C_CORPO, lw=1.5)
    ax.axvline(KDIGO, color=C_KDIGO, lw=1.5, ls="--")
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, n0 = load()
    rng = np.random.default_rng(SEED)
    lvl = 1 - 0.05 / len(PRIMARIE)

    resid = {v: adjust(df, v) for v in PRIMARIE + CONTROLLI}
    Z = pd.concat({v: resid[v] * SEGNO[v] for v in PROFILO}, axis=1).dropna()
    resid["profilo"] = Z.mean(1)

    rows, bands = [], []
    for v in PRIMARIE + CONTROLLI + ["profilo"]:
        livello = lvl if v in PRIMARIE else 0.95
        rows.append(analizza(df, resid[v], v, livello, rng))
        bands += per_fascia(df, resid[v], v)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "P2_punti_di_piega.csv", index=False)
    pd.DataFrame(bands).to_csv(OUT / "P2_medie_per_fascia.csv", index=False)

    # Sensibilità 1: aggiustamento anche per BMI e DM
    sens = []
    for v in PRIMARIE + ["profilo"]:
        if v == "profilo":
            zz = {}
            for u in PROFILO:
                extra = ("DM",) if u in ("BMI", "waist1") else ("BMI", "DM")
                zz[u] = adjust(df, u, extra) * SEGNO[u]
            y = pd.concat(zz, axis=1).dropna().mean(1)
        else:
            extra = ("DM",) if v in ("BMI", "waist1") else ("BMI", "DM")
            y = adjust(df, v, extra)
        r = analizza(df, y, v, lvl if v in PRIMARIE else 0.95, rng)
        r["analisi"] = "agg. età, sesso, BMI, DM"
        sens.append(r)
    # Sensibilità 2: albumina da sola come asse
    df["xa"] = np.log(df["UmALB"].where(df["UmALB"] > 0))
    for v in PRIMARIE + ["profilo"]:
        y = resid[v].loc[resid[v].index.intersection(df.index[df["xa"].notna()])]
        r = analizza(df, y, v, lvl if v in PRIMARIE else 0.95, rng, xcol="xa")
        r["analisi"] = "asse = albumina urinaria (UmALB), unità del dataset"
        sens.append(r)
    pd.DataFrame(sens).to_csv(OUT / "P2_sensibilita.csv", index=False)

    # Decisione preregistrata
    prof = res.set_index("variabile").loc["profilo"]
    prim = res[res["variabile"].isin(PRIMARIE)]
    n_sotto = int(((prim["ic_alto"] < KDIGO) & (prim["delta_bic"] >= 2) & ~prim["al_bordo"]).sum())
    ctrl = res[res["variabile"].isin(CONTROLLI)]
    allarme = bool(((ctrl["delta_bic"] >= 2) & ~ctrl["al_bordo"]).any())
    prof_ok = prof["ic_alto"] < KDIGO and prof["delta_bic"] >= 2 and not prof["al_bordo"]
    if prof_ok and n_sotto >= 3:
        esito = "POSITIVO"
    elif prof["delta_bic"] < 2 or prof["al_bordo"]:
        esito = "GRADUALE"
    elif prof["piega"] >= KDIGO:
        esito = "NEGATIVO"
    else:
        esito = "NON CONCLUSIVO (piega del profilo sotto 30 ma criteri non tutti soddisfatti)"
    pd.DataFrame([{
        "n_training": n0, "n_analisi": len(df), "esito": esito,
        "primarie_con_piega_sotto_30": n_sotto, "allarme_controlli_negativi": allarme,
        "piega_profilo": prof["piega"], "ic_profilo": f"{prof['ic_basso']:.1f}–{prof['ic_alto']:.1f}",
        "delta_bic_profilo": prof["delta_bic"],
    }]).to_csv(OUT / "P2_esito.csv", index=False)

    # ---------- Figure ----------
    fb = pd.DataFrame(bands)
    ai = pd.read_csv(FIG4)
    ai = ai[(ai["split"] == "G") & (ai["configurazione"] == "kmeans_k4")].reset_index(drop=True)
    pr = fb[fb["variabile"] == "profilo"].reset_index(drop=True)

    # Fig. 1: profilo complessivo + punteggio AI, stesse fasce, stessa unità (z)
    netta = lambda rw: rw["delta_bic"] >= 2 and not rw["al_bordo"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    decorate(ax, prof if netta(prof) else None)
    ax.errorbar(pr["centro"], pr["media_z"], yerr=[pr["media_z"] - pr["ic_basso"], pr["ic_alto"] - pr["media_z"]],
                fmt="o", ms=6, color=C_CORPO, ecolor=C_CORPO, elinewidth=1, capsize=0,
                label="Profilo del corpo (11 esami non renali, media z aggiustata)")
    if len(ai) == len(pr):
        ax.errorbar(pr["centro"] * 1.06, ai["punteggio_medio_z"],
                    yerr=[ai["punteggio_medio_z"] - ai["ic_basso"], ai["ic_alto"] - ai["punteggio_medio_z"]],
                    fmt="s", ms=6, color=C_AI, ecolor=C_AI, elinewidth=1, capsize=0,
                    label="Punteggio dell'AI non supervisionata (k-means, split G)")
    if netta(prof):
        xs, ys = fit_curve(df, resid["profilo"], prof)
        nota = f"piega del corpo {prof['piega']:.1f}\n(IC {prof['ic_basso']:.1f}–{prof['ic_alto']:.1f})"
    else:
        xp = df.loc[resid["profilo"].index, "x"].to_numpy()
        bl = np.polyfit(xp, resid["profilo"].to_numpy(), 1)
        xs = np.linspace(np.percentile(xp, 1), np.percentile(xp, 99), 200)
        ys = np.polyval(bl, xs)
        xs = np.exp(xs)
        nota = (f"nessuna piega: retta preferita (ΔBIC {prof['delta_bic']:.1f})\n"
                f"+{bl[0] * np.log(2):.3f} z per ogni raddoppio dell'ACR")
    ax.plot(xs, ys, color=C_CORPO, lw=2)
    ax.text(KDIGO * 1.04, ax.get_ylim()[1] * 0.92, "KDIGO 30", color=C_KDIGO, fontsize=9)
    ax.text(1.3, ax.get_ylim()[1] * 0.62, nota, color=C_CORPO, fontsize=9)
    ax.text(AI_ZONE[0] * 1.02, ax.get_ylim()[1] * 0.92, "zona AI\n13–26", color=C_AI, fontsize=9)
    ax.set_xlabel("ACR (unità del dataset, dichiarata mg/g), scala logaritmica")
    ax.set_ylabel("Deviazioni standard dalla media (z)")
    ax.set_title("Il corpo e l'AI lungo l'albuminuria, a parità di età e sesso (training, eGFR ≥ 60)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_profilo_e_ai.png", dpi=200)
    plt.close(fig)

    # Fig. 2: piccoli multipli
    vars_ = PRIMARIE + CONTROLLI
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), sharex=True)
    for ax, v in zip(axes.flat, vars_):
        row = res.set_index("variabile").loc[v]
        b = fb[fb["variabile"] == v]
        decorate(ax, row if (row["delta_bic"] >= 2 and not row["al_bordo"]) else None)
        ax.errorbar(b["centro"], b["media_z"], yerr=[b["media_z"] - b["ic_basso"], b["ic_alto"] - b["media_z"]],
                    fmt="o", ms=4, color=C_PUNTI, elinewidth=0.8)
        xv = df.loc[resid[v].index, "x"].to_numpy()
        if row["delta_bic"] >= 2 and not row["al_bordo"]:
            xs, ys = fit_curve(df, resid[v], row)
            sub = f"piega {row['piega']:.1f}, ΔBIC {row['delta_bic']:.1f}"
        else:
            bl = np.polyfit(xv, resid[v].to_numpy(), 1)
            g = np.linspace(np.percentile(xv, 1), np.percentile(xv, 99), 200)
            xs, ys = np.exp(g), np.polyval(bl, g)
            sub = f"nessuna piega; {bl[0] * np.log(2):+.3f} z per raddoppio"
        ax.plot(xs, ys, color=C_CORPO, lw=1.5)
        ax.set_title(f"{ETICHETTE[v]}\n{sub}", fontsize=9)
    for ax in axes.flat[len(vars_):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("ACR (log)")
    fig.suptitle("Esami non renali lungo l'ACR (residui z per età e sesso, medie per fascia con IC 95%). "
                 "Linea blu: andamento stimato; arancio: zona AI 13–26; tratteggio: KDIGO 30", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig2_esami_lungo_acr.png", dpi=200)
    plt.close(fig)

    # Fig. 3: forest dei punti di piega
    order = ["profilo"] + PRIMARIE + CONTROLLI
    r = res.set_index("variabile").loc[order[::-1]]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_xscale("log")
    ax.axvspan(*AI_ZONE, color=C_AI, alpha=0.12, lw=0)
    ax.axvline(KDIGO, color=C_KDIGO, lw=1.5, ls="--")
    ypos = np.arange(len(r))
    for yy, (v, row) in zip(ypos, r.iterrows()):
        forte = row["delta_bic"] >= 2 and not row["al_bordo"]
        col = C_CORPO if forte else "#9a9a9a"
        ax.plot([row["ic_basso"], row["ic_alto"]], [yy, yy], color=col, lw=2)
        ax.plot(row["piega"], yy, "o", ms=8 if v == "profilo" else 6, color=col,
                mec="white", mew=1.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([ETICHETTE[v] for v in r.index], fontsize=9)
    ax.set_xlabel("ACR al punto di piega (scala log)")
    ax.set_title("Dove si piega ogni esame. Blu: piega netta (ΔBIC ≥ 2); grigio: andamento graduale.\n"
                 "IC 99,6% per le primarie, 95% per profilo e controlli", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_punti_di_piega.png", dpi=200)
    plt.close(fig)

    print("n training", n0, "| n analisi", len(df))
    print(res[["variabile", "n", "piega", "ic_basso", "ic_alto", "percentile_piega", "delta_bic",
               "al_bordo", "pendenza_prima", "pendenza_dopo"]].round(3).to_string(index=False))
    print("\nSENSIBILITÀ")
    print(pd.DataFrame(sens)[["analisi", "variabile", "piega", "ic_basso", "ic_alto", "percentile_piega",
                               "delta_bic", "al_bordo"]].round(3).to_string(index=False))
    print("\nESITO:", esito, "| primarie con piega sotto 30:", n_sotto, "| allarme controlli:", allarme)


if __name__ == "__main__":
    main()
