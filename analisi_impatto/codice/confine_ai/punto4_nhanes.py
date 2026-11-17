"""Punto 4: replica esterna su NHANES 2005–2018 con mortalità collegata (2019).

Piano fissato prima: analisi_impatto/04_esperimenti/confine_ai/piano_punto4.md
4a ESPLICATIVO: il rischio di morte ha uno scalino a ACR 30?
4b ESPLICATIVO: uomini e donne, albumina e creatinina urinaria separate.
4c PREDITTIVO: replica del +22 (AI senza esami renali contro regola diabete/ipertensione noti).
Modello di Cox (Breslow) implementato qui con numpy (nessuna libreria di sopravvivenza nel venv).
Solo aggregati in uscita.
"""
from pathlib import Path
import importlib.util
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[3]
NH = ROOT / "data/external/nhanes"
OUT = ROOT / "analisi_impatto/04_esperimenti/confine_ai/punto4"
FIGDIR = ROOT / "analytics/soglia_ai/punto4"
SEED = 42
CICLI = {"D": 2005, "E": 2007, "F": 2009, "G": 2011, "H": 2013, "I": 2015, "J": 2017}

spec = importlib.util.spec_from_file_location(
    "passo0", ROOT / "analisi_impatto/codice/non_supervisionato/00_passo0_linea_guida.py")
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)

C_BLU = "#2f6db5"
C_ARA = "#c46b3c"
C_GRIGIO = "#8a8a8a"
C_SCURO = "#3a3a3a"


# ---------------------------------------------------------------- dati
def read(name, s, cols):
    d = pd.read_sas(NH / f"{name}_{s}.xpt")
    return d[[c for c in ["SEQN"] + cols if c in d.columns]]


def mort(y):
    f = NH / f"NHANES_{y}_{y + 1}_MORT_2019_PUBLIC.dat"
    d = pd.read_fwf(f, colspecs=[(0, 6), (14, 15), (15, 16), (16, 19), (45, 48)],
                    names=["SEQN", "eligstat", "mortstat", "ucod", "permth"], dtype=str)
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def egfr_2021(scr, age, female):
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    r = scr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.200 * 0.9938 ** age * np.where(female, 1.012, 1.0)


def load():
    parts = []
    for s, y in CICLI.items():
        d = read("DEMO", s, ["RIAGENDR", "RIDAGEYR", "RIDEXPRG", "WTMEC2YR"])
        for name, cols in [("ALB_CR", ["URXUMA", "URXUCR", "URDACT"]), ("BIOPRO", ["LBXSCR", "LBXSUA"]),
                           ("CBC", ["LBXHGB"]), ("BMX", ["BMXBMI", "BMXWAIST"]),
                           ("BPX", [f"BPXSY{i}" for i in range(1, 5)] + [f"BPXDI{i}" for i in range(1, 5)]),
                           ("GHB", ["LBXGH"]), ("GLU", ["LBXGLU"]), ("TRIGLY", ["LBXTR"]), ("HDL", ["LBDHDD"]),
                           ("DIQ", ["DIQ010"]), ("BPQ", ["BPQ020"])]:
            d = d.merge(read(name, s, cols), on="SEQN", how="left")
        d = d.merge(mort(y), on="SEQN", how="left")
        d["ciclo"] = y
        if s == "D":
            d["LBXSCR"] = -0.016 + 0.978 * d["LBXSCR"]      # calibrazione CDC 2005–2006
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    # 2005–2008: URDACT assente nel file; stessa formula del CDC (URXUMA / URXUCR × 100)
    d["URDACT"] = d["URDACT"].fillna(d["URXUMA"] / d["URXUCR"] * 100)
    sy = d[[f"BPXSY{i}" for i in range(1, 5)]].where(lambda x: x > 0)
    di = d[[f"BPXDI{i}" for i in range(1, 5)]].where(lambda x: x > 0)
    d["SBP"], d["DBP"] = sy.mean(1), di.mean(1)
    d["female"] = d["RIAGENDR"].eq(2)
    d["eGFR"] = egfr_2021(d["LBXSCR"].to_numpy(), d["RIDAGEYR"].to_numpy(), d["female"].to_numpy())
    d["diab_noto"] = d["DIQ010"].eq(1).astype(float)
    d["iper_nota"] = d["BPQ020"].eq(1).astype(float)
    n_tot = len(d)
    d = d[(d["RIDAGEYR"] >= 20) & ~d["RIDEXPRG"].eq(1) & d["eligstat"].eq(1)
          & (d["URDACT"] > 0) & (d["URXUMA"] > 0) & (d["URXUCR"] > 0) & d["eGFR"].notna()
          & d["permth"].notna() & d["mortstat"].notna()].copy()
    d["lacr"] = np.log(d["URDACT"])
    d["lalb"] = np.log(d["URXUMA"])
    d["lucr"] = np.log(d["URXUCR"])
    d["t"] = d["permth"].clip(lower=0.5)
    d["morte"] = d["mortstat"].eq(1).astype(int)
    d["morte_cv"] = (d["mortstat"].eq(1) & d["ucod"].isin([1, 5])).astype(int)
    d["y"] = ((d["URDACT"] >= 30) | (d["eGFR"] < 60)).astype(int)
    d["acr_std"] = d["URDACT"] * d["URXUCR"] / d["URXUCR"].median()
    d["y_std"] = ((d["acr_std"] >= 30) | (d["eGFR"] < 60)).astype(int)
    return d.reset_index(drop=True), n_tot


# ---------------------------------------------------------------- Cox
def cox(X, t, e, w=None, maxit=60):
    X = np.asarray(X, float)
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    o = np.argsort(-t, kind="stable")
    X, t, e, w = X[o], t[o], e[o], w[o]
    mu = X.mean(0)
    Xc = X - mu
    ends = np.r_[np.flatnonzero(np.diff(t) != 0), n - 1]
    gid = np.repeat(ends, np.diff(np.r_[-1, ends]))
    ev = e == 1
    b = np.zeros(p)

    def parts(b):
        eta = Xc @ b
        r = w * np.exp(eta - eta.max())
        S0 = np.cumsum(r)[gid]
        S1 = np.cumsum(r[:, None] * Xc, 0)[gid]
        S2 = np.cumsum(r[:, None, None] * Xc[:, :, None] * Xc[:, None, :], 0)[gid]
        ll = np.sum(w[ev] * (eta[ev] - eta.max() - np.log(S0[ev])))
        m = S1[ev] / S0[ev, None]
        g = np.sum(w[ev, None] * (Xc[ev] - m), 0)
        H = -np.sum(w[ev, None, None] * (S2[ev] / S0[ev, None, None] - m[:, :, None] * m[:, None, :]), 0)
        return ll, g, H

    ll, g, H = parts(b)
    for _ in range(maxit):
        step = np.linalg.solve(H - 1e-9 * np.eye(p), g)
        s = 1.0
        while True:
            nb = b - s * step
            nll, ng, nH = parts(nb)
            if nll >= ll - 1e-10 or s < 1e-4:
                break
            s /= 2
        conv = abs(nll - ll) < 1e-8
        b, ll, g, H = nb, nll, ng, nH
        if conv:
            break
    cov = np.linalg.inv(-H)
    return {"b": b, "se": np.sqrt(np.diag(cov)), "cov": cov, "ll": ll, "aic": -2 * ll + 2 * p}


def rcs(x, knots):
    k = knots
    K = len(k)
    cols = [x]
    for j in range(K - 2):
        c = (np.maximum(x - k[j], 0) ** 3
             - np.maximum(x - k[K - 2], 0) ** 3 * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
             + np.maximum(x - k[K - 1], 0) ** 3 * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2]))
        cols.append(c / (k[K - 1] - k[0]) ** 2)
    return np.column_stack(cols)


def base(d):
    return np.column_stack([d["RIDAGEYR"], d["female"].astype(float), d["eGFR"] / 10])


# ---------------------------------------------------------------- 4a
def analisi_4a(d):
    rows, curve = [], {}
    knots = np.percentile(d["lacr"], [5, 35, 65, 95])
    for esito in ["morte", "morte_cv"]:
        t, e = d["t"].to_numpy(), d[esito].to_numpy()
        B0 = base(d)
        x = d["lacr"].to_numpy()
        m_lin = cox(np.column_stack([B0, x]), t, e)
        S = rcs(x, knots)
        m_spl = cox(np.column_stack([B0, S]), t, e)
        m_step = cox(np.column_stack([B0, x, (d["URDACT"] >= 30).astype(float)]), t, e)
        grid = np.linspace(*np.percentile(x, [5, 95]), 41)
        best = None
        for c in grid:
            m = cox(np.column_stack([B0, x, np.maximum(x - c, 0)]), t, e)
            if best is None or m["ll"] > best[1]["ll"]:
                best = (c, m)
        aic_h = -2 * best[1]["ll"] + 2 * (B0.shape[1] + 3)
        w = d["WTMEC2YR"].to_numpy()
        mw_lin = cox(np.column_stack([B0, x]), t, e, w)
        mw_step = cox(np.column_stack([B0, x, (d["URDACT"] >= 30).astype(float)]), t, e, w)
        bl, sl = m_lin["b"][-1], m_lin["se"][-1]
        bs, ss = m_step["b"][-1], m_step["se"][-1]
        rows.append({
            "esito": esito, "n": len(d), "eventi": int(e.sum()),
            "HR_per_raddoppio_ACR": np.exp(bl * np.log(2)),
            "ic_basso": np.exp((bl - 1.96 * sl) * np.log(2)), "ic_alto": np.exp((bl + 1.96 * sl) * np.log(2)),
            "aic_lineare": m_lin["aic"], "aic_spline": m_spl["aic"], "aic_soglia30": m_step["aic"],
            "aic_hockey": aic_h, "piega_hockey_acr": float(np.exp(best[0])),
            "delta_aic_soglia30_vs_lineare": m_lin["aic"] - m_step["aic"],
            "HR_scalino_30_oltre_lineare": np.exp(bs), "scalino_ic_basso": np.exp(bs - 1.96 * ss),
            "scalino_ic_alto": np.exp(bs + 1.96 * ss),
            # con i pesi campionari l'AIC non è interpretabile: si riporta solo l'HR dello scalino
            "scalino_30_pesato_HR": np.exp(mw_step["b"][-1]),
            "scalino_30_pesato_ic_basso": np.exp(mw_step["b"][-1] - 1.96 * mw_step["se"][-1]),
            "scalino_30_pesato_ic_alto": np.exp(mw_step["b"][-1] + 1.96 * mw_step["se"][-1]),
        })
        # curva HR dalla spline, riferimento ACR = 5
        xs = np.log(np.geomspace(np.percentile(d["URDACT"], 1), np.percentile(d["URDACT"], 99), 200))
        Sx, S5 = rcs(xs, knots), rcs(np.array([np.log(5.0)]), knots)
        D = Sx - S5
        k = S.shape[1]
        bb, V = m_spl["b"][-k:], m_spl["cov"][-k:, -k:]
        lh = D @ bb
        se = np.sqrt(np.einsum("ij,jk,ik->i", D, V, D))
        curve[esito] = (np.exp(xs), np.exp(lh), np.exp(lh - 1.96 * se), np.exp(lh + 1.96 * se))
    return pd.DataFrame(rows), curve


# ---------------------------------------------------------------- 4b
def analisi_4b(d):
    rows = []
    for sesso, m in [("uomini", ~d["female"]), ("donne", d["female"])]:
        g = d[m]
        t, e = g["t"].to_numpy(), g["morte"].to_numpy()
        B0 = np.column_stack([g["RIDAGEYR"], g["eGFR"] / 10])
        m0 = cox(B0, t, e)
        ma = cox(np.column_stack([B0, g["lacr"]]), t, e)
        mb = cox(np.column_stack([B0, g["lalb"]]), t, e)
        mc = cox(np.column_stack([B0, g["lalb"], g["lucr"]]), t, e)
        ba, bu = mc["b"][-2], mc["b"][-1]
        rows.append({
            "sesso": sesso, "n": len(g), "morti": int(e.sum()),
            "creatinina_urinaria_mediana_mg_dl": float(g["URXUCR"].median()),
            "aic_base": m0["aic"], "aic_ACR": ma["aic"], "aic_albumina_sola": mb["aic"],
            "aic_albumina_e_creatinina": mc["aic"],
            "guadagno_aic_ACR": m0["aic"] - ma["aic"], "guadagno_aic_albumina": m0["aic"] - mb["aic"],
            "guadagno_aic_separati": m0["aic"] - mc["aic"],
            "albumina_meglio_di_ACR_delta_aic": ma["aic"] - mb["aic"],
            "HR_raddoppio_albumina": np.exp(ba * np.log(2)),
            "HR_raddoppio_albumina_ic": f"{np.exp((ba - 1.96 * mc['se'][-2]) * np.log(2)):.3f}–"
                                        f"{np.exp((ba + 1.96 * mc['se'][-2]) * np.log(2)):.3f}",
            "HR_raddoppio_creatinina_urinaria": np.exp(bu * np.log(2)),
            "HR_raddoppio_creatinina_urinaria_ic": f"{np.exp((bu - 1.96 * mc['se'][-1]) * np.log(2)):.3f}–"
                                                   f"{np.exp((bu + 1.96 * mc['se'][-1]) * np.log(2)):.3f}",
            "b_alb": ba, "se_alb": mc["se"][-2], "b_ucr": bu, "se_ucr": mc["se"][-1],
            "rapporto_peso_denominatore": -bu / ba,
            "prev_acr30": float((g["URDACT"] >= 30).mean()), "prev_acr_std30": float((g["acr_std"] >= 30).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4c
FEAT = ["RIDAGEYR", "female", "BMXBMI", "BMXWAIST", "SBP", "DBP", "LBXGH", "LBXGLU", "LBXTR", "LBDHDD",
        "LBXSUA", "LBXHGB", "diab_noto", "iper_nota"]


def analisi_4c(d):
    X = d[FEAT].astype(float).to_numpy()
    y = d["y"].to_numpy()
    score = np.zeros(len(d))
    fold = np.zeros(len(d), int)
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, n_jobs=-1, random_state=SEED)
        rf.fit(X[tr], y[tr])
        score[te] = rf.predict_proba(X[te])[:, 1]
        fold[te] = k
    rule = (d["diab_noto"].eq(1) | d["iper_nota"].eq(1)).to_numpy()
    tiebreak = np.random.default_rng(SEED).permutation(len(d))
    sel = p0.select_at_parity(score, fold, rule, tiebreak)
    rows = []
    for verita, yy in [("KDIGO (ACR ≥ 30 o eGFR < 60)", y), ("corretta per i muscoli", d["y_std"].to_numpy())]:
        c = p0.counts(sel, rule, yy)
        lo, hi = p0.boot_diff(sel, rule, yy, np.random.default_rng(SEED))
        rows.append({"verita": verita, "n": len(d), "testati": c["testati"], "positivi_totali": int(yy.sum()),
                     "pos_AI": c["pos_modello"], "pos_regola": c["pos_regola"],
                     "diff": c["pos_modello"] - c["pos_regola"], "ic_basso": lo, "ic_alto": hi})
    # mortalità: positivi trovati solo dall'AI contro solo dalla regola
    pos = y == 1
    solo_ai, solo_reg = sel & ~rule & pos, ~sel & rule & pos
    g = d[solo_ai | solo_reg]
    m = cox(np.column_stack([g["RIDAGEYR"], g["female"].astype(float), solo_ai[solo_ai | solo_reg].astype(float)]),
            g["t"].to_numpy(), g["morte"].to_numpy())
    mort_row = {"solo_AI_n": int(solo_ai.sum()), "solo_AI_morti": int(d.loc[solo_ai, "morte"].sum()),
                "solo_regola_n": int(solo_reg.sum()), "solo_regola_morti": int(d.loc[solo_reg, "morte"].sum()),
                "HR_soloAI_vs_soloRegola_agg_eta_sesso": np.exp(m["b"][-1]),
                "ic_basso": np.exp(m["b"][-1] - 1.96 * m["se"][-1]),
                "ic_alto": np.exp(m["b"][-1] + 1.96 * m["se"][-1]),
                "eta_mediana_solo_AI": float(d.loc[solo_ai, "RIDAGEYR"].median()),
                "eta_mediana_solo_regola": float(d.loc[solo_reg, "RIDAGEYR"].median())}
    return pd.DataFrame(rows), pd.DataFrame([mort_row])


# ---------------------------------------------------------------- figure
def spine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure(r4a, curve, r4b, r4c):
    # Fig 1: HR lungo l'ACR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, esito, tit in zip(axes, ["morte", "morte_cv"], ["Mortalità totale", "Mortalità cardiovascolare"]):
        x, hr, lo, hi = curve[esito]
        r = r4a.set_index("esito").loc[esito]
        ax.fill_between(x, lo, hi, color=C_BLU, alpha=0.15, lw=0)
        ax.plot(x, hr, color=C_BLU, lw=2)
        ax.axvline(30, color=C_SCURO, ls="--", lw=1.5)
        ax.axhline(1, color="#bbbbbb", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{tit} ({int(r.eventi)} morti)", fontsize=10)
        ax.text(31, hr.max() * 0.9, "KDIGO 30", fontsize=9, color=C_SCURO)
        ax.text(x[3], hr.max() * 0.62,
                f"scalino a 30: ΔAIC {r.delta_aic_soglia30_vs_lineare:+.1f}\n"
                f"HR extra {r.HR_scalino_30_oltre_lineare:.2f} ({r.scalino_ic_basso:.2f}–{r.scalino_ic_alto:.2f})",
                fontsize=9, color=C_BLU)
        ax.set_xlabel("ACR (mg/g), scala log")
        ax.grid(axis="y", color="#eeeeee", lw=0.8)
        spine(ax)
    axes[0].set_ylabel("Hazard ratio rispetto ad ACR 5 (età, sesso, eGFR)")
    fig.suptitle("NHANES 2005–2018: il rischio di morte sale in modo continuo con l'ACR?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig1_mortalita_lungo_acr.png", dpi=200)
    plt.close(fig)

    # Fig 2: albumina e creatinina urinaria per sesso
    fig, ax = plt.subplots(figsize=(8, 4.2))
    yy = np.arange(len(r4b))
    for i, (_, r) in enumerate(r4b.iterrows()):
        for off, b, se, col, lab in [(0.15, r.b_alb, r.se_alb, C_BLU, "albumina urinaria (numeratore)"),
                                     (-0.15, r.b_ucr, r.se_ucr, C_ARA, "creatinina urinaria (denominatore)")]:
            k = np.log(2)
            ax.plot([np.exp((b - 1.96 * se) * k), np.exp((b + 1.96 * se) * k)], [i + off] * 2, color=col, lw=2)
            ax.plot(np.exp(b * k), i + off, "o", color=col, ms=8, mec="white", mew=1.5,
                    label=lab if i == 0 else None)
        ax.text(1.32, i, f"peso del denominatore\nrispetto all'ACR: {r.rapporto_peso_denominatore:.2f}\n"
                         f"(ACR assume 1,00)", va="center", fontsize=9)
    ax.axvline(1, color="#bbbbbb", lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{s} ({n} persone, {m} morti)" for s, n, m in zip(r4b.sesso, r4b.n, r4b.morti)], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 1.75)
    ax.set_xlabel("Hazard ratio di morte per ogni raddoppio (età, eGFR)")
    ax.set_title("L'ACR divide l'albumina per la creatinina: il corpo è d'accordo?", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.4, -0.18), ncol=2)
    spine(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_albumina_creatinina_sesso.png", dpi=200)
    plt.close(fig)

    # Fig 3: replica del +22
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(r4c))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, r4c.pos_regola, w, color=C_GRIGIO, label="regola: diabete o ipertensione noti")
    ax.bar(x + w / 2 + 0.01, r4c.pos_AI, w, color=C_BLU, label="AI (Random Forest, stesse persone testate)")
    top = r4c[["pos_AI", "pos_regola"]].to_numpy().max()
    for i, r in r4c.iterrows():
        ax.text(i - w / 2, r.pos_regola + top * 0.01, f"{r.pos_regola}", ha="center", fontsize=9)
        ax.text(i + w / 2, r.pos_AI + top * 0.01, f"{r.pos_AI}", ha="center", fontsize=9)
        ax.text(i, max(r.pos_AI, r.pos_regola) + top * 0.08, f"{r['diff']:+d} (IC {r.ic_basso:+.0f}/{r.ic_alto:+.0f})",
                ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["Verità KDIGO", "Verità corretta per i muscoli"], fontsize=9)
    ax.set_ylabel(f"Malati trovati fra {int(r4c.testati[0])} persone testate")
    ax.set_ylim(0, top * 1.25)
    ax.set_title("NHANES: l'AI trova più malati della regola a parità di test?", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    spine(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_replica_piu22.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    d, n_tot = load()
    print("NHANES righe totali", n_tot, "| analizzati", len(d), "| morti", int(d.morte.sum()),
          "| morti CV", int(d.morte_cv.sum()), "| follow-up mediano mesi", float(d.permth.median()))
    r4a, curve = analisi_4a(d)
    r4b = analisi_4b(d)
    r4c, r4c_m = analisi_4c(d)
    r4a.to_csv(OUT / "P4a_scalino_30.csv", index=False)
    r4b.to_csv(OUT / "P4b_sesso_albumina_creatinina.csv", index=False)
    r4c.to_csv(OUT / "P4c_replica_piu22.csv", index=False)
    r4c_m.to_csv(OUT / "P4c_mortalita_trovati.csv", index=False)
    pd.DataFrame([{"n_nhanes_totale": n_tot, "n_analisi": len(d), "morti": int(d.morte.sum()),
                   "morti_cv": int(d.morte_cv.sum()), "followup_mediano_mesi": float(d.permth.median()),
                   "prevalenza_y": float(d.y.mean())}]).to_csv(OUT / "P4_campione.csv", index=False)
    figure(r4a, curve, r4b, r4c)
    for f in OUT.glob("fig*.png"):
        shutil.copy(f, FIGDIR / f.name)
    pd.set_option("display.width", 250)
    print(r4a.round(3).T.to_string())
    print(r4b.round(3).T.to_string())
    print(r4c.to_string(index=False))
    print(r4c_m.round(3).T.to_string())


if __name__ == "__main__":
    main()
