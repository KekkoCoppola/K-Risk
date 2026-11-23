"""Punto 3: reni o muscoli? Uomini e donne.

Previsioni PREDITTIVE (OOF Fase A RF, 74 colonne senza esami renali: il modello del +22),
lette contro bersagli renali scomposti (ESPLICATIVO). Solo training.
Piano fissato prima: analisi_impatto/04_esperimenti/confine_ai/piano_punto3.md
"""
from pathlib import Path
import importlib.util
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data import kidney  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "passo0", ROOT / "analisi_impatto/codice/non_supervisionato/00_passo0_linea_guida.py")
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)

OUT = ROOT / "analisi_impatto/04_esperimenti/confine_ai/punto3"
FIGDIR = ROOT / "analytics/soglia_ai/punto3"
SEED = 42
B = 1000
MODELLI = {"Random Forest (Fase A)": "phase_a/main/random_forest", "TabPFN (Fase D)": "phase_d/tabpfn"}
C_RENI = "#2f6db5"
C_MUSCOLI = "#c46b3c"
C_REGOLA = "#8a8a8a"
C_AI = "#2f6db5"


def quota(s, la, lu, lacr):
    ca = np.cov(s, lacr)[0, 1]
    return np.cov(s, la)[0, 1] / ca, -np.cov(s, lu)[0, 1] / ca


def boot_quota(s, la, lu, lacr, rng, mask=None):
    idx = np.flatnonzero(mask) if mask is not None else np.arange(len(s))
    q = []
    for _ in range(B):
        i = rng.choice(idx, len(idx))
        q.append(quota(s[i], la[i], lu[i], lacr[i])[1])
    return np.array(q)


def spine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    n = len(df)
    y = kidney.target(df).to_numpy()
    egfr = kidney.egfr(df).to_numpy()
    fem = kidney.is_female(df).to_numpy().astype(bool)
    acr = df["UMAUCR"].to_numpy(float)
    alb = df["UmALB"].to_numpy(float)
    ucre = df["UCRE"].to_numpy(float)
    ok = (acr > 0) & (alb > 0) & (ucre > 0)
    lacr, la, lu = np.log(np.where(ok, acr, np.nan)), np.log(np.where(ok, alb, np.nan)), np.log(np.where(ok, ucre, np.nan))
    acr_std = acr * ucre / np.nanmedian(ucre)
    y_std = ((acr_std >= 30) | (egfr < 60)).astype(int)
    rng = np.random.default_rng(SEED)

    # ---- 3.1 quota numeratore/denominatore ----
    rows, boots = [], {}
    gruppi = {"tutti": ok, "uomini": ok & ~fem, "donne": ok & fem, "eGFR ≥ 60": ok & (egfr >= 60)}
    scores = {}
    for nome, mid in MODELLI.items():
        sc, fold = p0.load_scores(mid, n)
        scores[nome] = (sc, fold)
        lg = np.log(np.clip(sc, 1e-6, 1 - 1e-6) / (1 - np.clip(sc, 1e-6, 1 - 1e-6)))
        for g, m in gruppi.items():
            if nome != "Random Forest (Fase A)" and g != "tutti":
                continue
            qn, qd = quota(lg[m], la[m], lu[m], lacr[m])
            bq = boot_quota(lg, la, lu, lacr, rng, m)
            boots[(nome, g)] = bq
            rows.append({"modello": nome, "gruppo": g, "n": int(m.sum()), "quota_numeratore": qn,
                         "quota_denominatore": qd, "ic_basso": np.percentile(bq, 2.5),
                         "ic_alto": np.percentile(bq, 97.5)})
    # differenza donne − uomini (bootstrap indipendenti nei due gruppi)
    dd = boots[("Random Forest (Fase A)", "donne")] - boots[("Random Forest (Fase A)", "uomini")]
    q31 = pd.DataFrame(rows)
    q31.to_csv(OUT / "P3_quota_reni_muscoli.csv", index=False)
    diff_sesso = {"diff_donne_meno_uomini": float(np.median(dd)), "ic_basso": float(np.percentile(dd, 2.5)),
                  "ic_alto": float(np.percentile(dd, 97.5))}

    # ---- 3.2 il +22 con verità standardizzata ----
    rule = (df["DM"].eq(1) | df["HypertenHis"].fillna(0).eq(1)).to_numpy()
    tiebreak = np.random.default_rng(SEED).permutation(n)
    sc, fold = scores["Random Forest (Fase A)"]
    sel = p0.select_at_parity(sc, fold, rule, tiebreak)
    div = (acr >= 30) & (acr_std < 30) & (egfr >= 60)        # positivi da divisione
    pos = y == 1
    gruppi_pos = {"solo AI": sel & ~rule & pos, "solo regola": ~sel & rule & pos, "entrambi": sel & rule & pos}
    r32 = []
    for g, m in gruppi_pos.items():
        r32.append({"gruppo_positivi": g, "n": int(m.sum()), "da_divisione": int((m & div).sum()),
                    "quota_da_divisione": float((m & div).sum() / max(m.sum(), 1)),
                    "ucre_mediana": float(np.nanmedian(ucre[m])), "umalb_mediana": float(np.nanmedian(alb[m])),
                    "quota_donne": float(fem[m].mean())})
    a, b = gruppi_pos["solo AI"], gruppi_pos["solo regola"]
    fisher = stats.fisher_exact([[int((a & div).sum()), int((a & ~div).sum())],
                                 [int((b & div).sum()), int((b & ~div).sum())]])
    conteggi = []
    for verita, yy in [("ACR ≥ 30 o eGFR < 60 (KDIGO)", y), ("ACR standardizzato ≥ 30 o eGFR < 60", y_std)]:
        c = p0.counts(sel, rule, yy)
        lo, hi = p0.boot_diff(sel, rule, yy, np.random.default_rng(SEED))
        conteggi.append({"verita": verita, "testati": c["testati"], "positivi_totali": int(yy.sum()),
                         "pos_AI": c["pos_modello"], "pos_regola": c["pos_regola"],
                         "diff": c["pos_modello"] - c["pos_regola"], "ic_basso": lo, "ic_alto": hi})
    pd.DataFrame(r32).assign(fisher_p=fisher.pvalue).to_csv(OUT / "P3_piu22_divisione.csv", index=False)
    cont = pd.DataFrame(conteggi)
    cont.to_csv(OUT / "P3_piu22_verita.csv", index=False)

    # ---- 3.3 sesso ----
    r33 = []
    for sesso, m in [("uomini", ~fem), ("donne", fem)]:
        r33.append({"sesso": sesso, "n": int(m.sum()), "ucre_mediana": float(np.nanmedian(ucre[m])),
                    "prev_acr30": float((acr[m] >= 30).mean()), "n_acr30": int((acr[m] >= 30).sum()),
                    "prev_acr_std30": float((acr_std[m] >= 30).mean()), "n_acr_std30": int((acr_std[m] >= 30).sum()),
                    "n_da_divisione": int((div & m).sum()),
                    "quota_scelti_AI": float((sel & m).sum() / sel.sum()),
                    "quota_scelti_regola": float((rule & m).sum() / rule.sum()),
                    "quota_popolazione": float(m.mean())})
    s33 = pd.DataFrame(r33)
    s33.to_csv(OUT / "P3_sesso.csv", index=False)

    # ---- decisioni preregistrate ----
    rf = q31[(q31.modello == "Random Forest (Fase A)") & (q31.gruppo == "tutti")].iloc[0]
    std = cont.iloc[1]
    esito = pd.DataFrame([{
        "ai_vede_in_parte_i_muscoli": bool(rf["ic_basso"] > 0.10),
        "quota_denominatore_RF": rf["quota_denominatore"],
        "ic_RF": f"{rf['ic_basso']:.3f}–{rf['ic_alto']:.3f}",
        "piu22_regge_controllo_muscolare": bool(std["ic_basso"] > 0),
        "diff_con_verita_standardizzata": std["diff"],
        "ic_diff_std": f"{std['ic_basso']:.0f}–{std['ic_alto']:.0f}",
        "fisher_p_divisione_soloAI_vs_soloRegola": fisher.pvalue,
        "differenza_sessi": bool(diff_sesso["ic_basso"] > 0 or diff_sesso["ic_alto"] < 0),
        **{f"sesso_{k}": v for k, v in diff_sesso.items()},
    }])
    esito.to_csv(OUT / "P3_esito.csv", index=False)

    # ---- figure ----
    # Fig 1: quota reni / muscoli
    q = q31.copy()
    q["etichetta"] = np.where(q.modello.str.startswith("Random"), "RF · " + q.gruppo, "TabPFN · tutti")
    q = q.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    yy = np.arange(len(q))
    ax.barh(yy, q.quota_numeratore, color=C_RENI, height=0.55, label="dall'albumina urinaria (reni)")
    ax.barh(yy, q.quota_denominatore, left=q.quota_numeratore, color=C_MUSCOLI, height=0.55,
            label="dalla creatinina urinaria (muscoli)")
    for i, (_, r) in enumerate(q.iterrows()):
        ax.text(1.01, i, f"muscoli {r.quota_denominatore:.0%} (IC {r.ic_basso:.0%}–{r.ic_alto:.0%})",
                va="center", fontsize=9)
    ax.set_yticks(yy)
    ax.set_yticklabels(q.etichetta, fontsize=9)
    ax.set_xlim(0, 1.55)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Quota di ciò che il punteggio dell'AI coglie dell'ACR")
    ax.set_title("Cosa vede l'AI addestrata senza esami renali: reni o muscoli?", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.35, -0.18), ncol=2)
    spine(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_reni_o_muscoli.png", dpi=200)
    plt.close(fig)

    # Fig 2: il +22 con le due verità
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(cont))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, cont.pos_regola, w, color=C_REGOLA, label="regola: diabete o ipertensione")
    ax.bar(x + w / 2 + 0.01, cont.pos_AI, w, color=C_AI, label="AI (Random Forest, stesse persone testate)")
    for i, r in cont.iterrows():
        ax.text(i - w / 2, r.pos_regola + 2, f"{r.pos_regola}", ha="center", fontsize=9)
        ax.text(i + w / 2, r.pos_AI + 2, f"{r.pos_AI}", ha="center", fontsize=9)
        ax.text(i, max(r.pos_AI, r.pos_regola) + 14, f"{r['diff']:+d} (IC {r.ic_basso:+.0f}/{r.ic_alto:+.0f})",
                ha="center", fontsize=10, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(["Verità KDIGO\n(ACR ≥ 30)", "Verità corretta per i muscoli\n(ACR standardizzato ≥ 30)"],
                       fontsize=9)
    ax.set_ylabel(f"Malati trovati fra {int(cont.testati[0])} persone testate")
    ax.set_ylim(0, cont[["pos_AI", "pos_regola"]].to_numpy().max() * 1.3)
    ax.set_title("Il vantaggio dell'AI regge se si tolgono i \"malati per divisione\"?", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    spine(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_piu22_muscoli.png", dpi=200)
    plt.close(fig)

    # Fig 3: sesso
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(2)
    ax.bar(x - w / 2 - 0.01, s33.prev_acr30 * 100, w, color=C_MUSCOLI, label="ACR ≥ 30 (regola KDIGO)")
    ax.bar(x + w / 2 + 0.01, s33.prev_acr_std30 * 100, w, color=C_RENI, label="ACR standardizzato ≥ 30")
    for i, r in s33.iterrows():
        ax.text(i - w / 2, r.prev_acr30 * 100 + 0.2, f"{r.prev_acr30:.1%}\n({r.n_acr30})", ha="center", fontsize=9)
        ax.text(i + w / 2, r.prev_acr_std30 * 100 + 0.2, f"{r.prev_acr_std30:.1%}\n({r.n_acr_std30})",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Uomini (n={s33.n[0]})", f"Donne (n={s33.n[1]})"])
    ax.set_ylabel("Persone con albuminuria (%)")
    ax.set_ylim(0, max(s33.prev_acr30.max(), s33.prev_acr_std30.max()) * 100 * 1.35)
    ax.set_title("Quanta albuminuria dipende dalla divisione per la creatinina?", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    spine(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sesso.png", dpi=200)
    plt.close(fig)

    for f in OUT.glob("fig*.png"):
        shutil.copy(f, FIGDIR / f.name)

    print(q31.round(3).to_string(index=False))
    print("diff donne-uomini", {k: round(v, 3) for k, v in diff_sesso.items()})
    print(pd.DataFrame(r32).round(3).to_string(index=False), "\nFisher p", round(fisher.pvalue, 4))
    print(cont.to_string(index=False))
    print(s33.round(3).to_string(index=False))
    print(esito.T.to_string())


if __name__ == "__main__":
    main()
