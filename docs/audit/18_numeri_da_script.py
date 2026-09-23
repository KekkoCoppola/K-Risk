# Audit del 22-23/09/2026 - ricalcola i numeri del documento che NON si ottengono con un filtro su un solo CSV
# (conteggi di soglie, differenze fra file diversi, t corretto, valori x100, segni invertiti, training).
# Esecuzione dalla radice del repository: python docs/audit/18_numeri_da_script.py. Legge tabelle e
# record di analytics/ e data/processed/train.csv (mai test.csv, mai data/raw). Ogni riga confronta
# il valore ricalcolato con la stringa del documento.
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
from src.data.kidney import egfr, target  # noqa: E402
from src.models import evaluation as ev  # noqa: E402
from src.models.clinical_utility import avoided_per_100, net_benefit, net_benefit_all, threshold_grid  # noqa: E402

doc = open("valorizzazione_tesi.md", encoding="utf-8").read()
errors = 0


def check(tid, text, value, dec, pct=False):
    """text = stringa come nel documento; value = valore ricalcolato."""
    global errors
    shown = float(text.replace("−", "-").replace("+", "").replace("%", "").replace(".", "").replace(",", "."))
    got = round(value * (100 if pct else 1), dec)
    ok = abs(got - shown) < 1e-9 and text in doc
    errors += not ok
    print(f"{'ok ' if ok else 'ERR'} {tid:7s} {text:>9s}  ricalcolato {value:.6g}")


A = "analytics/"
qdc = pd.read_csv(A + "quality/evaluation/decision_curve.csv")
tdc = pd.read_csv(A + "test/run_1/decision_curve.csv")


def wins(curve, model):
    c = curve[(curve.group == "tutti") & (curve.threshold >= 0.05 - 1e-9)]
    w = c.pivot(index="threshold", columns="model", values="net_benefit")
    return int((w[model] > w["lr_scored"]).sum()), len(w)


# T5/T41: soglie 5-20% in cui il net benefit supera SCORED (out-of-fold) e T6/T43 (test)
for tid, curve, model, text in [("T41a", qdc, "lr_penalized", "31"), ("T41b", qdc, "random_forest", "31"),
                                ("T41c", qdc, "xgboost", "26"), ("T43a", tdc, "lr_penalized", "10"),
                                ("T43b", tdc, "random_forest", "28"), ("T43c", tdc, "xgboost", "18")]:
    n_win, n_all = wins(curve, model)
    check(tid, text, n_win, 0)
    check(tid + "*", "31", n_all, 0)

# T64: soglia da cui ogni modello batte entrambe le strategie fino al 20%
c = qdc[(qdc.group == "tutti") & (qdc.model != "dummy")]
first = {}
for m, g in c.groupby("model"):
    g = g.sort_values("threshold")
    ok, th = g.beats_both.to_numpy(), g.threshold.to_numpy()
    first[m] = min(th[i] for i in range(len(th)) if ok[i:].all())
print("   T64 prima soglia da cui ogni modello batte entrambe le strategie fino al 20%:", first)
check("T64a", "5", max(first[m] for m in ["lr_penalized", "random_forest", "xgboost"]) * 100, 0)  # i tre con esami del sangue
check("T64b", "6%", max(first.values()), 0, pct=True)    # anche SCORED dal 6%

# T41/T45: persone da esaminare ogni 100 (alert_rate x 100) e differenze con SCORED
op = pd.read_csv(A + "phase_a/evaluation/q1_operating_points.csv")
op = op[op.feature_set == "main"].set_index(["model", "target_sensitivity"]).alert_rate
for s, texts in [(0.85, ["73,8", "69,2", "68,5", "66,2"]), (0.9, ["80,2", "78,9", "78,1", "76,9"])]:
    for m, text in zip(["lr_scored", "lr_penalized", "random_forest", "xgboost"], texts):
        check(f"T41-{s}", text, op[(m, s)] * 100, 1)
d85 = [(op[("lr_scored", 0.85)] - op[(m, 0.85)]) * 100 for m in ["lr_penalized", "random_forest", "xgboost"]]
check("T45a", "4,6", min(d85), 1)
check("T45b", "7,6", max(d85), 1)
check("T45c", "460", round(min(d85), 1) * 100, 0)
check("T45d", "760", round(max(d85), 1) * 100, 0)
check("T45e", "22.500", round(min(d85), 1) * 100 * 49, -2)
check("T45f", "37.200", round(max(d85), 1) * 100 * 49, -2)

# T41: IC delle differenze con SCORED per RF e XGBoost (riga lr_scored-modello, segno invertito)
cm = pd.read_csv(A + "phase_a/evaluation/comparison_models.csv")
cm = cm[(cm.feature_set == "main") & (cm.metric == "auc") & (cm.model_a == "lr_scored")].set_index("model_b")
check("T41-rf", "−0,008", -cm.loc["random_forest", "high"], 3)
check("T41-rf", "+0,066", -cm.loc["random_forest", "low"], 3)
check("T41-xgb", "−0,004", -cm.loc["xgboost", "high"], 3)
check("T41-xgb", "+0,054", -cm.loc["xgboost", "low"], 3)

# T53, T57, T59: confronti appaiati sui fold con SCORED (t corretto del progetto) e Holm sugli 11 candidati
fd = pd.read_csv(A + "phase_d/folds_auc.csv").sort_values("fold")
fa = pd.read_csv(A + "phase_a/evaluation/folds.csv")
sc = fa[(fa.feature_set == "main") & (fa.model == "lr_scored")].sort_values("fold").auc.to_numpy()
n, k = 4350, 5
n_test, n_train = n / k, n - n / k
cands = ["ensemble_mean", "xgboost_native_nan", "xgboost_extended", "target_decomposition", "catboost", "lightgbm",
         "ebm", "tabpfn", "xgboost_no_pruner", "lr_all_scaled", "tri_ensemble_top21"]
res = {c: ev.corrected_ttest(fd[c].to_numpy() - sc, n_train, n_test) for c in cands + ["tri_ensemble_leaky"]}
holm = dict(zip(cands, ev.holm([res[c]["p_value"] for c in cands])))
for tid, c, texts in [("T53a", "tri_ensemble_top21", ("+0,027", "−0,004", "+0,057", "0,07")),
                      ("T53b", "tri_ensemble_leaky", ("+0,039", "+0,006", "+0,071", "0,03")),
                      ("T57a", "tabpfn", ("+0,038", "+0,014", "+0,061", None)),
                      ("T57b", "ensemble_mean", ("+0,030", "+0,001", "+0,060", None))]:
    r = res[c]
    for text, key in zip(texts, ["difference", "low", "high", "p_value"]):
        if text:
            check(tid, text, r[key], 3 if key != "p_value" else 2)
check("T59a", "0,011", res["tabpfn"]["p_value"], 3)
check("T59b", "0,12", holm["tabpfn"], 2)
print("   Holm minimo fra gli 11 candidati contro SCORED:", round(min(holm.values()), 3))

# T57: differenze fra AUROC complessive (due file diversi)
pdd = pd.read_csv(A + "phase_d/discrimination.csv").set_index("model").auc
paq = pd.read_csv(A + "phase_a/evaluation/q1_discrimination.csv")
sc_auc = paq[(paq.feature_set == "main") & (paq.model == "lr_scored")].auc.iloc[0]
for text, c in [("+0,034", "tabpfn"), ("+0,032", "ensemble_mean"), ("+0,028", "tri_ensemble_top21")]:
    check("T57c", text, pdd[c] - sc_auc, 3)

# T99: differenza minima rilevabile della regola della Fase D
sds = [np.std(fd[c].to_numpy() - fd["random_forest"].to_numpy(), ddof=1) for c in cands]
ds = float(np.median(sds))
factor = np.sqrt(1 / k + n_test / n_train)
check("T99a", "0,0105", ds, 4)
check("T99b", "0,0195", stats.t.ppf(0.975, k - 1) * factor * ds, 4)
check("T99c", "0,0403", stats.t.ppf(1 - 0.05 / 11 / 2, k - 1) * factor * ds, 4)
check("T99d", "0,02", stats.t.ppf(0.975, k - 1) * factor * ds, 2)
check("T99e", "0,04", stats.t.ppf(1 - 0.05 / 11 / 2, k - 1) * factor * ds, 2)
print(f"   T99 t(0,975;4) = {stats.t.ppf(0.975, k - 1):.3f}; t(1-0,05/22;4) = {stats.t.ppf(1 - 0.05 / 22, k - 1):.3f}; "
      f"fattore = {factor:.4f}")

# T75: confronto con Bragg-Gresham (73,2% trovati con il 37,7% esaminati; 85% con "just under half")
scurve = pd.read_csv(A + "quality/evaluation/screening_curves.csv")
bg = scurve[scurve.setting == "bragg_gresham"]
extra73 = [100 * (np.interp(0.732, g.sort_values("screened").detected, g.sort_values("screened").screened) - 0.377)
           for _, g in bg.groupby("model")]
check("T75a", "12", min(extra73), 0)
check("T75b", "22", max(extra73), 0)
ext = pd.read_csv(A + "quality/evaluation/external_comparison.csv")
e85 = ext[(ext.setting == "bragg_gresham") & ext.rule.str.startswith("sensib")].screened
check("T75c", "19", 100 * (e85.min() - 0.5), 0)
check("T75d", "27", 100 * (e85.max() - 0.5), 0)

# T78: (B) casi gravi guadagnati meno persi sul test; T111: stessa cosa out-of-fold (Fase B, tecniche confermative)
tpc = pd.read_csv(A + "test/run_1/primary_comparison.csv")
net_t = tpc.gained - tpc.lost
check("T78a", "−2", net_t.min(), 0)
check("T78b", "+2", net_t.max(), 0)
pbc = pd.read_csv(A + "phase_b/evaluation/primary_comparison.csv")
pbc = pbc[pbc.confirmatory]
net_b = pbc.gained - pbc.lost
check("T111a", "−3", net_b.min(), 0)
check("T111b", "+3", net_b.max(), 0)

# T79: semiampiezza degli IC dell'AUROC sul test (circa 0,045)
tq = pd.read_csv(A + "test/run_1/q1_discrimination.csv")
tq = tq[(tq.technique == "fase_a") & (tq.group == "tutti") & (tq.model != "dummy")]
half = (tq.auc_high - tq.auc_low) / 2
print(f"   T79 semiampiezze IC test: {half.min():.4f}-{half.max():.4f} (documento: ±0,045)")

# T84: curva di apprendimento, dal 60% al 100% del training
lc = pd.read_csv(A + "phase_d/learning_curve.csv").groupby(["model", "fraction"]).auc.mean()
gain = [lc[(m, 1.0)] - lc[(m, 0.6)] for m in ["lr_penalized", "xgboost"]]
check("T84a", "0,006", min(gain), 3)
check("T84b", "0,008", max(gain), 3)

# T25/T28: quota di DM = 1 nel training (calibration.csv, n per gruppo)
qcal = pd.read_csv(A + "quality/evaluation/calibration.csv")
nn = qcal[qcal.model == "lr_penalized"].set_index("group").n
check("T25a", "6%", nn["diabetici"] / nn["tutti"], 0, pct=True)

# T82: 91% dei positivi con ACR >= 30 = somma dei positivi delle 4 fasce di ACR (label_noise_auroc.csv)
lna = pd.read_csv(A + "phase_d/label_noise_auroc.csv")
bands = lna[(lna.model == "lr_penalized") & lna.analysis.str.startswith("fascia ACR")].positives.sum()
check("T82a", "91%", bands / 425, 0, pct=True)

# --- dal training (data/processed/train.csv)
train = pd.read_csv("data/processed/train.csv")
y = target(train).to_numpy()
alb = (train.UMAUCR >= 30).to_numpy()
low = (egfr(train) < 60).to_numpy()
check("T82b", "25", int((alb & low).sum()), 0)
check("T82c", "362", int((alb & ~low).sum()), 0)
check("T97a", "387", int(alb.sum()), 0)
check("T97b", "248", int((100 * train.UmALB / train.UCRE >= 30).sum()), 0)
check("T97c", "17", 30 * 100 / 176.8, 0)
f = train.UMAUCR / (train.UmALB / train.UCRE)
check("T95a", "176,8", f.median(), 1)
check("T95b", "173,7", f.min(), 1)
check("T95c", "178,9", f.max(), 1)
check("T95d", "4.350", int(f.notna().sum()), 0)
check("T96a", "185", train.UCRE.median(), 0)
check("T96b", "2", 185 / 88.4, 0)
# T28: fra chi ha DM = 0, soglie diagnostiche ADA 2024 (Tabella 2.1) su glicemia a digiuno, HbA1c e glicemia a 2 ore
no_dm = train[train.DM == 0]
over = (no_dm.FPG >= 7.0) | (no_dm.HbA1c >= 6.5) | (no_dm.PG2h >= 11.1)
check("T28a", "4.087", len(no_dm), 0)
check("T28b", "75", int(over.sum()), 0)
check("T28c", "1,8%", over.mean(), 1, pct=True)
check("T28d", "5,45", no_dm.FPG.median(), 2)
check("T28e", "5,5%", no_dm.HbA1c.median(), 1)
check("T28f", "369", int(no_dm.HbA1c.isna().sum()), 0)
same = bool(((train.UMAUCR >= 30).astype(int) == train.HighACR).all())
print(f"{'ok ' if same else 'ERR'} T27     HighACR == (UMAUCR >= 30) su tutte le {len(train)} righe: {same}")
errors += not same
days = pd.DataFrame({"day": train.Data, "ucre": train.UCRE, "share": train.UMAUCR >= 30, "umalb": train.UmALB}).groupby("day").agg(
    ucre=("ucre", "median"), share=("share", "mean"), umalb=("umalb", "median"))
check("T87a", "−0,70", stats.spearmanr(days.ucre, days.share)[0], 2)
print(f"   T87 giornate {len(days)}; rho UCRE-UmALB {stats.spearmanr(days.ucre, days.umalb)[0]:.2f}")

# T7/T55/T56: tri-ensemble (record della Fase D) contro SCORED (previsioni out-of-fold della Fase A)
p_tri = np.full(len(y), np.nan)
feats = []
for o in range(5):
    rec = json.load(open(f"{A}phase_d/tri_ensemble_top21/outer{o}.json"))
    p_tri[rec["rows"]] = rec["probability"]
    feats.append(set(rec["features"]))
oof = pd.read_csv(A + "phase_a/oof_predictions.csv")
oof = oof[(oof.feature_set == "main") & (oof.model == "lr_scored")]
p_sc = np.full(len(y), np.nan)
p_sc[oof.row.to_numpy()] = oof.probability.to_numpy()
for s, t_tri, t_sc in [(0.85, "66,0", "73,8"), (0.9, "75,0", "80,2")]:
    a_tri = ev.operating_point(y, p_tri, ev.threshold_at_sensitivity(y, p_tri, s))["alert_rate"] * 100
    a_sc = ev.operating_point(y, p_sc, ev.threshold_at_sensitivity(y, p_sc, s))["alert_rate"] * 100
    check("T55", t_tri, a_tri, 1)
    check("T55", t_sc, a_sc, 1)
    if s == 0.85:
        check("T55", "7,8", a_sc - a_tri, 1)
th = [t for t in threshold_grid() if t >= 0.05 - 1e-9]
check("T55", "31", sum(net_benefit(y, p_tri, t) > net_benefit(y, p_sc, t) for t in th), 0)
prev = y.mean()
gain7 = (avoided_per_100(net_benefit(y, p_tri, 0.07), net_benefit_all(prev, 0.07), 0.07)
         - avoided_per_100(net_benefit(y, p_sc, 0.07), net_benefit_all(prev, 0.07), 0.07))
check("T55", "5,6", gain7, 1)
always = set.intersection(*feats)
four = {v for v in set.union(*feats) if sum(v in fs for fs in feats) == 4}
print(f"   T56 in 5 fold ({len(always)}): {sorted(always)}; in 4 fold ({len(four)}): {sorted(four)}")
check("T56a", "9", len(always), 0)
check("T56b", "6", len(four), 0)
print(f"errori: {errors}")
