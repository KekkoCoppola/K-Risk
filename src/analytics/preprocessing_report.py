"""Figure di controllo sul preprocessing (solo training)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analytics.plotting import BLUE, GRAY, ORANGE, TEXT_SECONDARY, label_bars, save
from src.config import CONFIG, resolve
from src.data.kidney import target
from src.data.preprocess import FEATURE_SETS, select_features
from src.data.split import PROCESSED

SECTION = "preprocessing"
FEATURES = CONFIG["features"]
IMPUTATION = resolve(CONFIG["imputation"]["output"])
LEAKAGE_AUC = 0.75
# colonne renali escluse, mostrate come riferimento accanto alle feature finali
RENAL = ["EUGFR90UACR01", "UMAUCR", "HighACR", "UmALB", "DN", "UCRE", "GFR", "SCRE", "BUN"]
TOP = 12

GROUP_LABELS = {
    "leakage": "leakage del target",
    "renal_exams": "esami renali",
    "admin": "amministrative",
    "undocumented": "non documentate",
    "sparse": "vuote (> 50% NA)",
    "knowledge_survey": "questionario conoscenza",
    "derived": "derivati a soglia",
    "duplicate_measures": "misure duplicate",
    "linear_combinations": "combinazioni lineari",
    "comorbidities": "comorbidità (codice 9)",
    "unstable_indices": "indici instabili",
    "high_missing": "NA > 15%",
}


def univariate_auc(values, y):
    observed = values.notna()
    auc = roc_auc_score(y[observed], values[observed])
    return max(auc, 1 - auc)


def column_map():
    rows = [(GROUP_LABELS[g], len(c), "esclusa") for g, c in FEATURES["exclude"].items()]
    rows += [("numeriche", len(FEATURES["numeric"]), "in input"),
             ("categoriche", len(FEATURES["categorical"]), "in input")]
    table = pd.DataFrame(rows, columns=["gruppo", "colonne", "stato"])
    table = table.sort_values(["stato", "colonne"], ascending=[False, True])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = [BLUE if s == "in input" else GRAY for s in table["stato"]]
    bars = ax.barh(table["gruppo"], table["colonne"], color=colors, height=0.7)
    label_bars(ax, bars, table["colonne"].astype(str), horizontal=True)
    included = table.loc[table["stato"] == "in input", "colonne"].sum()
    ax.set_xlabel("numero di colonne")
    ax.set_title(f"Destinazione delle {table['colonne'].sum()} colonne: "
                 f"{included} in input, {table['colonne'].sum() - included} escluse")
    ax.legend(handles=[Patch(color=BLUE, label="in input"), Patch(color=GRAY, label="esclusa")],
              loc="lower right")
    ax.grid(axis="x")
    save(fig, SECTION, "01_column_map")


def auc_plot(train):
    y = target(train)
    X = select_features(train)
    features = pd.Series({c: univariate_auc(X[c], y) for c in X.columns})
    renal = pd.Series({c: univariate_auc(train[c], y) for c in RENAL})
    table = pd.concat([features.nlargest(TOP), renal]).sort_values()
    colors = [ORANGE if c in RENAL else BLUE for c in table.index]

    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(table) + 1.4))
    ax.hlines(table.index, 0.5, table.values, color=colors, linewidth=2)
    ax.scatter(table.values, table.index, color=colors, s=36, zorder=3)
    ax.axvline(LEAKAGE_AUC, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.annotate(f"soglia di allarme {LEAKAGE_AUC}", (LEAKAGE_AUC, len(table) - 0.4),
                xytext=(4, 0), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    ax.scatter([], [], color=BLUE, label=f"feature in input (prime {TOP} su {X.shape[1]})")
    ax.scatter([], [], color=ORANGE, label="colonne renali escluse")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC univariato rispetto al target (training)")
    ax.set_title("Nessuna feature in input si avvicina alle colonne renali")
    ax.legend(loc="lower right")
    ax.grid(axis="x")
    save(fig, SECTION, "02_univariate_auc")


def missing_plot(train):
    X = select_features(train)
    na = (100 * X.isna().mean()).sort_values()
    na = na[na >= 1]
    excluded = 100 * train[FEATURES["exclude"]["high_missing"]].isna().mean()
    table = pd.concat([na, excluded]).sort_values()
    colors = [GRAY if c in excluded.index else BLUE for c in table.index]
    limit = 100 * FEATURES["max_missing"]

    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(table) + 1.4))
    bars = ax.barh(table.index, table.values, color=colors, height=0.7)
    label_bars(ax, bars, [f"{v:.1f}%" for v in table.values], horizontal=True)
    ax.axvline(limit, color=ORANGE, linestyle="--", linewidth=1.5)
    ax.annotate(f"soglia {limit:.0f}%", (limit, len(table) - 0.5), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("% di valori mancanti nel training")
    ax.set_title("Valori mancanti (colonne con almeno l'1%)")
    ax.legend(handles=[Patch(color=BLUE, label="in input, da imputare"),
                       Patch(color=GRAY, label="esclusa")], loc="lower right")
    ax.grid(axis="x")
    save(fig, SECTION, "03_missing_values")


def imputation_plot(feature_set):
    table = pd.read_csv(IMPUTATION / f"imputation_comparison_{feature_set}.csv", index_col=0)
    chosen = CONFIG["imputation"]["method"]
    methods = list(table.index)

    colors = [BLUE if m == chosen else GRAY for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    # RMSE: barre da zero
    axes[0].bar(methods, table["RMSE"], yerr=table["RMSE SE"], color=colors, width=0.6,
                capsize=4, error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1})
    axes[0].set_title("RMSE sui valori nascosti (più basso è meglio)")
    axes[0].set_ylabel("media sui 5 fold ± errore standard")
    rmse_best = table["RMSE"].idxmin()
    ceiling = table.loc[rmse_best, "RMSE"] + table.loc[rmse_best, "RMSE SE"]
    axes[0].axhline(ceiling, color=ORANGE, linestyle="--", linewidth=1.5)
    axes[0].annotate("migliore + 1 errore standard", (len(methods) - 0.55, ceiling),
                     xytext=(0, 4), textcoords="offset points", ha="right",
                     fontsize=8, color=TEXT_SECONDARY)
    # PR-AUC: punti con barre d'errore, l'asse non parte da zero
    for x, method in enumerate(methods):
        axes[1].errorbar(x, table.loc[method, "PR-AUC"], yerr=table.loc[method, "PR-AUC SE"],
                         fmt="o", color=colors[x], markersize=8, capsize=4,
                         ecolor=TEXT_SECONDARY, elinewidth=1)
    axes[1].set_xticks(range(len(methods)), methods)
    axes[1].set_xlim(-0.5, len(methods) - 0.5)
    axes[1].set_title("PR-AUC a valle (più alto è meglio)")
    best = table["PR-AUC"].idxmax()
    threshold = table.loc[best, "PR-AUC"] - table.loc[best, "PR-AUC SE"]
    axes[1].axhline(threshold, color=ORANGE, linestyle="--", linewidth=1.5)
    axes[1].annotate("migliore − 1 errore standard", (-0.45, threshold), xytext=(0, -10),
                     textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    for ax in axes:
        ax.grid(axis="y")
    fig.suptitle(f"Imputazione, set {feature_set}: adottato {chosen} (in blu, motivazione nel Notepad)",
                 x=0.01, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.82)
    save(fig, SECTION, f"04_imputation_{feature_set}")


if __name__ == "__main__":
    train = pd.read_csv(PROCESSED / "train.csv")
    column_map()
    auc_plot(train)
    missing_plot(train)
    for feature_set in FEATURE_SETS:
        if (IMPUTATION / f"imputation_comparison_{feature_set}.csv").exists():
            imputation_plot(feature_set)
        else:
            print(f"manca il confronto per {feature_set}: eseguire python -m src.data.imputation")
