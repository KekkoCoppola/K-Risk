"""C3 del piano, ultima voce: in T l'importanza di ogni colonna accanto alla sua quota di mancanti.

Non calcolata da `05_xai.py`; si ricava dai suoi CSV senza ristimare nulla. Quota di mancanti
sul file grezzo del training (prima di ogni imputazione); importanza = media sui 5 fold della
quota SHAP sul punteggio. Solo statistiche per colonna, nessuna riga di paziente.

Uscite: `04_esperimenti/non_supervisionato/xai/T/C3_importanza_e_mancanti.csv` (per colonna e
configurazione) e `C3_importanza_e_mancanti_sintesi.csv` (Spearman e quota delle colonne con più
del 50% di mancanti fra le prime 10).
"""
import pandas as pd
from scipy import stats

import prep

ROOT = prep.ROOT
X = ROOT / "analisi_impatto/04_esperimenti/non_supervisionato/xai/T"


def main():
    df = pd.read_csv(ROOT / "data/processed/train.csv", low_memory=False)
    miss = df.isna().mean()
    imp = pd.read_csv(X / "C1_importanze.csv")
    imp = imp[(imp.uscita == "punteggio") & (imp.metodo == "shap")]
    m = imp.groupby(["configurazione", "variabile", "famiglia"]).importanza.mean().reset_index()
    m["quota"] = m.importanza / m.groupby("configurazione").importanza.transform("sum")
    m["rango"] = m.groupby("configurazione").importanza.rank(ascending=False, method="first").astype(int)
    m["quota_mancanti_grezza"] = m.variabile.map(miss)
    m.sort_values(["configurazione", "rango"]).to_csv(X / "C3_importanza_e_mancanti.csv", index=False)
    rows = []
    for cfg, g in m.groupby("configurazione"):
        top = g.nsmallest(10, "rango")
        rows.append({"configurazione": cfg,
                     "spearman_importanza_mancanti": stats.spearmanr(g.importanza, g.quota_mancanti_grezza, nan_policy="omit")[0],
                     "top10_con_oltre_50pct_mancanti": int((top.quota_mancanti_grezza > 0.5).sum()),
                     "quota_shap_colonne_oltre_50pct_mancanti": float(g.loc[g.quota_mancanti_grezza > 0.5, "quota"].sum()),
                     "colonne_oltre_50pct_mancanti": int((g.quota_mancanti_grezza > 0.5).sum()),
                     "colonne": len(g)})
    s = pd.DataFrame(rows)
    s.to_csv(X / "C3_importanza_e_mancanti_sintesi.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
