"""Punto 2, verifica dell'esito GRADUALE: pendenza lineare di ogni esame lungo log(ACR).
Esplicativo, solo training. Uscita: punto2/P2b_pendenze.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import punto2_corpo_lungo_acr as p2
from scipy import stats

df, _ = p2.load()
res = {v: p2.adjust(df, v) for v in p2.PRIMARIE + p2.CONTROLLI}
Z = pd.concat({v: res[v] * p2.SEGNO[v] for v in p2.PROFILO}, axis=1).dropna()
res["profilo"] = Z.mean(1)
rows = []
for v, y in res.items():
    x = df.loc[y.index, "x"]
    r = stats.linregress(x, y)
    # z per raddoppio dell'ACR
    rows.append({"variabile": v, "n": len(y), "pendenza_z_per_raddoppio": r.slope*np.log(2),
                 "ic_basso": (r.slope-1.96*r.stderr)*np.log(2), "ic_alto": (r.slope+1.96*r.stderr)*np.log(2),
                 "p": r.pvalue})
out = pd.DataFrame(rows)
prim = out["variabile"].isin(p2.PRIMARIE)


pp = out.loc[prim, "p"].to_numpy(); o = np.argsort(pp); m = len(pp); adj = np.maximum.accumulate((m - np.arange(m)) * pp[o]); h = np.empty(m); h[o] = np.minimum(1, adj); out.loc[prim, "p_holm"] = h
out.to_csv(p2.OUT / "P2b_pendenze.csv", index=False)
print(out.round(4).to_string(index=False))
