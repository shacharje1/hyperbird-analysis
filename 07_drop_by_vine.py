"""Stage 7: per vine, the 0 -> 1 dpi near-infrared change under each isolate against the disease that isolate caused.

Per accession x isolate: mean change, hyphae and spores at 10 dpi (drop_by_vine_isolate.csv).
Per accession, Spearman of the change with hyphae, with spores and with scan order (drop_by_vine_correlations.csv):
  isolate means        across the 7 isolates
  disks                across all disks of the accession
  disks, within leaf   both variables centred on the leaf mean
each with all isolates and without isolate 258. Chardonnay is the check on scan order: every isolate causes disease
there, so a trend along the scan order in Chardonnay cannot come from the disease outcome.

Usage: python 07_drop_by_vine.py results/blackbird/disk_measures.csv results/roi      (or results/pixel)
"""
import sys
from pathlib import Path

import pandas as pd

from common import ISOLATES, NIR, T0, T1, Report, band, load_measures, load_route, spearman

VINES = ["Chardonnay", "Karadzhandal", "OlmoO35-11", "LateVavilov", "SochalKaraKash"]
m = load_measures(sys.argv[1])
route = Path(sys.argv[2])
W, X, wl = load_route(route, m)
out = route / "stats"
out.mkdir(exist_ok=True)
P = Report(out / "drop_by_vine.txt")

m["dNIR"] = band(X, wl, NIR, T1) - band(X, wl, NIR, T0)
F = m.dropna(subset=["dNIR", "hyphae_10dpi"])
T = F.groupby(["group", "cond"]).agg(n=("dNIR", "size"), dNIR=("dNIR", "mean"), dNIR_se=("dNIR", "sem"),
                                     hyphae=("hyphae_10dpi", "mean"), spores=("spores_10dpi", "mean")).reset_index()
T.to_csv(out / "drop_by_vine_isolate.csv", index=False)
P("Mean 0 -> 1 dpi near-infrared change, hyphae and spores at 10 dpi, per accession x isolate (scan order)")
for v in VINES:
    P("\n" + v)
    P(T[T.group == v].set_index("cond").loc[ISOLATES, ["n", "dNIR", "dNIR_se", "hyphae", "spores"]].round(4).to_string())

rows = []
for v in VINES:
    for tag, drop in (("all isolates", None), ("without 258", "258")):
        f, t = F[(F.group == v) & (F.cond != drop)], T[(T.group == v) & (T.cond != drop)]
        cols = ["dNIR", "hyphae_10dpi", "spores_10dpi"]
        c = f[cols] - f.groupby("leaf")[cols].transform("mean")
        for measure, tc, fc in (("hyphae", "hyphae", "hyphae_10dpi"), ("spores", "spores", "spores_10dpi")):
            for level, (r, p, n) in (("isolate means", spearman(t.dNIR, t[tc])), ("disks", spearman(f.dNIR, f[fc])),
                                     ("disks, within leaf", spearman(c.dNIR, c[fc]))):
                rows.append(dict(vine=v, isolates=tag, measure=measure, level=level, rho=r, p=p, n=n))
        r, p, n = spearman(t.dNIR, t.cond.map({k: i for i, k in enumerate(ISOLATES)}))
        rows.append(dict(vine=v, isolates=tag, measure="scan order", level="isolate means", rho=r, p=p, n=n))
R = pd.DataFrame(rows)
R.to_csv(out / "drop_by_vine_correlations.csv", index=False)
P("\nSpearman rho of the change with disease (rho > 0: bigger drop goes with less disease)")
P(R.pivot_table(index=["vine", "isolates"], columns=["measure", "level"], values="rho", sort=False).round(2).to_string())
P("\np-values")
P(R.pivot_table(index=["vine", "isolates"], columns=["measure", "level"], values="p", sort=False).round(4).to_string())
