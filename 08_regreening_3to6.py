"""Stage 8: the change between 3 and 6 dpi (re-greening of defense disks), and which band drives CIgreen.

  1  change per day by group for CIgreen, a*, visible (580-660 nm) and near-infrared (760-880 nm) reflectance, with
     Blackbird hyphae cover at 3 and 6 dpi; Mann-Whitney defense vs disease on resistant vines
  2  same leaf, disease minus defense, change per day (t-test over leaves): all leaves, Karadzhandal only, without 258
  3  does the change scale with the hyphae added between 3 and 6 dpi? Spearman, both centred on the accession mean
  4  shape of the (disease - defense) change spectrum on resistant vines, fitted in 400-720 nm by non-negative least
     squares with a white-layer template (1 - mean leaf reflectance at the first scan) and a pigment-loss template (palest
     third minus greenest third of disks per accession at the first scan, ranked on CIgreen)
  5  green against near-infrared: total change 3 -> 6 dpi of near-infrared, green (545-555 nm) and CIgreen per group,
     and Spearman between them within defense disks

Usage: python 08_regreening_3to6.py results/blackbird/disk_measures.csv results/blackbird/blackbird_long.csv results/roi
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls

from common import DAYS, GREEN, NIR, T0, T3, T6, VIS, Report, band, groups3, leaf_paired, load_measures, load_route, spearman

warnings.filterwarnings("ignore")
m = load_measures(sys.argv[1])
b = pd.read_csv(sys.argv[2])
route = Path(sys.argv[3])
W, X, wl = load_route(route, m)
(route / "stats").mkdir(exist_ok=True)
P = Report(route / "stats" / "regreening_3to6.txt")
b["p"] = 100 * b.inf / (b.inf + b.clr)
H = b[b.kind == "hyphae"].pivot_table(index="id", columns="dpi", values="p")
G = groups3(m)
span = DAYS[T6] - DAYS[T3]

level = lambda v, t: band(X, wl, {"NIR": NIR, "VIS": VIS, "GREEN": GREEN}[v], t) if v in ("NIR", "VIS", "GREEN") else W[(v, t)]
total = {v: (level(v, T6) - level(v, T3)).reindex(m.index) for v in ("CIgreen", "a*", "VIS", "NIR", "GREEN")}

P("1. Change per day, 3 -> 6 dpi")
rows = {name: dict(n=int(sel.sum()), hyphae_3dpi=H.loc[m.index[sel], 3].mean(), hyphae_6dpi=H.loc[m.index[sel], 6].mean(),
                   **{v: total[v][sel].mean() / span for v in ("CIgreen", "a*", "VIS", "NIR")}) for name, sel in G.items()}
P(pd.DataFrame(rows).round(4).to_string())
for v in ("CIgreen", "a*", "VIS", "NIR"):
    p = stats.mannwhitneyu(total[v][G["resistant, disease"]].dropna(), total[v][G["resistant, defense"]].dropna())[1]
    P(f"   {v:8s} resistant disease vs defense, Mann-Whitney p={p:.2g}")

P("\n2. Same leaf, disease minus defense, change per day (resistant vines)")
R = m[m.resistant]
for label, K in (("all leaves", R), ("Karadzhandal only", R[R.group == "Karadzhandal"]), ("without 258", R[R.cond != "258"])):
    for v in ("a*", "CIgreen"):
        d = leaf_paired(total[v] / span, K)
        P(f"   {label:18s} {v:8s} {d.mean():+.4f} per day, {len(d)} leaves, {100 * (d > 0).mean():.0f}% positive, p={stats.ttest_1samp(d, 0)[1]:.2g}")

P("\n3. Spearman of the change with the hyphae added 3 -> 6 dpi (accession-centred)")
dh = (H[6] - H[3]).reindex(m.index)
for scope, sel in (("resistant vines", m.resistant), ("resistant vines, disease disks", G["resistant, disease"]), ("Chardonnay", ~m.resistant)):
    line = f"   {scope:32s}"
    for v in ("VIS", "a*", "CIgreen", "NIR"):
        f = pd.DataFrame({"y": total[v], "dh": dh, "group": m.group})[sel].dropna()
        f[["y", "dh"]] -= f.groupby("group")[["y", "dh"]].transform("mean")
        r, p, n = spearman(f.y, f.dh)
        line += f"  {v} {r:+.2f} (p={p:.2g})"
    P(line + f"  n={n}")

P("\n4. Shape of the disease-minus-defense change spectrum, resistant vines, fitted in 400-720 nm (R2 about the mean)")
ids = X[T6].index.intersection(X[T3].index)
D = X[T6].loc[ids] - X[T3].loc[ids]
mean_of = lambda sel: D.loc[ids.intersection(m.index[sel])].values.mean(0)
target = mean_of(G["resistant, disease"]) - mean_of(G["resistant, defense"])
x0 = X[T0].loc[X[T0].index.intersection(m.index[m.resistant])]
white = 1 - x0.values.mean(0)
rank = W[("CIgreen", T0)].reindex(x0.index).groupby(m.group.reindex(x0.index)).rank(pct=True)
pale = x0[(rank < 1 / 3).values].values.mean(0) - x0[(rank > 2 / 3).values].values.mean(0)
vis = wl <= 720
for name, cols in (("white layer only", [white]), ("pigment loss only", [pale]), ("both", [white, pale])):
    A = np.column_stack(cols)[vis]
    c, _ = nnls(A, target[vis])
    r2 = 1 - ((target[vis] - A @ c) ** 2).sum() / ((target[vis] - target[vis].mean()) ** 2).sum()
    P(f"   {name:18s} R2={r2:+.2f}")

P("\n5. Green against near-infrared: total change 3 -> 6 dpi")
P(pd.DataFrame({name: {v: total[v][sel].mean() for v in ("NIR", "GREEN", "CIgreen")} for name, sel in G.items()}).round(4).to_string())
p = stats.mannwhitneyu(total["GREEN"][G["resistant, defense"]].dropna(), total["GREEN"][G["resistant, disease"]].dropna())[1]
P(f"   green change, defense vs resistant disease, Mann-Whitney p={p:.2g}")
d = pd.DataFrame(total)[G["resistant, defense"]].dropna()
P(f"   within defense disks: rho(CIgreen change, green change)={spearman(d.CIgreen, d.GREEN)[0]:+.2f}, "
  f"rho(CIgreen change, NIR change)={spearman(d.CIgreen, d.NIR)[0]:+.2f}")
P("   mean near-infrared of defense disks by day: " + ", ".join(f"{t} {band(X, wl, NIR, t).reindex(m.index[G['resistant, defense']]).mean():.3f}" for t in X))
P("   mean green of defense / resistant disease disks at 0 and 6 dpi: "
  + " / ".join(", ".join(f"{band(X, wl, GREEN, t).reindex(m.index[G[g]]).mean():.3f}" for t in (T0, T6)) for g in ("resistant, defense", "resistant, disease")))
