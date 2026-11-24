"""Stage 6: the near-infrared (760-880 nm) change from the first scan (0 dpi) to 1 dpi, defense vs disease disks.

Tests, all on the per-disk change dNIR = NIR(1 dpi) - NIR(0 dpi):
  1  group means and Mann-Whitney tests between the three groups
  2  Spearman with hyphae at 10 dpi on resistant vines: all disks; within leaf; within accession x isolate
     (the last two after centring both variables on the cell mean), each also without isolate 258
  3  same leaf: mean of disease disks minus mean of defense disks, one value per leaf, t-test and Wilcoxon against 0
  4  mixed model on resistant vines (accession fixed, leaf random), without and with the first-scan level as covariate
  5  mean change per isolate on resistant vines (isolates are in scan order)

Usage: python 06_first_day_nir.py results/blackbird/disk_measures.csv results/roi      (or results/pixel)
"""
import sys
import warnings
from pathlib import Path

import statsmodels.formula.api as smf
from scipy import stats

from common import ISOLATES, NIR, T0, T1, Report, band, centred_spearman, groups3, leaf_paired, load_measures, load_route, spearman

warnings.filterwarnings("ignore")
m = load_measures(sys.argv[1])
route = Path(sys.argv[2])
W, X, wl = load_route(route, m)
(route / "stats").mkdir(exist_ok=True)
P = Report(route / "stats" / "first_day_nir.txt")

m["NIR0"] = band(X, wl, NIR, T0)
m["dNIR"] = band(X, wl, NIR, T1) - m.NIR0
F = m.dropna(subset=["dNIR"])
G = groups3(F)

P("1. Change in near-infrared reflectance, 0 -> 1 dpi")
for name, sel in G.items():
    v = F.dNIR[sel]
    P(f"   {name:22s} n={len(v):3d}  mean {v.mean():+.4f}  median {v.median():+.4f}")
mw = lambda a, b: stats.mannwhitneyu(F.dNIR[G[a]], F.dNIR[G[b]])[1]
P(f"   Mann-Whitney: defense vs resistant disease p={mw('resistant, defense', 'resistant, disease'):.2g}; "
  f"defense vs Chardonnay p={mw('resistant, defense', 'Chardonnay, disease'):.2g}; "
  f"resistant disease vs Chardonnay p={mw('resistant, disease', 'Chardonnay, disease'):.2g}")

P("\n2. Spearman of the change with hyphae at 10 dpi, resistant vines (rho > 0: bigger drop, less disease)")
for label, R in (("all isolates", F[F.resistant]), ("without 258", F[F.resistant & (F.cond != "258")])):
    P(f"   {label:13s} all disks rho={spearman(R.dNIR, R.hyphae_10dpi)[0]:+.2f}"
      + "".join(f" | {name} rho={r:+.2f} p={p:.2g} n={n}" for name, (r, p, n) in (
          ("within leaf", centred_spearman(R.dNIR, R.hyphae_10dpi, R[["leaf"]])),
          ("within accession x isolate", centred_spearman(R.dNIR, R.hyphae_10dpi, R[["group", "cond"]])))))

P("\n3. Same leaf, disease minus defense (one value per leaf)")
for label, R in (("all isolates", F), ("without 258", F[F.cond != "258"])):
    d = leaf_paired(R.dNIR, R)
    P(f"   {label:13s} mean {d.mean():+.4f} +- {d.sem():.4f}, {len(d)} leaves, {int((d > 0).sum())} positive; "
      f"t-test p={stats.ttest_1samp(d, 0)[1]:.3g}, Wilcoxon p={stats.wilcoxon(d)[1]:.3g}")

P("\n4. Mixed model, resistant vines: dNIR ~ disease + accession (+ first-scan NIR), leaf random")
R = F[F.resistant & (F.cls != "mid")].assign(disease=lambda x: (x.cls == "disease").astype(int))
for label, formula in (("unadjusted", "dNIR ~ disease + C(group)"), ("adjusted for first-scan level", "dNIR ~ disease + C(group) + NIR0")):
    fit = smf.mixedlm(formula, R, groups=R.leaf).fit()
    P(f"   {label:30s} disease - defense = {fit.params['disease']:+.4f}, p={fit.pvalues['disease']:.2g}")

P("\n5. Mean per isolate, resistant vines (scan order)")
t = F[F.resistant].groupby("cond")[["dNIR", "hyphae_10dpi"]].mean().loc[ISOLATES]
P(t.round(4).to_string())
P(f"   across the 7 isolate means: rho={stats.spearmanr(t.dNIR, t.hyphae_10dpi)[0]:+.2f}")
