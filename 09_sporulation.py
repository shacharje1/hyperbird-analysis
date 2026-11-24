"""Stage 9: among colonized disks on resistant vines, do sporulating and low-spore disks differ?

Colonized = hyphae >= 50 % at 10 dpi. Sporulating = spores >= 20 % at 10 dpi; low-spore = spores < 10 %.
  1  mixed model per measure: value ~ sporulating, leaf random. Measures: near-infrared change 0 -> 1 dpi,
     CIgreen change 0 -> 1 dpi, CIgreen change 3 -> 6 dpi
  2  all resistant-vine disks: Spearman of hyphae with spores, and the partial Spearman of each measure with spores
     given hyphae (correlation of the residuals of the ranks)

Usage: python 09_sporulation.py results/blackbird/disk_measures.csv results/pixel      (or results/roi)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import NIR, T0, T1, T3, T6, Report, band, load_measures, load_route

SPORULATING_MIN, LOW_SPORE_MAX = 20, 10
warnings.filterwarnings("ignore")
m = load_measures(sys.argv[1])
route = Path(sys.argv[2])
W, X, wl = load_route(route, m)
(route / "stats").mkdir(exist_ok=True)
P = Report(route / "stats" / "sporulation.txt")

m["dNIR01"] = band(X, wl, NIR, T1) - band(X, wl, NIR, T0)
m["dCI01"] = W[("CIgreen", T1)] - W[("CIgreen", T0)]
m["dCI36"] = W[("CIgreen", T6)] - W[("CIgreen", T3)]
MEASURES = ["dNIR01", "dCI01", "dCI36"]

C = m[m.resistant & (m.cls == "disease")].copy()
C["spor"] = np.where(C.spores_10dpi >= SPORULATING_MIN, 1.0, np.where(C.spores_10dpi < LOW_SPORE_MAX, 0.0, np.nan))
C = C.dropna(subset=["spor"])
P(f"1. Colonized disks on resistant vines: {int(C.spor.sum())} sporulating, {int((C.spor == 0).sum())} low-spore")
for v in MEASURES:
    s = C.dropna(subset=[v])
    fit = smf.mixedlm(f"{v} ~ spor", s, groups=s.leaf).fit()
    P(f"   {v:7s} sporulating minus low-spore = {fit.params['spor']:+.4f}, p={fit.pvalues['spor']:.2g} (n={len(s)})")


def partial_spearman(x, y, z):
    f = pd.DataFrame({"x": x, "y": y, "z": z}).dropna().rank()
    res = lambda a: a - np.polyval(np.polyfit(f.z, a, 1), f.z)
    return stats.pearsonr(res(f.x), res(f.y))


R = m[m.resistant]
P(f"\n2. Resistant vines: hyphae vs spores at 10 dpi rho={stats.spearmanr(R.hyphae_10dpi, R.spores_10dpi)[0]:+.2f}")
for v in MEASURES:
    r, p = partial_spearman(R[v], R.spores_10dpi, R.hyphae_10dpi)
    P(f"   {v:7s} vs spores given hyphae: partial rho={r:+.2f}, p={p:.2g}")
