"""Stage 11: which index results hold on both routes (ROI and pixel level).

Joins the two index_screen.csv files. A test "passes on both routes" when q < 0.05 on each route and the effect has
the same sign. Three tests per index x measure: class (defense vs disease), cell (inside accession x isolate), leaf
(inside the leaf, without isolate 258). nir_free marks indices whose formula has no band at or above 750 nm.
Writes index_screen_both_routes.csv and prints the counts.

Usage: python 11_compare_routes.py results/roi results/pixel results
"""
import sys

import numpy as np
import pandas as pd

NIR_FREE = set("""MCARI TCARI VOG2 VOG3 SRPI NPQI NPCI CTRI1 DCabCxc CRI1 CRI2 CRI550_515 CRI700_515 PRI PRI515 PRIm1 PRIm2
PRIm3 PRIm4 RI GI BI BGI1 BGI2 BRI1 BRI2 BF1 BF2 BF3 BF4 BF5 RGI LIC2 CUR HI ARI""".split())
roi, pixel, out = sys.argv[1:4]
a = pd.read_csv(f"{roi}/stats/index_screen.csv")
b = pd.read_csv(f"{pixel}/stats/index_screen.csv")
m = a.merge(b, on=["index", "measure", "family", "source"], suffixes=("_roi", "_px"))
both = lambda q, effect: (m[f"{q}_roi"] < 0.05) & (m[f"{q}_px"] < 0.05) & (np.sign(m[f"{effect}_roi"]) == np.sign(m[f"{effect}_px"]))
m["nir_free"] = m["index"].isin(NIR_FREE)
m["class_both"] = both("q", "d")
m["cell_both"] = both("q_cell_hyphae", "rho_cell_hyphae")
m["leaf_both"] = (m.q_leaf_no258_roi < 0.05) & (m.q_leaf_no258_px < 0.05)
m["all_three"] = m.class_both & m.cell_both & m.leaf_both
m.to_csv(f"{out}/index_screen_both_routes.csv", index=False)
for measure, g in m.groupby("measure", sort=False):
    print(f"{measure}: class {g.class_both.sum()}, cell {g.cell_both.sum()}, all three {g.all_three.sum()} "
          f"(without near-infrared: {(g.all_three & g.nir_free).sum()}: {', '.join(g[g.all_three & g.nir_free]['index'])})")
