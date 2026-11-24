"""Stage 10: screen every index (7 core + 62 from Zarco-Tejada et al. 2018, Supplementary Table 1).

Per index and per measure (first-scan level, change 0 -> 1 dpi, change 3 -> 6 dpi, change 0 -> 6 dpi):
  defense, disease, chardonnay   class means
  d, auc, p, q                   defense vs resistant disease: standardized difference, Mann-Whitney AUC
                                 (> 0.5: higher in disease) and p
  rho_cell_hyphae / _spores      Spearman with hyphae (spores) at 10 dpi inside accession x isolate: both centred on
                                 the cell mean, cells of >= 3 disks, resistant vines, all disks
  rho_leaf_no258                 Spearman with hyphae inside the leaf (centred on the leaf mean), isolate 258 left out
q = Benjamini-Hochberg over all index x measure tests of the same kind.
Writes index_screen.csv and index_screen.png (the two correlation columns as heat maps).

Usage: python 10_index_screen.py results/blackbird/disk_measures.csv results/roi      (or results/pixel)
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import T0, T1, T3, T6, centred_spearman, groups3, load_measures, load_route
from indices import CORE, EXTRA, FAMILY

warnings.filterwarnings("ignore")
m = load_measures(sys.argv[1])
route = Path(sys.argv[2])
W, X, wl = load_route(route, m)
out = route / "stats"
out.mkdir(exist_ok=True)
INDICES = CORE + EXTRA
MEASURES = {f"level {T0}": lambda i: W[(i, T0)], f"change {T0}->{T1}": lambda i: W[(i, T1)] - W[(i, T0)],
            f"change {T3}->{T6}": lambda i: W[(i, T6)] - W[(i, T3)], f"change {T0}->{T6}": lambda i: W[(i, T6)] - W[(i, T0)]}
G = groups3(m)
res, no258 = m[m.resistant], m[m.resistant & (m.cond != "258")]

rows = []
for i in INDICES:
    for name, fn in MEASURES.items():
        v = fn(i).reindex(m.index)
        x, y, z = (v[G[g]].dropna() for g in ("resistant, defense", "resistant, disease", "Chardonnay, disease"))
        u, p = stats.mannwhitneyu(y, x)
        rc, pc, nc = centred_spearman(v, m.hyphae_10dpi, res[["group", "cond"]], min_size=3)
        rs, ps, _ = centred_spearman(v, m.spores_10dpi, res[["group", "cond"]], min_size=3)
        rl, pl, nl = centred_spearman(v, m.hyphae_10dpi, no258[["group", "leaf"]], min_size=3)
        rows.append(dict(index=i, family=FAMILY[i], source="core" if i in CORE else "table", measure=name,
                         defense=x.mean(), disease=y.mean(), chardonnay=z.mean(), n_def=len(x), n_dis=len(y),
                         d=(y.mean() - x.mean()) / np.sqrt((x.var() + y.var()) / 2), auc=u / (len(x) * len(y)), p=p,
                         rho_cell_hyphae=rc, p_cell_hyphae=pc, rho_cell_spores=rs, p_cell_spores=ps, n_cell=nc,
                         rho_leaf_no258=rl, p_leaf_no258=pl, n_leaf=nl))
R = pd.DataFrame(rows)
for c in ("p", "p_cell_hyphae", "p_cell_spores", "p_leaf_no258"):
    ok = R[c].notna()
    R.loc[ok, c.replace("p", "q", 1)] = multipletests(R.loc[ok, c], method="fdr_bh")[1]
R.to_csv(out / "index_screen.csv", index=False)
for name in MEASURES:
    q = R[R.measure == name]
    print(f"{name}: {(q.q < 0.05).sum()} of {len(q)} differ defense vs disease; {(q.q_cell_hyphae < 0.05).sum()} hold inside "
          f"accession x isolate; {(q.q_leaf_no258 < 0.05).sum()} hold inside the leaf without 258")

order = R[R.measure == f"level {T0}"].set_index("index").rho_cell_hyphae.sort_values().index
fig, axs = plt.subplots(1, 2, figsize=(9.5, 15), sharey=True)
panels = (("rho_cell_hyphae", "q_cell_hyphae", "inside accession × isolate"), ("rho_leaf_no258", "q_leaf_no258", "inside the leaf, without 258"))
for ax, (rho, q, title) in zip(axs, panels):
    M = R.pivot(index="index", columns="measure", values=rho).loc[order, list(MEASURES)]
    Q = R.pivot(index="index", columns="measure", values=q).loc[order, list(MEASURES)]
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    for r_, c_ in zip(*np.where(Q.values < 0.05)):
        ax.text(c_, r_, f"{M.values[r_, c_]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(MEASURES)))
    ax.set_xticklabels(list(MEASURES), rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{i}{' *' if i in CORE else ''}" for i in order], fontsize=7)
    ax.set_title(f"Spearman ρ with hyphae at 10 dpi\n{title}", fontsize=9)
fig.colorbar(im, ax=axs, shrink=0.4, label="ρ (number shown where q < 0.05)")
fig.suptitle("Exp4 – all indices against hyphae at 10 dpi, resistant vines (* = core index)", fontsize=11)
fig.savefig(out / "index_screen.png", dpi=140, bbox_inches="tight")
