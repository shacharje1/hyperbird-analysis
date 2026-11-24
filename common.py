"""Shared loaders and helpers for the analysis stages (06-10)."""
import numpy as np
import pandas as pd
from scipy import stats

from design import TIMES

ISOLATES = ["258", "281", "287", "288", "300", "301", "G19"]  # also the scan order within every scan day
DISEASE_MIN, DEFENSE_MAX = 50, 25                             # hyphae % at 10 dpi
NIR, GREEN, VIS = (760, 880), (545, 555), (580, 660)          # band windows, nm
T0, T1, T3, T6 = TIMES[0], TIMES[1], TIMES[3], TIMES[4]
DAYS = {t: float(t.split()[0]) for t in TIMES}


def load_measures(path):
    """Blackbird measures per disk (from 02_blackbird.py) with the disk class and a resistant-vine flag."""
    m = pd.read_csv(path, dtype={"cond": str}).set_index("id")
    m["cls"] = np.where(m.hyphae_10dpi >= DISEASE_MIN, "disease", np.where(m.hyphae_10dpi < DEFENSE_MAX, "defense", "mid"))
    m["resistant"] = m.group != "Chardonnay"
    return m


def load_route(route_dir, m):
    """Indices and mean spectra of one route (results/roi or results/pixel), for disks with Blackbird measures.

    Returns W: indices as a (index, time) x disk table; X: {time: disk x wavelength reflectance}; wl: wavelengths.
    """
    d = pd.read_csv(f"{route_dir}/indices.csv", dtype={"cond": str})
    d = d[d.id.isin(m.index)].drop_duplicates(["id", "dpi"])
    sp = pd.read_csv(f"{route_dir}/spectra_mean.csv")
    wl = sp.columns[5:].astype(float).values
    S = d[["folder", "name", "id", "dpi"]].merge(sp.drop(columns=["group", "cond", "rep"]), on=["folder", "name"])
    X = {t: pd.DataFrame(S[S.dpi == t].iloc[:, 4:].values.astype(float), index=S[S.dpi == t].id.values) for t in TIMES}
    num = [c for c in d.columns if c not in ("folder", "name", "group", "cond", "rep", "series", "id", "dpi", "session", "folded")]
    W = d.pivot(index="id", columns="dpi", values=num)
    return W, X, wl


def band(X, wl, window, t):
    """Mean reflectance of every disk in a wavelength window at time t."""
    lo, hi = window
    return X[t].iloc[:, np.where((wl >= lo) & (wl <= hi))[0]].mean(1)


def groups3(m):
    """The three disk groups of the report."""
    return {"Chardonnay, disease": ~m.resistant & (m.cls == "disease"),
            "resistant, disease": m.resistant & (m.cls == "disease"),
            "resistant, defense": m.resistant & (m.cls == "defense")}


def spearman(x, y):
    k = x.notna() & y.notna()
    if k.sum() < 5 or np.ptp(x[k]) == 0 or np.ptp(y[k]) == 0:
        return np.nan, np.nan, int(k.sum())
    r, p = stats.spearmanr(x[k], y[k])
    return r, p, int(k.sum())


def centred_spearman(x, y, by, min_size=1):
    """Spearman of x and y after subtracting the mean of each `by` cell from both (by: DataFrame of grouping columns)."""
    f = pd.DataFrame({"x": x, "y": y}).join(by, how="inner").dropna()
    g = f.groupby(list(by.columns))
    c = f[["x", "y"]] - g[["x", "y"]].transform("mean")
    return spearman(*(c[g.x.transform("size") >= min_size][k] for k in ("x", "y")))


def leaf_paired(v, m):
    """Per leaf: mean of v over its disease disks minus mean over its defense disks (leaves that have both)."""
    k = m[m.cls != "mid"].join(v.rename("v"), how="inner").dropna(subset=["v"])
    w = k.groupby(["leaf", "cls"]).v.mean().unstack().dropna()
    return w.disease - w.defense


class Report:
    """Print and write the same lines to a text file."""

    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *x):
        s = " ".join(str(v) for v in x)
        print(s)
        self.f.write(s + "\n")
