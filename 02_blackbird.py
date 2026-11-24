"""Stage 2: Blackbird hyphae and spore readings -> one row per disk.

Input: the two Blackbird spreadsheets (hyphae network Hyphal2019Net2, spore network SporeNet3). Each has one block of
four columns per day (%, inf, CLR, ALL) and one row per disk; SampleID = <id>-<Accession>_<Leaf#>_<Isolate>.
Percentage used everywhere = 100 * infected patches / (infected + clear patches), recomputed from the counts.

Output:
  blackbird_long.csv   one row per disk x network x day
  disk_measures.csv    disks with a hyphae reading at 10 dpi: hyphae at 6 and 10 dpi, spores at 10 dpi, AUDPC of both
                       (trapezoid over 1-10 dpi), spores/hyphae at 10 dpi (where hyphae >= 10 %), first day from which
                       spores stay >= 5 %

Usage: python 02_blackbird.py HYPHAE.xlsx SPORES.xlsx results/blackbird
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAME = re.compile(r"^(?P<id>\d+)-(?P<group>[^_]+)_(?P<rep>\d+)_(?P<cond>[^_]+)$")


def read_sheet(path, kind):
    x = pd.read_excel(path, header=None)
    rows = []
    for c in range(1, x.shape[1], 4):
        date, dpi = re.match(r"(.+)_(\d+)dpi", str(x.iat[0, c])).groups()
        for r in range(2, len(x)):
            mt = NAME.match(str(x.iat[r, 0]).strip())
            if mt and pd.notna(x.iat[r, c + 1]):
                rows.append(dict(sample=x.iat[r, 0], kind=kind, dpi=int(dpi), date=date, pct=x.iat[r, c], inf=x.iat[r, c + 1],
                                 clr=x.iat[r, c + 2], all=x.iat[r, c + 3], id=int(mt["id"]), group=mt["group"],
                                 rep=int(mt["rep"]), cond=mt["cond"]))
    return pd.DataFrame(rows)


def main(hyphae_xlsx, spores_xlsx, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    b = pd.concat([read_sheet(hyphae_xlsx, "hyphae"), read_sheet(spores_xlsx, "spores")], ignore_index=True)
    b.to_csv(out / "blackbird_long.csv", index=False)

    b["p"] = 100 * b.inf / (b.inf + b.clr)
    b["leaf"] = b.group + "_" + b.rep.astype(str)
    W = b.pivot_table(index="id", columns=["kind", "dpi"], values="p")
    days = sorted(b.dpi.unique())
    info = b.drop_duplicates("id").set_index("id")[["group", "cond", "leaf"]]
    m = info[W[("hyphae", 10)].notna().reindex(info.index).fillna(False)].copy()
    audpc = lambda kind: np.trapezoid(W.loc[m.index, kind][days].values, days, axis=1)
    m["hyphae_6dpi"] = W.loc[m.index, ("hyphae", 6)]
    m["hyphae_10dpi"] = W.loc[m.index, ("hyphae", 10)]
    m["hyphae_AUDPC"] = audpc("hyphae")
    m["spores_10dpi"] = W.loc[m.index, ("spores", 10)]
    m["spores_AUDPC"] = audpc("spores")
    m["spor_ratio"] = np.where(m.hyphae_10dpi >= 10, m.spores_10dpi / m.hyphae_10dpi, np.nan)
    stays = (W.loc[m.index, "spores"][days] >= 5).iloc[:, ::-1].cummin(axis=1).iloc[:, ::-1]
    m["first_spore_day"] = np.where(stays.any(axis=1), stays.idxmax(axis=1), np.nan)
    m.to_csv(out / "disk_measures.csv")
    print(f"{len(m)} of {len(info)} disks have a full Blackbird series to 10 dpi -> {out / 'disk_measures.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
