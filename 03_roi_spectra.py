"""Stage 3 (ROI route): mean spectrum per disk from HyperBird's own processed outputs. No cube is read.

For every sample in the inventory (white/black skipped) reads
  <dataset>/processed/<sample>/<sample>_roi_stats.csv   ROI mean and SD reflectance per band
  <dataset>/processed/<sample>/<sample>_masks.npy       ROI mask, for the ROI area
and writes, in the same format as stage 4:
  spectra_mean.csv, spectra_std.csv   400-900 nm
  summary.csv                         ROI area; low_tissue = ROI area < MIN_REL_AREA x the folder median
                                      (HyperBird's segmentation locked onto a vein or speck: scan dropped downstream)

Usage: python 03_roi_spectra.py DATA_ROOT results/inventory.csv results/roi [--wl-min 400 --wl-max 900]
"""
import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


def read_roi_stats(path):
    """HyperBird v2 roi_stats.csv -> (wavelengths, mean, sd). A few header lines, then 'roi_id,type,<wl...>' rows."""
    lines = Path(path).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("roi_id,"))
    df = pd.read_csv(StringIO("\n".join(lines[start:])))
    row = lambda kind: df[df["type"].str.strip() == kind].iloc[0, 2:].astype(float).values
    return df.columns[2:].astype(float).values, row("mean"), row("std")


def main(a):
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(a.inventory)
    inv = inv[~inv["name"].str.contains("white|black", case=False)]
    means, sds, summary, missing, bands = [], [], [], [], None
    for _, r in inv.iterrows():
        dataset = Path(r.folder).parent if Path(r.folder).name == "raw" else Path(r.folder)
        pdir = root / dataset / "processed" / r["name"]
        csv = pdir / f"{r['name']}_roi_stats.csv"
        if not csv.exists():
            missing.append(f"{r.folder}/{r['name']}")
            continue
        wl, mean, sd = read_roi_stats(csv)
        keep = (wl >= a.wl_min) & (wl <= a.wl_max)
        if bands is None:
            bands = wl[keep]
        elif len(bands) != keep.sum() or not np.allclose(bands, wl[keep]):
            raise SystemExit(f"band set differs in {csv}")
        meta = dict(folder=r.folder, name=r["name"], group=r.get("group"), cond=r.get("cond"), rep=r.get("rep"))
        cols = [f"{w:.2f}" for w in bands]
        means.append({**meta, **dict(zip(cols, mean[keep]))})
        sds.append({**meta, **dict(zip(cols, sd[keep]))})
        mask = pdir / f"{r['name']}_masks.npy"
        area = int(np.load(mask).reshape(-1, *np.load(mask, mmap_mode="r").shape[-2:]).any(0).sum()) if mask.exists() else np.nan
        summary.append(dict(folder=r.folder, name=r["name"], tissue_px=area, source="hyperbird"))
    s = pd.DataFrame(summary)
    s["low_tissue"] = s.tissue_px / s.groupby("folder").tissue_px.transform("median") < a.min_rel_area
    pd.DataFrame(means).to_csv(out / "spectra_mean.csv", index=False)
    pd.DataFrame(sds).to_csv(out / "spectra_std.csv", index=False)
    s.to_csv(out / "summary.csv", index=False)
    print(f"{len(s)} samples -> {out} ({len(bands)} bands {bands[0]:.1f}-{bands[-1]:.1f} nm); "
          f"failed segmentation {int(s.low_tissue.sum())}; missing {len(missing)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root"); p.add_argument("inventory"); p.add_argument("out")
    p.add_argument("--wl-min", type=float, default=400.0); p.add_argument("--wl-max", type=float, default=900.0)
    p.add_argument("--min-rel-area", type=float, default=0.8)
    main(p.parse_args())
