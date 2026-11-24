"""Stage 1: inventory of the raw ENVI cubes under DATA_ROOT (processed/ folders are skipped). Reads headers only.

One row per cube: folder, name, header fields (samples, lines, bands, wavelength range), file size against the size
the header implies (size_ok), and id / group (accession) / rep (leaf number) / cond (isolate) parsed from the sample
name with name_regex of design.json. Prints a summary of cubes per folder and any problem file.

Usage: python 01_inventory.py DATA_ROOT results/inventory.csv
"""
import os
import sys
from pathlib import Path

import pandas as pd
from spectral.io import envi

from design import NAME_RE

DTYPE_BYTES = {1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 12: 2, 13: 4, 14: 8, 15: 8}


def walk(root: Path):
    """Every .hdr / .raw under root, skipping hidden folders, '@' folders and HyperBird's processed/ outputs."""
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith((".", "@")) and d != "processed"]
        for f in files:
            if not f.startswith(".") and f.lower().endswith((".hdr", ".raw")):
                yield Path(folder) / f


def main(root: Path, out: Path):
    rows = []
    files = list(walk(root))
    stems = {}
    for p in files:
        stems.setdefault(p.with_suffix(""), set()).add(p.suffix.lower())

    for stem, exts in sorted(stems.items()):
        hdr, raw = stem.with_suffix(".hdr"), stem.with_suffix(".raw")
        row = {"folder": str(stem.parent.relative_to(root)), "name": stem.name,
               "has_hdr": ".hdr" in exts, "has_raw": ".raw" in exts}
        m = NAME_RE.match(stem.name)
        if m:
            row.update(m.groupdict())
        if row["has_raw"]:
            row["raw_bytes"] = raw.stat().st_size
        if row["has_hdr"]:
            h = envi.read_envi_header(str(hdr))
            s, l, b = int(h["samples"]), int(h["lines"]), int(h["bands"])
            wl = [float(w) for w in h.get("wavelength", [])]
            row.update(samples=s, lines=l, bands=b, interleave=h.get("interleave"),
                       data_type=int(h["data type"]), tint=h.get("tint"),
                       wl_min=min(wl) if wl else None, wl_max=max(wl) if wl else None,
                       expected_bytes=s * l * b * DTYPE_BYTES.get(int(h["data type"]), 0)
                       + int(h.get("header offset", 0)))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["size_ok"] = df.get("raw_bytes") == df.get("expected_bytes")
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    root = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "inventory.csv")
    df = main(root, out)
    print(f"{len(df)} cubes -> {out}\n")
    print(df.groupby("folder").agg(cubes=("name", "size"), missing_hdr=("has_hdr", lambda x: (~x).sum()),
                                   missing_raw=("has_raw", lambda x: (~x).sum()),
                                   bad_size=("size_ok", lambda x: (~x).sum()),
                                   GB=("raw_bytes", lambda x: round(x.sum() / 1e9)),
                                   ).to_string())
    print("\nDistinct shapes (samples, lines, bands, interleave, tint, wl range):")
    print(df.groupby(["folder", "samples", "lines", "bands", "interleave", "tint", "wl_min", "wl_max"],
                     dropna=False).size().to_string())
    print("\nGroups x condition:")
    print(pd.crosstab([df["folder"]], [df["group"], df["cond"]]).to_string())
    print("\nProblem files:")
    print(df[~df["size_ok"] | ~df["has_hdr"] | ~df["has_raw"]][
        ["folder", "name", "has_hdr", "has_raw", "raw_bytes", "expected_bytes"]].to_string())
