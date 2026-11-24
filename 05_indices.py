"""Stage 5: indices and CIELAB colour per disk and day, from a route's mean spectra.

Formulas are in indices.py. Scans flagged low_tissue in summary.csv are dropped; time point and disk id are added
from design.json and the sample name.

Usage: python 05_indices.py results/roi      (or results/pixel)
"""
import sys

from indices import compute

route = sys.argv[1]
d = compute(f"{route}/spectra_mean.csv", f"{route}/summary.csv")
d.to_csv(f"{route}/indices.csv", index=False)
print(f"{len(d)} disk-days -> {route}/indices.csv")
print(d.groupby("dpi").size().to_string())
