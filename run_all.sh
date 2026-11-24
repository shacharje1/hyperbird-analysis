#!/bin/sh
# Run every stage in order.
# Usage: sh run_all.sh DATA_ROOT HYPHAE.xlsx SPORES.xlsx
#   DATA_ROOT   folder holding Exp4_11032026_0dpi ... Exp4_11032026_6dpi (each with raw/ and processed/)
set -e
ROOT=$1; HYPHAE=$2; SPORES=$3
M=results/blackbird/disk_measures.csv

python 01_inventory.py "$ROOT" results/inventory.csv
python 02_blackbird.py "$HYPHAE" "$SPORES" results/blackbird
python 03_roi_spectra.py "$ROOT" results/inventory.csv results/roi
python 04_pixel_spectra.py "$ROOT" results/inventory.csv results/pixel    # reads every cube: about 3 min each

for R in results/roi results/pixel; do
    python 05_indices.py $R
    python 06_first_day_nir.py $M $R
    python 07_drop_by_vine.py $M $R
    python 08_regreening_3to6.py $M results/blackbird/blackbird_long.csv $R
    python 09_sporulation.py $M $R
    python 10_index_screen.py $M $R
done
python 11_compare_routes.py results/roi results/pixel results
