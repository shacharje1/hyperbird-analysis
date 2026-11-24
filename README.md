# Exp4 hyperspectral analysis

Leaf-disk assay of 5 grape accessions x 7 powdery mildew isolates, scanned with HyperBird V2 at 0, 1, 2, 3 and 6 dpi
(the 0 dpi scan is taken shortly after inoculation) and scored with Blackbird (hyphae and spores, 1-10 dpi).
The code finds which spectral changes go with a successful defense and which with disease development.

## Stages

| Stage | Script | Input | Output |
|---|---|---|---|
| 1 | `01_inventory.py` | cube headers | `results/inventory.csv` |
| 2 | `02_blackbird.py` | two Blackbird spreadsheets | `results/blackbird/disk_measures.csv`, `blackbird_long.csv` |
| 3 | `03_roi_spectra.py` | HyperBird `processed/` outputs | `results/roi/spectra_mean.csv`, `summary.csv` |
| 4 | `04_pixel_spectra.py` | raw cubes | `results/pixel/spectra_mean.csv`, `summary.csv`, masks, previews |
| 5 | `05_indices.py` | a route's spectra | `<route>/indices.csv` |
| 6 | `06_first_day_nir.py` | measures + route | `<route>/stats/first_day_nir.txt` |
| 7 | `07_drop_by_vine.py` | measures + route | `<route>/stats/drop_by_vine*.csv`, `.txt` |
| 8 | `08_regreening_3to6.py` | measures + route | `<route>/stats/regreening_3to6.txt` |
| 9 | `09_sporulation.py` | measures + route | `<route>/stats/sporulation.txt` |
| 10 | `10_index_screen.py` | measures + route | `<route>/stats/index_screen.csv`, `.png` |
| 11 | `11_compare_routes.py` | both routes' index screens | `results/index_screen_both_routes.csv` |

Stages 3 and 4 are two independent routes to the same table (mean reflectance per disk and day, 400-900 nm).
Stages 5-10 run once per route (`results/roi`, `results/pixel`).

Helper modules: `design.py` (reads `design.json`: folders, references, time points, sample-name pattern),
`cube.py` (reference averaging), `colour.py` (CIELAB / sRGB from a spectrum, needs `D_illuminants.mat`),
`indices.py` (index formulas), `common.py` (loaders, disk classes, shared statistics).

## Run

```sh
pip install -r requirements.txt        # Python 3.13
sh run_all.sh DATA_ROOT Exp4_11032026_Hyphal2019Net2.xlsx Exp4_11032026_SporeNet3.xlsx
```

`DATA_ROOT` holds `Exp4_11032026_0dpi` ... `Exp4_11032026_6dpi`, each with `raw/` (cubes, `001_black`, `002_white`)
and `processed/` (HyperBird outputs). Sample names are `<id>_<Accession>_<Leaf#>_<Isolate>`; the id is the scan
position and identifies the disk across days and in Blackbird.

Stage 4 reads about 13 TB of cubes (about 3 minutes per cube) and is resumable. Run one copy per scan day in
parallel by passing an inventory filtered to that day's folder.

## Definitions

- Disk class, from Blackbird hyphae at 10 dpi: disease >= 50 %, defense < 25 %, in between left out.
- Resistant vines: every accession except Chardonnay.
- Near-infrared = mean reflectance 760-880 nm. CIgreen = R800 / R550 - 1.
- PRI = (R570 - R531) / (R570 + R531), the opposite sign to HyperBird's PRI.
