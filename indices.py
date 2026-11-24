"""Per-cube indices from a batch spectra CSV.

Core indices (columns kept under their old names): NDVI, PRI, PSRI, ARI, CIgreen, CRI1, CRI2. PRI and PSRI use the mean
reflectance within +-5 nm (as HyperBird does), the others the nearest single band.

Formulas follow Supplementary Table 1 of Zarco-Tejada et al. 2018 (Nature Plants 4: 432-439) where the index is in that
table:
  NDVI, PSRI, CRI1 (= CRI550), CRI2 (= CRI700)   already identical to the table
  PRI    = (R570 - R531) / (R570 + R531)          the table's PRI570. NOTE: opposite sign to HyperBird's own PRI.
  ARI, CIgreen                                    not in the table, kept as they were.
All other narrow-band indices of that table are added (TABLE_INDICES, nearest single band), written as the last columns.
Not computable from these cubes: SIF (FLD) and canopy temperature / CWSI. Where the table's printed equation has a
typesetting error the formula of the cited paper is used: MSAVI, MTVI2, MCARI2, MSR, HI, and EVI (third term 7.5 * R500;
the table prints R800 there). R400 is the first band of the sensor range (noisy).

Also the disk colour as the eye sees it under D65: CIELAB L*, a*, b* (colour.py, same maths as HyperBird).
Drops low-tissue cubes and folders marked analyse=false in design.json; adds dpi (time point), session, series and disk id.
Used by 05_indices.py (which writes the table) and 10_index_screen.py (which reads the index lists)."""
import sys
import numpy as np
import pandas as pd

from design import annotate
from colour import to_lab

HALF = 5
CORE = ["NDVI", "PRI", "PSRI", "ARI", "CIgreen", "CRI1", "CRI2"]
TABLE_INDICES = {
    "structural": ["RDVI", "OSAVI", "MSAVI", "TVI", "MTVI1", "MTVI2", "MCARI", "MCARI1", "MCARI2", "SR", "MSR", "EVI"],
    "pigment": ["VOG1", "VOG2", "VOG3", "GM1", "GM2", "TCARI", "TCARI_OSAVI", "CIre", "SRPI", "NPQI", "NPCI", "CTRI1", "CAR",
                "DCabCxc", "DNIRCabCxc", "SIPI", "CRI550_515", "CRI700_515", "RNIR_CRI550", "RNIR_CRI700",
                "PSSRa", "PSSRb", "PSSRc", "PSNDc"],
    "xanthophyll": ["PRI515", "PRIm1", "PRIm2", "PRIm3", "PRIm4", "PRIn", "PRI_CI"],
    "RGB": ["RI", "GI", "BI", "BGI1", "BGI2", "BRI1", "BRI2", "BF1", "BF2", "BF3", "BF4", "BF5", "RGI", "RARS", "LIC1", "LIC2", "LIC3"],
    "fluorescence": ["CUR"],
    "disease": ["HI"],
}
EXTRA = [i for v in TABLE_INDICES.values() for i in v]
FAMILY = {i: k for k, v in TABLE_INDICES.items() for i in v}
FAMILY.update({"NDVI": "structural", "PRI": "xanthophyll", "PSRI": "pigment", "ARI": "pigment", "CIgreen": "pigment",
               "CRI1": "pigment", "CRI2": "pigment"})


def table_indices(X, wl, pri):
    """Zarco-Tejada et al. 2018 Supplementary Table 1 indices that are not in CORE. X: (n, bands) reflectance."""
    cache = {}

    def R(nm):
        if nm not in cache:
            cache[nm] = X[:, np.argmin(np.abs(wl - nm))]
        return cache[nm]

    e = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        e["RDVI"] = (R(800) - R(670)) / np.sqrt(R(800) + R(670))
        e["OSAVI"] = 1.16 * (R(800) - R(670)) / (R(800) + R(670) + 0.16)
        e["MSAVI"] = 0.5 * (2 * R(800) + 1 - np.sqrt((2 * R(800) + 1) ** 2 - 8 * (R(800) - R(670))))
        e["TVI"] = 0.5 * (120 * (R(750) - R(550)) - 200 * (R(670) - R(550)))
        e["MTVI1"] = 1.2 * (1.2 * (R(800) - R(550)) - 2.5 * (R(670) - R(550)))
        den = np.sqrt((2 * R(800) + 1) ** 2 - (6 * R(800) - 5 * np.sqrt(R(670))) - 0.5)
        e["MTVI2"] = 1.5 * (1.2 * (R(800) - R(550)) - 2.5 * (R(670) - R(550))) / den
        e["MCARI"] = ((R(700) - R(670)) - 0.2 * (R(700) - R(550))) * (R(700) / R(670))
        e["MCARI1"] = 1.2 * (2.5 * (R(800) - R(670)) - 1.3 * (R(800) - R(550)))
        e["MCARI2"] = 1.5 * (2.5 * (R(800) - R(670)) - 1.3 * (R(800) - R(550))) / den
        e["SR"] = R(800) / R(670)
        e["MSR"] = (e["SR"] - 1) / (np.sqrt(e["SR"]) + 1)
        e["EVI"] = 2.5 * (R(800) - R(670)) / (R(800) + 6 * R(670) - 7.5 * R(500) + 1)
        e["VOG1"] = R(740) / R(720)
        e["VOG2"] = (R(734) - R(747)) / (R(715) + R(726))
        e["VOG3"] = (R(734) - R(747)) / (R(715) + R(720))
        e["GM1"] = R(750) / R(550)
        e["GM2"] = R(750) / R(700)
        e["TCARI"] = 3 * ((R(700) - R(670)) - 0.2 * (R(700) - R(550)) * (R(700) / R(670)))
        e["TCARI_OSAVI"] = e["TCARI"] / e["OSAVI"]
        e["CIre"] = R(750) / R(710)
        e["SRPI"] = R(430) / R(680)
        e["NPQI"] = (R(415) - R(435)) / (R(415) + R(435))
        e["NPCI"] = (R(680) - R(430)) / (R(680) + R(430))
        e["CTRI1"] = R(695) / R(420)
        e["CAR"] = R(695) / R(760)
        e["DCabCxc"] = R(672) / (R(550) * 3 * R(708))
        e["DNIRCabCxc"] = R(860) / (R(550) * R(708))
        e["SIPI"] = (R(800) - R(445)) / (R(800) + R(680))
        e["CRI550_515"] = 1 / R(515) - 1 / R(550)
        e["CRI700_515"] = 1 / R(515) - 1 / R(700)
        e["RNIR_CRI550"] = (1 / R(510) - 1 / R(550)) * R(770)
        e["RNIR_CRI700"] = (1 / R(510) - 1 / R(700)) * R(770)
        e["PSSRa"] = R(800) / R(675)
        e["PSSRb"] = R(800) / R(650)
        e["PSSRc"] = R(800) / R(500)
        e["PSNDc"] = (R(800) - R(470)) / (R(800) + R(470))
        e["PRI515"] = (R(515) - R(531)) / (R(515) + R(531))
        e["PRIm1"] = (R(512) - R(531)) / (R(512) + R(531))
        e["PRIm2"] = (R(600) - R(531)) / (R(600) + R(531))
        e["PRIm3"] = (R(670) - R(531)) / (R(670) + R(531))
        e["PRIm4"] = (R(570) - R(531) - R(670)) / (R(570) + R(531) + R(670))
        e["PRIn"] = pri / (e["RDVI"] * (R(700) / R(670)))
        e["PRI_CI"] = (R(570) - R(530)) / (R(570) + R(530)) * (R(760) / R(700) - 1)
        e["RI"] = R(700) / R(670)
        e["GI"] = R(570) / R(670)
        e["BI"] = R(450) / R(490)
        e["BGI1"] = R(400) / R(550)
        e["BGI2"] = R(450) / R(550)
        e["BRI1"] = R(400) / R(690)
        e["BRI2"] = R(450) / R(690)
        for k, nm in enumerate((410, 420, 430, 440, 450), 1):
            e[f"BF{k}"] = R(400) / R(nm)
        e["RGI"] = R(690) / R(550)
        e["RARS"] = R(746) / R(513)
        e["LIC1"] = (R(800) - R(680)) / (R(800) + R(680))
        e["LIC2"] = R(440) / R(690)
        e["LIC3"] = R(440) / R(740)
        e["CUR"] = R(675) * R(690) / R(683) ** 2
        e["HI"] = (R(534) - R(698)) / (R(534) + R(698)) - 0.5 * R(704)
    return pd.DataFrame({k: e[k] for k in EXTRA})


def compute(spec_csv, sum_csv):
    sp = pd.read_csv(spec_csv)
    meta, X = sp.iloc[:, :5], sp.iloc[:, 5:].astype(float)
    wl = X.columns.astype(float).values
    W = lambda nm: X.values[:, np.abs(wl - nm) <= HALF].mean(1)
    N = lambda nm: X.values[:, np.argmin(np.abs(wl - nm))]
    r = {nm: N(nm) for nm in (510, 550, 670, 700, 800)}
    w = {nm: W(nm) for nm in (500, 531, 570, 680, 750)}
    d = meta.copy()
    d["NDVI"] = (r[800] - r[670]) / (r[800] + r[670])
    d["PRI"] = (w[570] - w[531]) / (w[570] + w[531])  # table form (PRI570); opposite sign to HyperBird's PRI
    d["PSRI"] = (w[680] - w[500]) / w[750]
    d["ARI"] = 1 / r[550] - 1 / r[700]
    d["CIgreen"] = r[800] / r[550] - 1
    d["CRI1"] = 1 / r[510] - 1 / r[550]
    d["CRI2"] = 1 / r[510] - 1 / r[700]
    lab = to_lab(np.nan_to_num(X.values), wl)
    d["L*"], d["a*"], d["b*"] = lab[:, 0], lab[:, 1], lab[:, 2]
    ex = table_indices(X.values, wl, d["PRI"].values)
    ex.index = d.index
    d = pd.concat([d, ex], axis=1)
    s = pd.read_csv(sum_csv)
    keep = [c for c in ("folder", "name", "low_tissue", "lesion_frac", "tissue_px", "folded", "circle_fill", "frac_offangle") if c in s]
    d = d.merge(s[keep], on=["folder", "name"])
    d = annotate(d[~d.low_tissue].drop(columns="low_tissue"))
    return d[[c for c in d.columns if c not in EXTRA] + EXTRA]  # table indices last, so older column positions hold

