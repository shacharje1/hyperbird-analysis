"""Reflectance spectra -> CIE XYZ, CIELAB, sRGB (what the eye sees), same maths as the HyperBird system's
hsi2color_zarr.py: CIE 1931 2-degree colour-matching functions and a CIE D illuminant (default D65) from
reference/D_illuminants.mat (the HyperBird engineer's file), PCHIP-interpolated to the camera bands,
trapezoid integration over <= 780 nm, normalised so a perfect white reflector has Y = 1 (L* = 100).

Also CIEDE2000 colour difference (dE00). Rough guide: dE00 < 1 not perceptible, 1-2 perceptible on close
side-by-side inspection, 2-10 noticeable at a glance.
"""
from pathlib import Path

import numpy as np
import scipy.io as spio
from scipy.interpolate import PchipInterpolator

import os

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [os.environ.get("HB_ILLUMINANTS"), _HERE / "D_illuminants.mat", _HERE.parent / "reference" / "D_illuminants.mat",
               Path.home() / ".claude/skills/hyperbird-image-analysis/reference/D_illuminants.mat"]
MAT = next((Path(p) for p in _CANDIDATES if p and Path(p).is_file()), None)
if MAT is None:
    raise FileNotFoundError("D_illuminants.mat not found: copy it next to colour.py or set $HB_ILLUMINANTS")
_COL = {65: 1, 50: 2, 55: 3, 75: 4}  # column of each D illuminant in the .mat (as in hsi2color_zarr.py)
_M = np.array([[3.2404542, -1.5371385, -0.4985314],
               [-0.9692660, 1.8760108, 0.0415560],
               [0.0556434, -0.2040259, 1.0572252]])


def weights(wl, illuminant=65, truncate_nm=780.0, mat=MAT):
    """Per-band integration weights (B, 3) for X, Y, Z (zero above truncate_nm), normalised to Y_white = 1."""
    wl = np.asarray(wl, float)
    m = spio.loadmat(str(mat))
    cmf, D = m["wxyz"], m["D"]
    I = PchipInterpolator(D[:, 0], D[:, _COL[illuminant]], extrapolate=True)(wl)
    xyz = np.stack([PchipInterpolator(cmf[:, 0], cmf[:, k], extrapolate=True)(wl) for k in (1, 2, 3)], 1)
    keep = wl <= truncate_nm
    # trapezoid weights for non-uniform band spacing
    tw = np.zeros_like(wl)
    w_ = wl[keep]
    d = np.diff(w_)
    t = np.zeros(len(w_)); t[:-1] += d / 2; t[1:] += d / 2
    tw[keep] = t
    W = (I * tw)[:, None] * xyz
    return W / W[:, 1].sum()


def to_xyz(R, wl, **kw):
    """R (..., B) reflectance -> XYZ (..., 3)."""
    return np.asarray(R, float) @ weights(wl, **kw)


def xyz_to_lab(XYZ, white=(1.0, 1.0, 1.0)):
    t = np.asarray(XYZ, float) / np.asarray(white, float)
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def white_xyz(wl, **kw):
    return weights(wl, **kw).sum(0)


def to_lab(R, wl, **kw):
    """R (..., B) reflectance -> CIELAB (..., 3), reference white = perfect reflector under the illuminant."""
    return xyz_to_lab(to_xyz(R, wl, **kw), white_xyz(wl, **kw))


def to_srgb(R, wl, exposure=1.0, **kw):
    """R (..., B) -> display sRGB in [0, 1] (gamma-encoded, clipped)."""
    lin = to_xyz(R, wl, **kw) @ _M.T * exposure
    s = np.where(lin > 0.0031308, 1.055 * np.power(np.maximum(lin, 0), 1 / 2.4) - 0.055, 12.92 * lin)
    return np.clip(s, 0, 1)


def delta_e00(lab1, lab2):
    from skimage.color import deltaE_ciede2000
    return deltaE_ciede2000(np.asarray(lab1, float), np.asarray(lab2, float))


if __name__ == "__main__":  # self-check
    wl = np.arange(395, 1000, 0.63)
    print("perfect white Lab:", to_lab(np.ones_like(wl), wl).round(3), "sRGB:", to_srgb(np.ones_like(wl), wl).round(3))
    print("50% grey Lab:", to_lab(np.full_like(wl, .5), wl).round(2))
