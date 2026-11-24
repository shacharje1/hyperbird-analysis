"""Reading ENVI cubes and averaging the white and black references (used by 04_pixel_spectra.py)."""
import numpy as np
from scipy import ndimage

CHUNK = 40      # scan lines per read (~170 MB raw)
BIN = 4         # spatial binning for the stored reduced cube
SAT = 4095      # 12-bit camera


def open_cube(hdr):
    img = envi.open(str(hdr))
    wl = np.array([float(w) for w in img.metadata["wavelength"]])
    return img.open_memmap(interleave="bip", writable=False), wl


def ref_stats(mm, pix_ok=None, line_ok=None):
    """Per-(sample, band) mean over the valid (line, sample) pixels, plus the per-line mean profile.

    pix_ok: (L, S) bool -- e.g. inside the round white target. line_ok: (L,) bool -- e.g. no light leak.
    """
    L, S, B = mm.shape
    acc = np.zeros((S, B), np.float64)
    cnt = np.zeros(S, np.int64)
    line_mean = np.empty(L, np.float64)
    for l0 in range(0, L, CHUNK):
        blk = np.asarray(mm[l0:l0 + CHUNK], dtype=np.float32)
        line_mean[l0:l0 + len(blk)] = blk.mean((1, 2))
        ok = np.ones(blk.shape[:2], bool)
        if pix_ok is not None:
            ok &= pix_ok[l0:l0 + len(blk)]
        if line_ok is not None:
            ok &= line_ok[l0:l0 + len(blk), None]
        acc += (blk * ok[..., None]).sum(0)
        cnt += ok.sum(0)
    with np.errstate(invalid="ignore"):
        return (acc / cnt[:, None]).astype(np.float32), line_mean, cnt


def white_target_mask(mm, wl, frac=0.5):
    """(L, S) mask of the bright round white target, from one band near 650 nm."""
    b = nearest(wl, 650)
    band = np.stack([mm[l, :, b] for l in range(mm.shape[0])]).astype(np.float32)
    m = band > frac * np.percentile(band, 99)
    m = ndimage.binary_fill_holes(ndimage.binary_opening(m, iterations=3))  # fill dead-column stripes
    m = ndimage.binary_erosion(m, iterations=15)  # stay off the rim
    lab, n = ndimage.label(m)
    return lab == (1 + np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) if n else m


def black_good_lines(mm, factor=1.5):
    """Drop scan lines with light leaking into the dark reference."""
    lm = np.array([np.asarray(mm[l, ::8, ::8]).mean() for l in range(mm.shape[0])])
    bad = lm > factor * np.median(lm)
    return ~ndimage.binary_dilation(bad, iterations=5)


def nearest(wl, nm):
    return int(np.argmin(np.abs(wl - nm)))

