"""Stage 4 (pixel route): mean spectrum per disk from the raw cubes, with its own calibration and disk outline.

Calibration: reflectance = (raw - black) / (white - black); white averaged per (sample, band) inside the round white
target, black averaged over lines without light leak. Masks, on a BIN x BIN binned NDVI map, wherever the disk sits:
  green  = largest NDVI > 0.4 component (+ nearby pieces split off by streaks), closed across pale veins, holes filled
  disk   = RANSAC circle fitted to the green outline, so lesions at the disk edge are included
  lesion = disk pixels with NDVI < median green NDVI - LESION_DELTA; pale veins removed by opening,
           thin streak / edge-rim pieces dropped
Outputs in OUT: spectra_mean.csv and spectra_std.csv (all disk pixels, 400-900 nm), spectra_mean_green.csv (disk
minus lesion), summary.csv, cubes/<scan>.json (one per cube), masks/<scan>.npz, previews/<scan>.png (true colour,
disk outline yellow, lesions magenta), refs/ (cached reference averages).
Resumable: a cube whose JSON exists is skipped. The combined tables are rewritten at the end of every run.

Usage: python 04_pixel_spectra.py DATA_ROOT results/inventory.csv results/pixel [--only-first] [--only "folder/name;..."]
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import CircleModel, ransac, regionprops, label as sk_label
from spectral.io import envi

from cube import CHUNK, BIN, SAT, nearest, ref_stats, white_target_mask, black_good_lines
from colour import to_srgb

WL_MIN, WL_MAX, NDVI_THR = 400.0, 900.0, 0.4
# mask parameters, in binned pixels (1 binned px = BIN x BIN full-res px)
CLOSE_B = 4          # closing across pale veins (~30 px full-res)
JOIN_B = 15          # join green pieces this close to the main piece (streak-split disks)
MIN_GREEN_B = 2000   # below this there is no disk (empty well, failed scan)
BG_NDVI = 0.15       # dish background inside the fitted circle
LESION_DELTA = 0.12  # lesion = NDVI below (median green NDVI - this)
LESION_OPEN_B = 5    # opening that removes pale veins from the lesion mask
LESION_MIN_B = 150   # smallest lesion kept (~2400 full-res px)
STREAK_MINOR = 0.09  # lesion pieces thinner than this x radius are streaks/edge rims, not lesions

# folder -> (white, black), relative to the drive root: from design.json
from design import REFS  # noqa: E402


def open_mm(hdr, raw=None):
    img = envi.open(str(hdr), image=str(raw) if raw else None)
    wl = np.array([float(w) for w in img.metadata["wavelength"]])
    return img.open_memmap(interleave="bip", writable=False), wl


def load_refs(root, folder, cache_dir, log):
    cache = cache_dir / (folder.replace("/", "__") + ".npz")
    if cache.exists():
        z = np.load(cache)
        return z["W"], z["K"], z["wmask"]
    wh, bl = (root / p for p in REFS[folder])
    white, wl = open_mm(wh)
    black, _ = open_mm(bl)
    log(f"[refs] {folder}: white target")
    wmask = white_target_mask(white, wl)
    log(f"[refs] {folder}: white mean")
    W, _, cnt = ref_stats(white, pix_ok=wmask)
    kgood = black_good_lines(black)
    log(f"[refs] {folder}: black mean ({(~kgood).sum()} leak lines dropped)")
    K, _, _ = ref_stats(black, line_ok=kgood)
    np.savez_compressed(cache, W=W, K=K, wmask=wmask, kgood=kgood)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].imshow(wmask, cmap="gray"); ax[0].set_title(f"{folder}: white target ({wmask.sum()} px, {(cnt > 0).sum()} cols)")
    ax[1].plot(wl, np.nanmean(W, 0), label="white"); ax[1].plot(wl, K.mean(0), label="black")
    ax[1].set(xlabel="nm", ylabel="counts"); ax[1].legend()
    fig.tight_layout(); fig.savefig(cache.with_suffix(".png"), dpi=80); plt.close(fig)
    return W, K, wmask


def leaf_masks(nb):
    """Masks on the binned (BIN x BIN) NDVI map. Returns green, disk, lesion (bool, binned) and circle info.

    green  : NDVI > NDVI_THR, closed across pale veins, plus pieces cut off by streaks that sit right
             next to the main piece (neighbouring wells are further away), holes filled
    disk   : circle fitted (RANSAC) to the green outline -- leaf disks are punched circles -- so lesions on
             the edge are inside; dish background (NDVI <= BG_NDVI) inside the circle is excluded
    lesion : disk pixels with NDVI < (median green NDVI - LESION_DELTA), opened to drop pale veins, minus
             thin pieces (minor axis < STREAK_MINOR * radius: white streaks/threads, cut-edge rims)
    """
    empty = np.zeros_like(nb, bool)
    g = ndimage.binary_opening(np.nan_to_num(nb) > NDVI_THR, iterations=1)
    g = ndimage.binary_closing(np.pad(g, CLOSE_B + 1), iterations=CLOSE_B)[CLOSE_B + 1:-CLOSE_B - 1, CLOSE_B + 1:-CLOSE_B - 1]
    lab, n = ndimage.label(g)
    if not n:
        return empty, empty, empty, None
    sizes = ndimage.sum(g, lab, range(1, n + 1))
    main = lab == 1 + np.argmax(sizes)
    near = np.unique(lab[ndimage.binary_dilation(main, iterations=JOIN_B) & g])
    keep = [k for k in near if k > 0 and sizes[k - 1] >= 0.005 * sizes.max()]
    green = ndimage.binary_fill_holes(np.isin(lab, keep))
    if green.sum() < MIN_GREEN_B:  # empty well / failed scan: no disk to fit
        return green, green, empty, None
    edge = green & ~ndimage.binary_erosion(green)
    pts = np.argwhere(edge)[:, ::-1].astype(float)
    H, Wd = nb.shape
    pts = pts[(pts[:, 0] > 0) & (pts[:, 0] < Wd - 1) & (pts[:, 1] > 0) & (pts[:, 1] < H - 1)]  # not the frame edge
    model, inl = ransac(pts, CircleModel, min_samples=3, residual_threshold=3, max_trials=500, rng=0)
    if model is None:
        return green, green, empty, None
    (xc, yc), r = model.center, model.radius
    yy, xx = np.mgrid[:H, :Wd]
    disk = ((xx - xc) ** 2 + (yy - yc) ** 2 <= (r - 1) ** 2) & np.isfinite(nb) & (nb > BG_NDVI)
    disk = ndimage.binary_opening(disk | green, iterations=2)
    med = np.nanmedian(nb[green])
    les = ndimage.binary_opening(disk & (nb < med - LESION_DELTA), iterations=LESION_OPEN_B)
    L = sk_label(les)
    lesion = np.zeros_like(les)
    for p in regionprops(L):
        if p.area >= LESION_MIN_B and p.axis_minor_length >= STREAK_MINOR * r:
            lesion[L == p.label] = True
    circ = (xx - xc) ** 2 + (yy - yc) ** 2 <= r ** 2  # clipped to the frame automatically
    circle = dict(xc=float(xc * BIN), yc=float(yc * BIN), r=float(r * BIN), inliers=float(inl.mean()),
                  green_median_ndvi=float(med), fill=float((disk & circ).sum() / max(circ.sum(), 1)))
    return green, disk, lesion, circle


def process(raw_mm, wl, W, K):
    L, S, B = raw_mm.shape
    denom = np.maximum(W - K, 1.0)
    ib = {k: nearest(wl, nm) for k, nm in dict(R=650, G=560, B=460, NIR=800, RED=670).items()}
    full = {k: np.empty((L, S), np.float32) for k in ib}
    Lb, Sb = L // BIN, S // BIN
    binned = np.zeros((Lb, Sb, B), np.float32)
    sat = 0
    for l0 in range(0, L, CHUNK):
        blk = np.asarray(raw_mm[l0:l0 + CHUNK])
        sat += int((blk >= SAT).sum())
        refl = (blk.astype(np.float32) - K) / denom
        for k, i in ib.items():
            full[k][l0:l0 + len(blk)] = refl[..., i]
        n = len(blk) // BIN * BIN
        binned[l0 // BIN:l0 // BIN + n // BIN] = refl[:n, :Sb * BIN].reshape(n // BIN, BIN, Sb, BIN, B).mean((1, 3))

    ndvi = (full["NIR"] - full["RED"]) / np.maximum(full["NIR"] + full["RED"], 1e-6)
    ndvi_bin = ndvi[:Lb * BIN, :Sb * BIN].reshape(Lb, BIN, Sb, BIN).mean((1, 3))
    green_b, disk_b, les_b, circle = leaf_masks(ndvi_bin)
    px, px_h = binned[disk_b], binned[disk_b & ~les_b]
    nan = np.full(B, np.nan)
    up = lambda m: np.pad(np.repeat(np.repeat(m, BIN, 0), BIN, 1), ((0, L - Lb * BIN), (0, S - Sb * BIN)))
    vis = wl <= 780
    rgb_b = to_srgb(np.nan_to_num(binned[..., vis]), wl[vis]).astype(np.float32)  # true colour, binned
    rgb = np.repeat(np.repeat(rgb_b, BIN, 0), BIN, 1)
    rgb = np.pad(rgb, ((0, L - rgb.shape[0]), (0, S - rgb.shape[1]), (0, 0)))
    return dict(ndvi=ndvi, ndvi_bin=ndvi_bin, mask=up(disk_b), green=up(green_b), lesion=up(les_b),
                masks_bin=(green_b, disk_b, les_b), circle=circle, rgb=rgb, sat=sat,
                mean=np.nanmean(px, 0) if len(px) else nan, std=np.nanstd(px, 0) if len(px) else nan,
                mean_green=np.nanmean(px_h, 0) if len(px_h) else nan)


def save_preview(path, rgb, mask, title, lesion):
    """True-colour sRGB in [0, 1], not stretched, so colours are comparable between cubes."""
    s = slice(None, None, 2)
    m, le = mask[s, s], lesion[s, s]
    img = np.nan_to_num(np.clip(rgb[s, s], 0, 1)).copy()
    img[m & ~ndimage.binary_erosion(m, iterations=2)] = (1, 1, 0)
    img[le & ~ndimage.binary_erosion(le, iterations=2)] = (1, 0, 1)
    fig = plt.figure(figsize=(5.5, 5.2)); plt.imshow(img); plt.title(title, fontsize=9); plt.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)



def main(a):
    root, out = Path(a.root), Path(a.out)
    for d in ("cubes", "previews", "refs", "masks"):
        (out / d).mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    inv = pd.read_csv(a.inventory)
    inv = inv[~inv.name.str.contains("white|black", case=False) & inv.has_raw & inv.folder.isin(REFS)]
    skipped = inv[inv.has_hdr & ~inv.size_ok.astype(bool)]
    for _, r in skipped.iterrows():
        log(f"[skip] {r.folder}/{r['name']}: raw is {int(r.raw_bytes)} bytes, expected {int(r.expected_bytes)}")
    todo = inv[~inv.index.isin(skipped.index)].copy()
    todo["order"] = ~todo.index.isin(todo.groupby("folder").head(1).index)  # first cube of each folder first
    todo = todo.sort_values(["order", "folder", "name"])
    if a.only_first:
        todo = todo[~todo.order]
    if a.only:
        want = {x.strip() for x in a.only.split(";")}
        todo = todo[(todo.folder + "/" + todo.name).isin(want)]
    log(f"{len(todo)} cubes to process ({len(skipped)} skipped as truncated)")

    refs = {}
    t_start, done = time.time(), 0
    for _, r in todo.iterrows():
        key = f"{r.folder.replace('/', '__')}__{r['name']}"
        jpath = out / "cubes" / f"{key}.json"
        if jpath.exists():
            continue
        try:
            if r.folder not in refs:
                refs[r.folder] = load_refs(root, r.folder, out / "refs", log)
            W, K, wmask = refs[r.folder]
            raw = root / r.folder / f"{r['name']}.raw"
            hdr = root / r.folder / f"{r['name']}.hdr"
            borrowed = not r.has_hdr
            if borrowed:  # identical acquisition settings across the drive; borrow the folder's white header
                hdr = root / REFS[r.folder][0]
            t0 = time.time()
            raw_mm, wl = open_mm(hdr, raw)
            res = process(raw_mm, wl, W, K)
            keep = (wl >= WL_MIN) & (wl <= WL_MAX)
            m = res["mask"]
            ys, xs = np.nonzero(m)
            cov = np.nonzero(~np.isnan(W[:, 0]))[0]
            rec = dict(folder=r.folder, name=r["name"], group=r.get("group"), cond=r.get("cond"),
                       rep=None if pd.isna(r.get("rep")) else str(int(r.get("rep"))),
                       header_borrowed=borrowed, tissue_px=int(m.sum()), green_px=int(res["green"].sum()),
                       lesion_px=int(res["lesion"].sum()),
                       lesion_frac=round(float(res["lesion"].sum() / max(m.sum(), 1)), 4),
                       circle=res["circle"],
                       centroid_line=float(ys.mean()) if len(ys) else None,
                       centroid_sample=float(xs.mean()) if len(xs) else None,
                       extent=[int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())] if len(ys) else None,
                       touches_frame=bool(len(ys) and (ys.min() < 5 or ys.max() > m.shape[0] - 6)),
                       near_white_edge=bool(len(xs) and (xs.min() < cov.min() + 10 or xs.max() > cov.max() - 10)),
                       ndvi_median=float(np.nanmedian(res["ndvi"][m])) if m.any() else None,
                       saturated_px=res["sat"],
                       wavelength=wl[keep].round(3).tolist(),
                       mean=res["mean"][keep].round(5).tolist(), std=res["std"][keep].round(5).tolist(),
                       mean_green=res["mean_green"][keep].round(5).tolist(),
                       seconds=round(time.time() - t0, 1))
            gb, dbm, lb = res["masks_bin"]  # binned resolution; upsample x BIN for full-res
            np.savez_compressed(out / "masks" / f"{key}.npz", bin=BIN, shape_bin=np.array(dbm.shape),
                                disk=np.packbits(dbm), lesion=np.packbits(lb), green=np.packbits(gb),
                                ndvi_bin=res["ndvi_bin"].astype(np.float16))
            save_preview(out / "previews" / f"{key}.png", res["rgb"], m,
                         f"{r.folder}/{r['name']}  NDVI~{rec['ndvi_median'] or float('nan'):.2f}  lesion {100 * rec['lesion_frac']:.1f}%",
                         res["lesion"])
            jpath.write_text(json.dumps(rec))
            done += 1
            el = time.time() - t_start
            log(f"[ok] {r.folder}/{r['name']}  tissue={rec['tissue_px']}  NDVI={rec['ndvi_median'] or float('nan'):.3f}"
                f"  lesion={100 * rec['lesion_frac']:.1f}%  {rec['seconds']}s  ({done} done, ~{el / done * (len(todo) - done) / 3600:.1f} h left)")
        except Exception as e:
            log(f"[error] {r.folder}/{r['name']}: {e}")
            (out / "cubes" / f"{key}.error.txt").write_text(traceback.format_exc())

    collect(out, log)


def collect(out, log):
    recs = [json.loads(p.read_text()) for p in sorted((out / "cubes").glob("*.json")) if not p.name.startswith("._")]
    if not recs:
        return
    wl = recs[0]["wavelength"]
    kinds = ["mean", "std", "mean_green"]
    meta_cols = [k for k in recs[0] if k not in ("wavelength", "extent", "circle", *kinds)]
    meta = pd.DataFrame([{k: r[k] for k in meta_cols} for r in recs])
    for kind in kinds:
        df = pd.concat([meta[["folder", "name", "group", "cond", "rep"]],
                        pd.DataFrame([r[kind] for r in recs], columns=[f"{w:.2f}" for w in wl])], axis=1)
        df.to_csv(out / f"spectra_{kind}.csv", index=False)
    meta["low_tissue"] = meta.green_px < 0.5 * meta.groupby("folder").green_px.transform("median")  # empty well / failed scan
    meta.to_csv(out / "summary.csv", index=False)
    log(f"[collect] {len(recs)} cubes -> " + ", ".join(f"spectra_{k}.csv" for k in kinds) + ", summary.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root"); p.add_argument("inventory"); p.add_argument("out")
    p.add_argument("--only-first", action="store_true", help="just the first cube of each folder (QC)")
    p.add_argument("--only", help='semicolon-separated "folder/name" list')
    main(p.parse_args())
