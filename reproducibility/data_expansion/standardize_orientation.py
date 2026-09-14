#!/usr/bin/env python3
"""
Materialize orientation-standardized v2 head volumes.

Every native head is mapped into the canonical anatomical frame
    +X = right, +Y = anterior (forward), +Z = superior
by the per-head proper rotation in v2_manifest (atlas = identity; scb = 180 deg
about the A-P axis, because scb is stored upside-down vs atlas). Standardized
volumes are written as float32 under the SAME key/naming so the existing pmcx
and pyramid-extraction loaders work unchanged by just pointing them at DATASET_V2.

Usage:
    python standardize_orientation.py --verify     # before/after montage only
    python standardize_orientation.py --run        # write all 18 standardized .mat
"""
import os
import sys
import argparse
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2_manifest as M


def load_native(head):
    """Load native (X,Y,Z,4) property volume (axis-order corrected) as float32."""
    mat = sio.loadmat(M.src_mat(head))
    key = M.PROP_KEY if M.PROP_KEY in mat else [k for k in mat if not k.startswith("_")][0]
    v = mat[key].astype(np.float32)
    if v.ndim == 4 and v.shape[0] == 4 and v.shape[-1] != 4:
        v = np.moveaxis(v, 0, -1)
    return v


def standardize_volume(head):
    """Native -> canonical (X,Y,Z,4) via the head's proper-rotation transform."""
    v = load_native(head)
    t = M.transform_for(head)
    return t.apply_volume(v)        # (X,Y,Z,4) preserved; spatial axes rotated


def materialize_all(overwrite=False):
    os.makedirs(M.DATASET_V2, exist_ok=True)
    for head in M.HEADS:
        out = M.std_mat(head)
        if os.path.isfile(out) and not overwrite:
            print(f"[skip] {head} (exists)")
            continue
        v = standardize_volume(head)
        sio.savemat(out, {M.PROP_KEY: v.astype(np.float32)}, do_compression=True)
        print(f"[write] {head:14s} -> {os.path.basename(out)}  shape={v.shape}")


# ---------------------------------------------------------------------------
def _mask(v):
    import scipy.ndimage as ndi
    m = v[..., 3] > 1.05
    lab, nc = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, nc + 1))
    return ndi.binary_fill_holes(lab == int(np.argmax(sizes)) + 1)


def _crop(ax, mk2d, margin=4):
    """Tighten the axes to the mask bbox (in display orientation) + a small margin."""
    rr, cc = np.where(mk2d)
    if rr.size == 0:
        return
    ny, nx = mk2d.shape
    ax.set_xlim(max(cc.min() - margin, 0), min(cc.max() + margin, nx - 1))
    ax.set_ylim(max(rr.min() - margin, 0), min(rr.max() + margin, ny - 1))


def verify(heads=("scb01", "scb11", "bw01", "bw19"), out="orient_standardize_check.png",
           right_titles=True, crop_margin=4):
    """Render native vs standardized sagittal+coronal to confirm consistency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(M.VIZ_V2, exist_ok=True)
    fig, axes = plt.subplots(len(heads), 4, figsize=(13, 3.2 * len(heads)))
    for r, head in enumerate(heads):
        nat = load_native(head)
        std = standardize_volume(head)
        for c, (v, tag) in enumerate([(nat, "native"), (std, "standardized")]):
            mua = v[..., 0]; mk = _mask(v); cen = np.argwhere(mk).mean(0).astype(int)
            disp = np.where(mk, mua, np.nan)
            show_title = right_titles or c == 0
            # sagittal (X mid): Y horiz, Z vert
            axes[r, 2*c].imshow(disp[cen[0], :, :].T, origin="lower", cmap="bone", vmin=0, vmax=0.05)
            _crop(axes[r, 2*c], mk[cen[0], :, :].T, crop_margin)
            if show_title:
                axes[r, 2*c].set_title(f"{head} {tag}\nsagittal (Y->, Z up)", fontsize=8)
            # coronal (Y mid): X horiz, Z vert
            axes[r, 2*c+1].imshow(disp[:, cen[1], :].T, origin="lower", cmap="bone", vmin=0, vmax=0.05)
            _crop(axes[r, 2*c+1], mk[:, cen[1], :].T, crop_margin)
            if show_title:
                axes[r, 2*c+1].set_title(f"{head} {tag}\ncoronal (X->, Z up)", fontsize=8)
            for k in (2*c, 2*c+1):
                axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
    fig.tight_layout()
    fp = os.path.join(M.VIZ_V2, out)
    fig.savefig(fp, dpi=115); print("wrote", fp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--heads", default=None, help="comma-sep head list")
    ap.add_argument("--out", default="orient_standardize_check.png")
    ap.add_argument("--no-right-titles", action="store_true")
    ap.add_argument("--crop-margin", type=int, default=4)
    a = ap.parse_args()
    if a.verify:
        verify(heads=tuple(a.heads.split(",")) if a.heads else
               ("scb01", "scb11", "bw01", "bw19"), out=a.out,
               right_titles=not a.no_right_titles, crop_margin=a.crop_margin)
    if a.run:
        materialize_all(overwrite=a.overwrite)
    if not (a.verify or a.run):
        ap.print_help()
