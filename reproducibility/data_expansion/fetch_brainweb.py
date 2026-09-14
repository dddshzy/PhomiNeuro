#!/usr/bin/env python3
"""
BrainWeb-20 fetch + 1 mm loader + orientation/label inspection (Stage 0 gate).

BrainWeb crisp anatomical models are 362x434x362 @ 0.5 mm (FOV 181x217x181 mm),
uint16 with 12 tissue intensity codes that match the .m tissueratio12 EXACTLY:
  0=bg 16=CSF 32=GM 48=WM 64=fat 80=muscle 96=skin 112=skull 128=vessel
  145=around-fat 161=dura 177=marrow
We DOWNSAMPLE 2x -> 1 mm to match the scb heads (optical units are mm^-1).

  python fetch_brainweb.py --inspect    # render bw04 vs scb01 orthogonal slices
"""
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import repro_config as RC

CACHE = RC.BRAINWEB_CACHE
# 12 BrainWeb intensity codes (confirmed empirically, all 20 subjects share them)
BW_CODES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 145, 161, 177]
BW_NAMES = ["bg", "CSF", "GM", "WM", "fat", "muscle", "skin", "skull",
            "vessel", "around-fat", "dura", "marrow"]


def subject_files():
    """Cached list of the 20 subject .bin.gz paths (downloads on first call)."""
    import brainweb
    os.makedirs(CACHE, exist_ok=True)
    return sorted(brainweb.get_files(cache_dir=CACHE, progress=False))


def bw_head_ids():
    return [f"bw{i:02d}" for i in range(1, len(subject_files()) + 1)]


def load_label_1mm(path):
    """Load a subject's crisp model and downsample 0.5 mm -> 1 mm (181,217,181)."""
    import brainweb
    v = brainweb.load_file(path)            # (362,434,362) uint16 @ 0.5 mm
    return np.ascontiguousarray(v[::2, ::2, ::2])


def inspect():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import scipy.io as sio
    files = subject_files()
    bw = load_label_1mm(files[0])           # subject_04 -> bw01
    print(f"bw01 (subject_04) 1mm shape={bw.shape}")
    occ = bw > 0; bb = np.argwhere(occ)
    print(f"  occupancy extents (mm) per array axis: {(bb.max(0)-bb.min(0)+1).tolist()}")
    u = np.unique(bw)
    assert set(u.tolist()) <= set(BW_CODES), f"unexpected labels {u}"
    print(f"  labels present: {u.tolist()}  (all known)")

    scb = sio.loadmat(os.path.join(os.path.dirname(HERE),
          "dataset_v2_head810", "scb01_copmri_withHermiteF810.mat"))["vol_prop_eye_aseg"]
    scb_mua = scb[..., 0]; scb_m = scb[..., 3] > 1.05

    fig, ax = plt.subplots(2, 3, figsize=(13, 9))
    # bw01: mid-slice perpendicular to each array axis (show raw labels)
    cen = bb.mean(0).astype(int)
    bwd = np.where(occ, bw, np.nan)
    bw_views = [(bwd[cen[0], :, :], "bw axis0 mid  (shows axis1 H, axis2 V)"),
                (bwd[:, cen[1], :], "bw axis1 mid  (shows axis0 H, axis2 V)"),
                (bwd[:, :, cen[2]], "bw axis2 mid  (shows axis0 H, axis1 V)")]
    for k, (img, t) in enumerate(bw_views):
        ax[0, k].imshow(img.T, origin="lower", cmap="nipy_spectral")
        ax[0, k].set_title(t, fontsize=8); ax[0, k].set_xlabel("H"); ax[0, k].set_ylabel("V")
    # scb01 canonical (KNOWN: axis0=X L-R, axis1=Y A-P, axis2=Z S-I)
    cs = np.argwhere(scb_m).mean(0).astype(int); sd = np.where(scb_m, scb_mua, np.nan)
    scb_views = [(sd[cs[0], :, :], "scb01 sagittal  (Y A-P H, Z S-I V)"),
                 (sd[:, cs[1], :], "scb01 coronal   (X L-R H, Z S-I V)"),
                 (sd[:, :, cs[2]], "scb01 axial     (X L-R H, Y A-P V)")]
    for k, (img, t) in enumerate(scb_views):
        ax[1, k].imshow(img.T, origin="lower", cmap="bone", vmin=0, vmax=0.05)
        ax[1, k].set_title(t, fontsize=8); ax[1, k].set_xlabel("H"); ax[1, k].set_ylabel("V")
    fig.suptitle("Top: BrainWeb bw01 native axes  |  Bottom: scb01 canonical (+X right,+Y ant,+Z sup)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(HERE), "viz", "out", "v3", "bw_orient_inspect.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    if a.inspect:
        inspect()
    else:
        print("subjects:", len(subject_files()))
