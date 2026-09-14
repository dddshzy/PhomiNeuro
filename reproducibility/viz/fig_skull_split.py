#!/usr/bin/env python3
"""Split the phantom's single SKULL row back into cancellous (diploe) and cortical bone.

The simulated phantom carries ONE bone row: GRACE separates cancellous (label 7) from
cortical (label 8), but the optical LUT collapses both into OASIS9 skull -> _P0F row 4,
by explicit design decision. That collapse is irreversible from the phantom alone -- no
amount of inverting optical properties brings it back, because both bone types were
written with identical (mu_a, mu_s).

So the sub-classification is recovered from the source instead: GRACE's raw 11-class
segmentation, which is saved for every head in external/GRACE/seg_out/ and lives on the
SAME iso-1mm grid as the pre-MNI phantom (verified per subject before use). It therefore
rides the identical stored rigid transform into the common grid -- no new registration.

The phantom's own skull mask stays authoritative for WHERE bone is: it survived the
hybrid pipeline's brain-mask overrides and hole filling, so it is what MCX actually saw.
GRACE only answers WHICH KIND of bone inside that mask. Voxels the phantom calls skull
but GRACE does not resolve are drawn as a third colour rather than silently assigned.

  python fig_skull_split.py --subjects oas30208 oas30812 oas30948 oas30926
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import nibabel as nib
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_expansion"))
import repro_config as RC                                          # noqa: E402
import mni_normalize as MN                                        # noqa: E402
from optical_tissue_lut import OASIS9_LABEL_NAME, OASIS9_LABEL_TO_P0ROW   # noqa: E402

GRACE_DIR = os.environ.get(
    "GRACE_SEG_DIR", os.path.join(RC.GRACE_DIR, "seg_out")
)
SRC_DIR = str(RC.OASIS_DATASET_DIR)
MNI_DIR = str(RC.MNI_DATASET_DIR)
CACHE = str(RC.PUP_MNI_DIR)
OUT_DIR = str(RC.RESULTS_DIR / "PET-ROI")
PROP_KEY = "vol_prop_eye_aseg"

P0ROW_NAME = {row: OASIS9_LABEL_NAME[lab] for lab, row in OASIS9_LABEL_TO_P0ROW.items()}
SKULL_ROW = OASIS9_LABEL_TO_P0ROW[2]
GRACE_CANCELLOUS, GRACE_CORTICAL = 7, 8

# display classes: phantom rows, with the skull row replaced by three bone classes
CANC, CORT, UNRES = 20, 21, 22
COL = {"air": "#000000", "WM": "#f5f5f5", "GM": "#9b9b9b", "CSF": "#4da6ff",
       "vessel": "#d7263d", "fat": "#ffd166", "muscle": "#c1666b", "skin": "#f2c9a0"}


def grace_in_mni(sid):
    """GRACE 11-class labels on the common MNI grid, cached next to the PUP volumes."""
    p = os.path.join(CACHE, f"{sid}_grace_mni.npy")
    if os.path.isfile(p):
        return np.load(p)
    import SimpleITK as sitk
    cand = [os.path.join(GRACE_DIR, f"{n}_grace_seg.nii.gz")
            for n in (sid.upper(), sid.lower())]
    src_p = next((c for c in cand if os.path.isfile(c)), None)
    if src_p is None:
        raise SystemExit(f"[skull] no GRACE segmentation for {sid} in {GRACE_DIR}")
    lab = np.asanyarray(nib.load(src_p).dataobj).astype(np.uint8)

    ref_shape = sio.loadmat(os.path.join(
        SRC_DIR, f"{sid}_copmri_withHermiteF810.mat"))[PROP_KEY].shape[:3]
    if lab.shape != ref_shape:
        raise SystemExit(f"[skull] {sid}: GRACE {lab.shape} != pre-MNI phantom "
                         f"{ref_shape}; they must share the iso grid for the stored "
                         "transform to apply")

    T = sitk.ReadTransform(os.path.join(MNI_DIR, f"{sid}_v11mni_rigid.tfm"))
    flip = MN.DATASET_FLIP[MN.dataset_of(sid)]
    src = lab[::-1].copy() if flip else lab
    ref = sitk.Image(int(MN.GDIM[0]), int(MN.GDIM[1]), int(MN.GDIM[2]), sitk.sitkFloat32)
    ref.SetSpacing((1, 1, 1))
    ref.SetOrigin(tuple(float(-x) for x in MN.AC))
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    res = sitk.Resample(MN._to_sitk(src.astype(np.float32), (0, 0, 0)), ref, T,
                        sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)
    out = np.round(sitk.GetArrayFromImage(res).transpose(2, 1, 0)).astype(np.uint8)
    np.save(p, out)
    return out


def split_labels(sid):
    prop = sio.loadmat(os.path.join(MNI_DIR, f"{sid}_v11mni_F810.mat"))[PROP_KEY]
    lab = MN.optical_to_label(prop).astype(np.int16)
    g = grace_in_mni(sid)
    skull = lab == SKULL_ROW
    out = lab.copy()
    out[skull] = UNRES
    out[skull & (g == GRACE_CANCELLOUS)] = CANC
    out[skull & (g == GRACE_CORTICAL)] = CORT
    n = {k: int((out == k).sum()) for k in (CANC, CORT, UNRES)}
    return out, skull, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--dx", type=float, default=-8.0)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    x = int(round(MN.AC[0] + a.dx))

    rows = sorted(P0ROW_NAME) + [CANC, CORT, UNRES]
    names = {**{r: P0ROW_NAME[r] for r in P0ROW_NAME},
             CANC: "cancellous bone (diploe)", CORT: "cortical bone", UNRES: "bone, unresolved"}
    # Cortical bone must not be another cream: at #f7f3e3 it sat one step from WM
    # (#f5f5f5) and the vault zoom -- the panel whose entire job is telling the two bone
    # layers apart -- read as one undifferentiated pale mass. Dense bone dark, spongy
    # diploe light, which also matches how the two look on CT.
    colours = {**COL, "skull": "#e8e3d3",
               "cancellous bone (diploe)": "#a86edb",
               "cortical bone": "#6b5b45", "bone, unresolved": "#2ecc71"}
    idx = {r: i for i, r in enumerate(rows)}
    cmap = ListedColormap([colours[names[r]] for r in rows])

    for sid in a.subjects:
        vol, skull, n = split_labels(sid)
        tot = sum(n.values())
        disp = np.vectorize(idx.get)(vol).astype(np.int16)
        # v[x] is (Y,Z); transpose puts superior up / anterior right under origin="lower"
        S = disp[x, :, :].T
        occ = S != idx[0]
        rr, cc = np.nonzero(occ)
        r0, r1 = max(rr.min() - 6, 0), min(rr.max() + 7, S.shape[0])
        c0, c1 = max(cc.min() - 6, 0), min(cc.max() + 7, S.shape[1])
        S = S[r0:r1, c0:c1]

        # zoom on the vertex, where the three-layer table/diploe/table sandwich is thickest
        bone = np.isin(S, [idx[CANC], idx[CORT], idx[UNRES]])
        br, bc = np.nonzero(bone)
        zr = int(np.percentile(br, 97))                     # near the top of the skull
        zc = int(np.median(bc[br > np.percentile(br, 92)]))
        h, w = 46, 76
        zs = (slice(max(zr - h, 0), min(zr + 12, S.shape[0])),
              slice(max(zc - w // 2, 0), min(zc + w // 2, S.shape[1])))

        fig, ax = plt.subplots(1, 2, figsize=(15.4, 8.2),
                               gridspec_kw={"width_ratios": [1.35, 1]})
        norm = BoundaryNorm(np.arange(-0.5, len(rows) + 0.5), len(rows))
        ax[0].imshow(S, cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
        ax[0].add_patch(Rectangle((zs[1].start, zs[0].start),
                                  zs[1].stop - zs[1].start, zs[0].stop - zs[0].start,
                                  fill=False, edgecolor="#ff2d95", linewidth=1.8))
        ax[0].set_title(f"Tissue segmentation, SKULL split into two bone types\n"
                        f"sagittal x = AC{a.dx:+.0f} mm  (up = superior, right = anterior)",
                        fontsize=11.5)
        present = sorted(set(np.unique(S).tolist()))
        ax[0].legend(handles=[Patch(facecolor=cmap(i), edgecolor="#555",
                                    label=names[rows[i]]) for i in present if i != idx[0]],
                     loc="lower left", fontsize=8, framealpha=0.92, ncol=2)

        ax[1].imshow(S[zs], cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
        ax[1].set_title("zoom on the vault (magenta box):\n"
                        "outer table / diploe / inner table sandwich", fontsize=11.5)

        for k in (0, 1):
            ax[k].set_xticks([]); ax[k].set_yticks([])
        fig.suptitle(
            f"{sid} — cancellous {n[CANC]:,} vox ({100*n[CANC]/tot:.1f}% of bone) · "
            f"cortical {n[CORT]:,} ({100*n[CORT]/tot:.1f}%) · "
            f"unresolved {n[UNRES]:,} ({100*n[UNRES]/tot:.1f}%)\n"
            "the phantom itself stores ONE bone row — the split is recovered from GRACE's "
            "raw 11-class output, not from the optical properties",
            fontsize=12.5, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p = os.path.join(a.out, f"{sid}_skull_split.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {p}   cancellous {n[CANC]:,} cortical {n[CORT]:,} "
              f"unresolved {n[UNRES]:,} ({100*n[UNRES]/tot:.1f}%)")


if __name__ == "__main__":
    main()
