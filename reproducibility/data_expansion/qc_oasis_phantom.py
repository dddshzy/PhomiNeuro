#!/usr/bin/env python3
"""QC figure for an OASIS 8-tissue phantom: orientation, structure, optical properties.

  python qc_oasis_phantom.py --sid oa_test01
"""
import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import repro_config as RC
from optical_tissue_lut import OASIS8_LABEL_NAME, OASIS8_LABEL_TO_P0ROW

DS = RC.OASIS_DATASET_DIR
COLORS = ["#000000", "#e8b892", "#f2f2f0", "#5fa8e8", "#9b9b9b",
          "#e8e2c8", "#d92b2b", "#f5e04a", "#a84fd0"]
NAMES = [OASIS8_LABEL_NAME[i] for i in range(9)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sid", default="oa_test01")
    a = ap.parse_args()
    lab = np.load(os.path.join(DS, f"{a.sid}_labels8.npy"))
    t1 = np.load(os.path.join(DS, f"{a.sid}_t1_iso.npy"))
    print(f"labels {lab.shape}, T1 {t1.shape}")

    cmap = ListedColormap(COLORS); norm = BoundaryNorm(np.arange(-.5, 9.5), cmap.N)
    # centre of the head, and the centre of the vessel tree (to show TOF coverage)
    occ = np.argwhere(lab > 0); ctr = occ.mean(0).astype(int)
    ves = np.argwhere(lab == 6)
    vctr = ves.mean(0).astype(int) if len(ves) else ctr
    # RAS: axis0=R(sagittal), axis1=A(coronal), axis2=S(axial)
    planes = [("Sagittal (R-L)", 0), ("Coronal (A-P)", 1), ("Axial (S-I)", 2)]

    fig, ax = plt.subplots(3, 3, figsize=(16, 15))
    for c, (nm, axis) in enumerate(planes):
        sl = [slice(None)] * 3; sl[axis] = int(ctr[axis])
        ax[0, c].imshow(np.rot90(t1[tuple(sl)]), cmap="gray", origin="lower")
        ax[0, c].set_title(f"T1w  {nm}", fontsize=11)
        ax[1, c].imshow(np.rot90(lab[tuple(sl)]), cmap=cmap, norm=norm,
                        origin="lower", interpolation="nearest")
        ax[1, c].set_title(f"8-tissue labels  {nm}", fontsize=11)
        # vessel-centred slice so the TOF slab is actually visible
        sl2 = [slice(None)] * 3; sl2[axis] = int(vctr[axis])
        ax[2, c].imshow(np.rot90(lab[tuple(sl2)]), cmap=cmap, norm=norm,
                        origin="lower", interpolation="nearest")
        ax[2, c].set_title(f"at vessel centroid  {nm}", fontsize=11)
        for r in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    # orientation ticks on the label row so anatomy can be checked at a glance
    ax[1, 0].set_xlabel("← posterior      anterior →", fontsize=9)
    ax[1, 0].set_ylabel("← inferior      superior →", fontsize=9)
    ax[1, 1].set_xlabel("← right          left →", fontsize=9)
    ax[1, 1].set_ylabel("← inferior      superior →", fontsize=9)
    ax[1, 2].set_xlabel("← right          left →", fontsize=9)
    ax[1, 2].set_ylabel("← posterior     anterior →", fontsize=9)

    counts = {i: int((lab == i).sum()) for i in range(1, 9)}
    tot = sum(counts.values())
    handles = [Patch(facecolor=COLORS[i], edgecolor="k", lw=.4,
                     label=f"{i} {NAMES[i]:7s} {counts[i]:>9,} ({100*counts[i]/tot:4.1f}%)")
               for i in range(1, 9)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=True)
    fig.suptitle(f"OASIS-3 8-tissue phantom  {a.sid}   shape={lab.shape}  1 mm iso, RAS\n"
                 f"row1 T1w · row2 labels at head centre · row3 labels at vessel centroid "
                 f"(TOF slab)", fontsize=13)
    fig.tight_layout(rect=[0, 0.075, 1, 0.96])
    out = os.path.join(DS, f"{a.sid}_phantom_qc.png")
    fig.savefig(out, dpi=105); print("saved", out)

    # ---- optical property maps (what MCX actually consumes) ----
    import scipy.io as sio
    import v2_manifest as M
    prop = sio.loadmat(os.path.join(DS, f"{a.sid}_copmri_withHermiteF810.mat"))[M.PROP_KEY]
    fig2, ax2 = plt.subplots(1, 4, figsize=(20, 5.5))
    titles = [r"$\mu_a$ (mm$^{-1}$)", r"$\mu_s$ (mm$^{-1}$)", "g", "n"]
    z = int(vctr[2])
    for i in range(4):
        im = ax2[i].imshow(np.rot90(prop[:, :, z, i]), cmap="viridis", origin="lower")
        ax2[i].set_title(titles[i], fontsize=12); ax2[i].set_xticks([]); ax2[i].set_yticks([])
        plt.colorbar(im, ax=ax2[i], fraction=.046)
    fig2.suptitle(f"{a.sid}  F810 optical properties, axial slice z={z} (vessel centroid)",
                  fontsize=13)
    fig2.tight_layout()
    out2 = os.path.join(DS, f"{a.sid}_optical_qc.png")
    fig2.savefig(out2, dpi=105); print("saved", out2)


if __name__ == "__main__":
    main()
