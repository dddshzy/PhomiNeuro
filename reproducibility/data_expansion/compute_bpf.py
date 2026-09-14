#!/usr/bin/env python3
"""Brain Parenchymal Fraction (BPF) for the 84-head held-out OASIS AD/HC cohort.

BPF = (WM + GM) / (WM + GM + CSF)  — brain parenchyma volume / intracranial volume (ICV).
Tissue labels come from the phantom (ANTs-Atropos CSF/GM/WM, the validated brain segmentation
in the hybrid pipeline). The phantom folds the eyes into CSF (optical-property choice), so eye
voxels (GRACE label 3, already transported to MNI) are removed from CSF before the ratio.
Volumes are voxel counts on the 1-mm isotropic grid (mL = voxels / 1000).
"""
import os, sys, csv
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "viz"))
import repro_config as RC
import mni_normalize as MN
import fig_skull_split as SK

META = os.path.join(RC.OASIS_METADATA_DIR, "selected_84_groups.csv")
MNI_DIR = RC.MNI_DATASET_DIR
OUT = os.path.join(RC.OASIS_METADATA_DIR, "cohort_bpf.csv")
WM, GM, CSF = 1, 2, 3          # phantom _P0F rows

def bpf(sid):
    lab = MN.optical_to_label(sio.loadmat(
        os.path.join(MNI_DIR, f"{sid}_v11mni_F810.mat"))["vol_prop_eye_aseg"])
    eyes = SK.grace_in_mni(sid) == 3
    wm = int((lab == WM).sum()); gm = int((lab == GM).sum())
    csf = int(((lab == CSF) & ~eyes).sum())
    icv = wm + gm + csf
    return dict(wm_ml=wm/1000, gm_ml=gm/1000, csf_ml=csf/1000, icv_ml=icv/1000,
                bpf=(wm+gm)/icv, gm_wm=gm/wm)

def main():
    heads = [(r["subject"].lower(), r["group"]) for r in csv.DictReader(open(META))]
    rows = []
    for i, (sid, grp) in enumerate(heads):
        try:
            b = bpf(sid); rows.append((sid, grp, b))
            print(f"[{i+1}/84] {sid} {grp:7s} BPF={b['bpf']:.4f} ICV={b['icv_ml']:.0f}mL "
                  f"(WM{b['wm_ml']:.0f}/GM{b['gm_ml']:.0f}/CSF{b['csf_ml']:.0f})", flush=True)
        except Exception as e:
            print(f"[{i+1}/84] FAIL {sid}: {e}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "group", "bpf", "icv_ml", "wm_ml", "gm_ml", "csf_ml", "gm_wm"])
        for sid, grp, b in rows:
            w.writerow([sid, grp, f"{b['bpf']:.4f}", f"{b['icv_ml']:.1f}", f"{b['wm_ml']:.1f}",
                        f"{b['gm_ml']:.1f}", f"{b['csf_ml']:.1f}", f"{b['gm_wm']:.3f}"])
    import numpy as np
    for g in ("AD", "healthy"):
        v = np.array([r[2]["bpf"] for r in rows if r[1] == g])
        print(f"\n{g}: BPF {v.mean():.4f} ± {v.std(ddof=1):.4f}  [{v.min():.3f}, {v.max():.3f}]  n={len(v)}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
