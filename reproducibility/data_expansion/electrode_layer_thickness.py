#!/usr/bin/env python3
"""Per-electrode extra-cerebral tissue-layer THICKNESS on a labeled segmentation.

Algorithm migrated from BrainCalculator (Junha0Zhang/BrainCalculator, `calculation.ipynb`):
that tool measures layer thickness as the local PERPENDICULAR distance between a tissue's
outer and inner surface (it samples the outer surface, uses vertex normals, and takes the
k-NN distance to the inner surface). We adapt it to our scenario:

  * work on the GRACE 11-class SEGMENTATION (not the optical-property volume), which resolves
    skin, fat, muscle and BOTH bone tables (cancellous / cortical);
  * localize the measurement to the 19 EEG 10-20 electrode sites used by our MCX scenes;
  * at each electrode, estimate the local scalp-surface NORMAL (gradient of the smoothed head
    mask) and march inward along it, recording the run-length (mm) of each label until the ray
    reaches brain. This is BrainCalculator's perpendicular outer->inner thickness, evaluated
    per layer along the electrode normal (which is also the transcranial beam direction).

GRACE labels: 1 WM, 2 GM, 3 eyes, 4 CSF, 5 air, 6 blood, 7 cancellous bone, 8 cortical bone,
9 skin, 10 fat, 11 muscle.
"""
import os, sys, csv
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "viz"))
import repro_config as RC
import mni_normalize as MN
import fig_skull_split as SK          # reuse grace_in_mni() (GRACE -> MNI via stored rigid tfm)

WM, GM, EYES, CSF, AIR, BLOOD, CANC, CORT, SKIN, FAT, MUSCLE = range(1, 12)
BRAIN = {WM, GM, CSF}                 # ray stop: sustained brain/CSF = past the skull
BONE = {CANC, CORT}
STEP = 0.25                           # mm per ray sample (grid is 1 mm iso)
MAXD = 45.0                           # mm marched inward

def local_normal_field(solid):
    """Outward-pointing unit normal at every voxel, from the smoothed head-mask gradient."""
    sm = gaussian_filter(solid.astype(np.float32), sigma=2.0)
    gx, gy, gz = np.gradient(sm)       # points toward INCREASING mask density = inward
    return gx, gy, gz                  # inward gradient; we orient per-electrode below

def march(grace, start, indir, solid_grad):
    """Return per-label thickness (mm) along the inward ray from `start`."""
    gx, gy, gz = solid_grad
    p0 = np.round(start).astype(int)
    # local inward normal from the smoothed-mask gradient, oriented to agree with `indir`
    n = np.array([gx[tuple(p0)], gy[tuple(p0)], gz[tuple(p0)]], float)
    if np.linalg.norm(n) < 1e-6 or np.dot(n, indir) < 0:
        n = indir.copy()
    n = n / np.linalg.norm(n)
    labs = []
    for r in np.arange(0.0, MAXD, STEP):
        ijk = np.round(start + r * n).astype(int)
        if not all(0 <= ijk[i] < grace.shape[i] for i in range(3)):
            break
        labs.append(int(grace[tuple(ijk)]))
    labs = np.array(labs)
    out = dict(skin=0.0, fat=0.0, muscle=0.0, skull=0.0, cancellous=0.0, cortical=0.0,
               normal=n, scalp=np.asarray(start, float), labs=labs,
               entry=labs[0] if len(labs) else 0, flag="")
    if len(labs) == 0:
        return out
    # extra-cerebral column = from scalp to where the ray enters SUSTAINED brain (>=3 mm WM/GM)
    win = int(round(3.0 / STEP))
    brain = np.isin(labs, [WM, GM]).astype(int)
    col_end = len(labs)
    for i in range(len(labs) - win):
        if brain[i:i + win].mean() > 0.8:
            col_end = i; break
    col = labs[:col_end]
    bone_idx = np.where(np.isin(col, list(BONE)))[0]
    if len(bone_idx) == 0:
        out["flag"] = "no-bone"; return out
    i0, i1 = bone_idx[0], bone_idx[-1]
    pre = col[:i0]; span = col[i0:i1 + 1]
    out["skin"] = np.sum(pre == SKIN) * STEP
    out["fat"] = np.sum(pre == FAT) * STEP
    out["muscle"] = np.sum(pre == MUSCLE) * STEP
    out["cancellous"] = np.sum(col == CANC) * STEP        # total bone the beam crosses
    out["cortical"] = np.sum(col == CORT) * STEP
    out["skull"] = out["cancellous"] + out["cortical"]
    # reliability: soft tissue / air interleaved between the outer & inner bone, or implausibly
    # thick skull -> the ray grazed a complex region (sphenoid, temporal fossa, sinus)
    interleaved = np.sum(np.isin(span, [SKIN, FAT, MUSCLE, AIR])) * STEP
    if out["skull"] > 9.0 or interleaved > 1.0:
        out["flag"] = "check(%s%s)" % ("thick;" if out["skull"] > 9 else "",
                                       "interleaved" if interleaved > 1 else "")
    return out

def measure(sid, grace=None):
    """Return (grace_volume, [(electrode, thickness_dict), ...]). thickness_dict carries the
    per-layer mm, plus 'labs' (label sequence), 'scalp', 'normal' for plotting/QC."""
    if grace is None:
        grace = SK.grace_in_mni(sid)                   # (224,256,300) uint8 labels on MNI grid
    solid = (grace > 0) & (grace != AIR)               # scalp-bounded head
    grad = local_normal_field(solid)
    M = MN._load_mni()
    rows = []
    for name in MN.NAMES:
        d_out = M["dirs"][name]                        # outward radial (AC -> electrode)
        scalp = MN.snap_scalp(solid, d_out)            # outermost tissue voxel along the ray
        rows.append((name, march(grace, scalp, -d_out, grad)))
    return grace, rows

def run(sid):
    return measure(sid)[1]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = run(a.subject)
    hdr = ["electrode", "skin_mm", "muscle_mm", "fat_mm", "skull_mm", "cancellous_mm",
           "cortical_mm", "entry_label", "flag"]
    print(f"\n{a.subject}  extra-cerebral layer thickness at 19 EEG 10-20 electrodes (mm)")
    print(f"{'elec':5s}{'skin':>7s}{'muscle':>7s}{'fat':>6s}{'skull':>7s}{'canc':>6s}{'cort':>6s}  flag")
    print("-" * 52)
    agg = {k: [] for k in ("skin", "muscle", "fat", "skull", "cancellous", "cortical")}
    for name, th in rows:
        print(f"{name:5s}{th['skin']:7.1f}{th['muscle']:7.1f}{th['fat']:6.1f}"
              f"{th['skull']:7.1f}{th['cancellous']:6.1f}{th['cortical']:6.1f}  {th['flag']}")
        if not th["flag"]:
            for k in agg: agg[k].append(th[k])
    print("-" * 52)
    print(f"{'mean*':5s}{np.mean(agg['skin']):7.1f}{np.mean(agg['muscle']):7.1f}"
          f"{np.mean(agg['fat']):6.1f}{np.mean(agg['skull']):7.1f}"
          f"{np.mean(agg['cancellous']):6.1f}{np.mean(agg['cortical']):6.1f}  (*reliable sites only)")
    n_musc = sum(1 for _, th in rows if th["muscle"] > 0.5)
    n_flag = sum(1 for _, th in rows if th["flag"])
    print(f"\nmuscle present (>0.5 mm) at {n_musc}/19 electrodes; {n_flag} site(s) flagged "
          f"(sphenoid/fossa/sinus — GRACE bone/soft-tissue label unreliable there).")
    out = a.csv or str(RC.PUP_MNI_DIR / f"{a.subject}_electrode_thickness.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr)
        for name, th in rows:
            w.writerow([name, f"{th['skin']:.2f}", f"{th['muscle']:.2f}", f"{th['fat']:.2f}",
                        f"{th['skull']:.2f}", f"{th['cancellous']:.2f}", f"{th['cortical']:.2f}",
                        th["entry"], th["flag"]])
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
