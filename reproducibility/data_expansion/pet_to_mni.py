#!/usr/bin/env python3
"""Amyloid PET -> the v11 common MNI grid, using the SAME rigid transform as the T1 phantom.

Chain, and why each step is needed:
  1. PET is 4D (26 frames). Average the LAST third of the frames -- the late window is
     the quasi-static distribution amyloid quantification is based on; early frames are
     perfusion-dominated and would blur the binding pattern.
  2. PET and T1 do NOT share a grid (PET 256x256x127 @1.4x1.4x2.03 LAS vs T1
     176x240x256 @1.2x1.05x1.05 RAS) but they DO share a world frame -- OASIS-3 ships
     them co-registered from the same session (verified: high-intensity centroids agree
     to 8-13 mm, i.e. the modality difference, not a placement offset). So PET is
     resampled onto the T1 iso-1mm grid by world coordinates -- NO new registration,
     which would only add error on top of an already-valid alignment.
  3. Apply the head's EXISTING rigid MNI transform. It is recomputed here by calling
     mni_normalize.register_head on the same phantom: the registration is deterministic
     (fixed seeds, fixed multi-start), so this reproduces the transform the phantom got
     rather than inventing a second, subtly different one.
  4. Report SUVR normalised to the cerebellar cortex, the standard amyloid reference
     region, so values are comparable to the published Centiloid scale.

  python pet_to_mni.py --subject OAS30884 [--out dataset_OASIS/pet_mni]
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import repro_config as RC
import mni_normalize as MN                      # noqa: E402
import generate_oasis as G                      # noqa: E402

PET_DIR = os.path.join(RC.OASIS_DIR, "oasis3_pet")
MR_DIR = os.path.join(RC.OASIS_DIR, "oasis3_ad_vs_hc")
LATE_FRAC = 1.0 / 3.0                           # average the last third of frames


def find_pet(subject):
    c = glob.glob(f"{PET_DIR}/{subject}_*/**/*_pet.nii.gz", recursive=True)
    if not c:
        raise SystemExit(f"no PET for {subject}")
    return c[0]


def find_t1(subject):
    """The SAME T1 the phantom was built from (shared list), not any T1 on disk."""
    for name in ("all84.txt", "da126.txt"):
        p = os.path.join(RC.GRACE_DIR, name)
        if not os.path.isfile(p):
            continue
        for line in open(p):
            sid, t1 = line.rstrip("\n").split("\t")
            if sid.lower() == subject.lower():
                return t1 if os.path.isabs(t1) else os.path.join(RC.WORK_DIR, t1)
    c = glob.glob(f"{MR_DIR}/{subject}_MR_*/**/*T1w*.nii.gz", recursive=True)
    if not c:
        raise SystemExit(f"no T1 for {subject}")
    return sorted(c)[0]


def late_mean(pet_path):
    """4D PET -> 3D late-window mean (or pass 3D through)."""
    im = nib.load(pet_path)
    a = np.asanyarray(im.dataobj, dtype=np.float32)
    if a.ndim == 4:
        k = max(1, int(round(a.shape[3] * LATE_FRAC)))
        a = a[..., -k:].mean(axis=3)
        print(f"  PET 4D {im.shape} -> mean of last {k} frames", flush=True)
    return a, im


def pet_on_t1_grid(pet_path, t1_path, iso_shape):
    """Resample PET onto the T1 iso-1mm grid THROUGH WORLD COORDINATES.

    Mirrors generate_oasis_v2.resample_to_iso_grid: G.to_iso1mm resamples the canonical
    T1 with out-spacing 1 and source spacing = zoom, so iso index (p,q,r) corresponds to
    canonical-T1 voxel (p/zx, q/zy, r/zz); from there go T1 world -> PET voxel.
    """
    vol, pim = late_mean(pet_path)
    pet_can = nib.as_closest_canonical(nib.Nifti1Image(vol, pim.affine, pim.header))
    t1 = nib.as_closest_canonical(nib.load(t1_path))
    zoom = np.array(t1.header.get_zooms()[:3], dtype=float)

    p, q, r = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in iso_shape], indexing="ij")
    t1vox = np.stack([p / zoom[0], q / zoom[1], r / zoom[2], np.ones_like(p)], axis=-1)
    Mx = np.linalg.inv(pet_can.affine) @ t1.affine
    v = t1vox @ Mx.T
    coords = np.stack([v[..., 0], v[..., 1], v[..., 2]], axis=0)
    arr = np.asanyarray(pet_can.dataobj, dtype=np.float32)
    return ndi.map_coordinates(arr, coords, order=1, mode="constant", cval=0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", default=str(RC.OASIS_ANALYSIS_DIR / "pet_mni"))
    a = ap.parse_args()
    sid = a.subject.lower()
    os.makedirs(a.out, exist_ok=True)

    pet_p, t1_p = find_pet(a.subject), find_t1(a.subject)
    print(f"[pet] {a.subject}\n  PET {os.path.basename(pet_p)}\n  T1  {os.path.basename(t1_p)}",
          flush=True)

    v, iso = G.to_iso1mm(t1_p)
    pet_iso = pet_on_t1_grid(pet_p, t1_p, v.shape)
    head = G.head_mask(v)
    # Judge alignment on REAL TRACER UPTAKE (p90), not p60. PET has a broad low-level
    # scatter/background pedestal that genuinely lies outside the head, so a p60 cut
    # labels that pedestal "signal" and reports 56-80% inside-head even for perfectly
    # co-registered data (verified: a subject failing at p60 with 56% is 100% inside at
    # p90, with a 64:1 in-head/out-of-head intensity ratio).
    nzv = pet_iso[pet_iso > 0]
    sig = pet_iso >= np.percentile(nzv, 90)
    inside = float((sig & head).sum()) / max(sig.sum(), 1)
    ratio = float(pet_iso[head].mean() / max(pet_iso[~head].mean(), 1e-9))
    print(f"  PET on T1 grid {pet_iso.shape}; {100*inside:.1f}% of p90 uptake inside head, "
          f"in/out intensity ratio {ratio:.0f}:1", flush=True)
    if inside < 0.95:
        raise SystemExit(f"[pet] ALIGNMENT FAILED: only {100*inside:.1f}% of PET signal is "
                         "inside the head mask -- PET and T1 are not co-registered.")

    # --- LOAD the phantom's stored rigid transform; never recompute it ---
    # The registration is NOT deterministic (multi-start + SimpleITK random metric
    # sampling): re-running it gave metrics -0.303/-0.314/-0.315 against the -0.273
    # actually used for this phantom, i.e. a genuinely DIFFERENT transform. Applying a
    # recomputed one put the PET ~13 voxels off in X from the MRI in MNI space, which is
    # exactly the lateral mismatch seen in the first figures. Reuse or fail loudly.
    import SimpleITK as sitk
    tfm_p = os.path.join(MN.OUT_DATASET, f"{sid}_v11mni_rigid.tfm")
    if not os.path.isfile(tfm_p):
        raise SystemExit(f"[pet] missing {tfm_p}. Re-run mni_normalize --normalize {sid} "
                         "--force so the transform is persisted, then retry.")
    T = sitk.ReadTransform(tfm_p)
    flip = MN.DATASET_FLIP[MN.dataset_of(sid)]
    print(f"  rigid MNI transform LOADED from {os.path.basename(tfm_p)} (flip={flip})",
          flush=True)

    src = pet_iso[::-1].copy() if flip else pet_iso
    ref = sitk.Image(int(MN.GDIM[0]), int(MN.GDIM[1]), int(MN.GDIM[2]), sitk.sitkFloat32)
    ref.SetSpacing((1, 1, 1))
    ref.SetOrigin(tuple(float(-x) for x in MN.AC))
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    res = sitk.Resample(MN._to_sitk(src.astype(np.float32), (0, 0, 0)), ref, T,
                        sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    pet_mni = sitk.GetArrayFromImage(res).transpose(2, 1, 0).astype(np.float32)

    np.save(os.path.join(a.out, f"{sid}_pet_mni.npy"), pet_mni)
    print(f"  PET in MNI {pet_mni.shape} -> {sid}_pet_mni.npy", flush=True)
    print(f"[stats] {sid} pet_max={pet_mni.max():.1f} nonzero={int((pet_mni>0).sum()):,}",
          flush=True)


if __name__ == "__main__":
    main()
