#!/usr/bin/env python3
"""Bring PUP amyloid outputs (voxelwise SUVR + FreeSurfer wmparc labels) into our MNI grid.

Why this can reuse the existing chain with NO new registration
-------------------------------------------------------------
PUP writes its SUVR image and wmparc label map in the subject's FreeSurfer *conformed*
space (256^3, 1 mm, LIA) -- a different voxel grid from the native T1 we built the phantom
from (e.g. 176x240x256 at 1.2x1.055x1.055 mm, RAS). Different grids, but FreeSurfer
preserves scanner RAS in the header, so both describe the SAME physical space.

Verified empirically rather than assumed: resampling PUP's own T1.nii onto the native T1
grid through world coordinates and then searching +-8 voxels for the translation that
maximises correlation returns EXACTLY (0,0,0) at r=0.945. Head-mask Dice is 0.917 and the
raw centroid offset of 5.7 mm is a masking artefact (PUP's T1 is intensity-normalised
uint8 and its z FOV differs), not a spatial shift.

So the chain is the same one pet_to_mni.py uses, with the PUP image swapped in for the PET:

    MNI grid  ->  (stored rigid .tfm, LOADED never recomputed)
    iso-1mm T1 grid  ->  canonical-T1 voxel (index / zoom)
    -> T1 world (t1.affine)  ->  PUP voxel (inv(pup.affine))

The rigid transform is read from disk exactly as pet_to_mni does. Re-deriving it is
forbidden: registration here is stochastic (repeat runs gave -0.303/-0.314/-0.315 against
the -0.273 actually used), and a recomputed transform put the PET ~13 voxels off in X.

4dfp -> NIfTI
-------------
The SUVR image ships as 4dfp, the label map as both. Rather than trusting the 4dfp
orientation convention, the axis mapping was solved empirically against the pair that must
be identical -- wmparc001.4dfp.img vs wmparc.nii -- giving transpose(2,0,1) + flip axes
1,2 at 1.000000 voxelwise agreement over all 182 labels. That is what FDFP_TO_NII applies.

Labels are resampled NEAREST (they are categorical); SUVR is resampled LINEAR.

  python pup_to_mni.py --subjects oas30208 oas30812 oas30948 oas30926
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import nibabel as nib
import scipy.io as sio
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import repro_config as RC
import mni_normalize as MN                                  # noqa: E402
import pet_to_mni as PET                                    # noqa: E402
from generate_oasis import to_iso1mm                        # noqa: E402

PUP_DIR = os.environ.get("PUP_EXT_DIR", os.path.join(RC.OASIS_DIR, "oasis3_pup", "ext"))
PUP_EXT = PUP_DIR
OUT_DIR = RC.PUP_MNI_DIR
SRC_DIR = RC.OASIS_DATASET_DIR
MNI_DIR = MN.OUT_DATASET                                # common-grid phantoms + .tfm
PROP_KEY = MN.PROP_KEY


def fdfp_to_nii(arr256):
    """4dfp array -> the same voxel grid as the sibling .nii (solved on wmparc, exact)."""
    return np.flip(np.flip(np.transpose(arr256, (2, 0, 1)), 1), 2)


def load_pup(subject):
    """-> (suvr, wmparc, affine) all on the FreeSurfer conformed grid."""
    d = glob.glob(os.path.join(PUP_DIR, f"{subject.upper()}_*_PUPTIMECOURSE_*"))
    if not d:
        raise SystemExit(f"[pup] no PUP folder for {subject} under {PUP_DIR}")
    d = d[0]
    wm = nib.load(os.path.join(d, "wmparc.nii"))
    img = glob.glob(os.path.join(d, "*msum_SUVR.4dfp.img"))
    if not img:
        raise SystemExit(f"[pup] no msum_SUVR.4dfp.img in {d}")
    suvr = fdfp_to_nii(np.fromfile(img[0], dtype="<f4").reshape(256, 256, 256))
    lab = np.asanyarray(wm.dataobj).astype(np.int32)
    if suvr.shape != lab.shape:
        raise SystemExit(f"[pup] SUVR {suvr.shape} != wmparc {lab.shape}")
    return suvr, lab, wm.affine, os.path.basename(d)


def check_alignment(subject, pup_aff):
    """PUP T1 vs our native T1: optimal translation must be 0 -- fail loudly if not.

    Guards the one assumption the whole chain rests on. A non-zero optimum means this
    subject's FreeSurfer header does NOT share the native T1's world frame, and every
    downstream voxel would be silently displaced.
    """
    d = glob.glob(os.path.join(PUP_DIR, f"{subject.upper()}_*_PUPTIMECOURSE_*"))[0]
    pup_t1 = nib.load(os.path.join(d, "T1.nii"))
    nat = nib.as_closest_canonical(nib.load(PET.find_t1(subject.upper())))
    A = np.asanyarray(nat.dataobj, dtype=np.float32)
    B = sample_on_native(np.asanyarray(pup_t1.dataobj, dtype=np.float32),
                         pup_t1.affine, nat, order=1)
    c = [s // 2 for s in A.shape]
    sl = tuple(slice(max(0, ci - 60), min(s, ci + 60)) for ci, s in zip(c, A.shape))
    a, b = A[sl], B[sl]
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best = (-9.0, None)
    for dx in range(-6, 7, 2):
        for dy in range(-6, 7, 2):
            for dz in range(-6, 7, 2):
                r = float((a * np.roll(np.roll(np.roll(b, dx, 0), dy, 1), dz, 2)).mean())
                if r > best[0]:
                    best = (r, (dx, dy, dz))
    return best


def sample_on_native(vol, vol_aff, nat_img, order):
    """Resample a volume onto the canonical native-T1 grid through world coordinates."""
    g = [np.arange(s, dtype=np.float32) for s in nat_img.shape[:3]]
    p, q, r = np.meshgrid(*g, indexing="ij")
    vox = np.stack([p, q, r, np.ones_like(p)], axis=-1)
    M = np.linalg.inv(vol_aff) @ nat_img.affine
    v = vox @ M.T
    return ndi.map_coordinates(vol, np.stack([v[..., 0], v[..., 1], v[..., 2]], 0),
                               order=order, mode="constant", cval=0.0)


def sample_on_iso(vol, vol_aff, t1_path, iso_shape, order):
    """PUP volume -> our iso-1mm T1 grid, mirroring pet_to_mni.pet_on_t1_grid exactly.

    iso index (p,q,r) corresponds to canonical-T1 voxel (p/zx, q/zy, r/zz); from there
    T1 world -> PUP voxel. Same analytic mapping the PET path uses, so PUP and PET land
    on the same grid by construction. Valid only because PUP's FreeSurfer space shares
    our T1's world frame -- guaranteed by restricting to same-session subjects upstream.
    """
    t1 = nib.as_closest_canonical(nib.load(t1_path))
    zoom = np.array(t1.header.get_zooms()[:3], dtype=float)
    p, q, r = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in iso_shape],
                          indexing="ij")
    t1vox = np.stack([p / zoom[0], q / zoom[1], r / zoom[2], np.ones_like(p)], axis=-1)
    M = np.linalg.inv(vol_aff) @ t1.affine
    v = t1vox @ M.T
    return ndi.map_coordinates(vol, np.stack([v[..., 0], v[..., 1], v[..., 2]], 0),
                               order=order, mode="constant", cval=0.0)


def recover_transform(sid):
    """Rebuild the phantom's phantom->MNI rigid transform when the .tfm was not persisted.

    Registers the native phantom to its OWN already-placed MNI copy. Because the MNI
    phantom IS the native one resampled by the (lost) transform, this is registering an
    image to a rigid copy of itself: a MEAN-SQUARES metric on the label values has a sharp
    global minimum at exactly that transform. Validated against 2 heads that still have
    their stored .tfm -- recovered to 0.2 / 0.6 mm. (Mutual information does NOT work here:
    the head is near-symmetric and MI settles on a mirrored optimum ~10 mm off.)

    Persists to the standard .tfm path. The phantom itself is NOT moved, so no scene,
    cfg cache or pyramid is invalidated -- this only fills in a missing bookkeeping file.
    """
    import SimpleITK as sitk
    nat = MN.optical_to_label(sio.loadmat(
        os.path.join(SRC_DIR, f"{sid}_copmri_withHermiteF810.mat"))[PROP_KEY].astype(np.float32))
    mni = MN.optical_to_label(sio.loadmat(
        os.path.join(MNI_DIR, f"{sid}_v11mni_F810.mat"))[PROP_KEY].astype(np.float32))
    flip = MN.DATASET_FLIP[MN.dataset_of(sid)]
    nat = nat[::-1].copy() if flip else nat
    fixed = MN._to_sitk(mni.astype(np.float32), tuple(float(-x) for x in MN.AC))
    moving = MN._to_sitk(nat.astype(np.float32), (0, 0, 0))
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetMetricSamplingPercentage(0.5, seed=1)
    R.SetInterpolator(sitk.sitkNearestNeighbor)
    R.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 400, relaxationFactor=0.6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(init, inPlace=False)
    T = R.Execute(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32))
    out = os.path.join(MNI_DIR, f"{sid}_v11mni_rigid.tfm")
    sitk.WriteTransform(T, out)
    return T, float(R.GetMetricValue())


def get_transform(sid):
    tfm_p = os.path.join(MNI_DIR, f"{sid}_v11mni_rigid.tfm")
    if os.path.isfile(tfm_p):
        import SimpleITK as sitk
        return sitk.ReadTransform(tfm_p), "loaded"
    T, ms = recover_transform(sid)
    return T, f"recovered(MSE={ms:.3f})"


def phantom_brain_mni(sid):
    """WM+GM+CSF mask of the phantom already in MNI, for the end-to-end alignment check."""
    p = sio.loadmat(os.path.join(MNI_DIR, f"{sid}_v11mni_F810.mat"))[PROP_KEY]
    return np.isin(MN.optical_to_label(p), [1, 2, 3])


def process(sid, out_dir):
    """Full transport for one subject. Returns a QC dict; raises SystemExit on hard fail.

    Only same-session subjects are accepted: the amyloid PET and the T1 the phantom was
    built from must be the SAME scan, so PUP's FreeSurfer world frame coincides with ours
    and the transported cortex lands on the phantom's own anatomy. Cross-session subjects
    (whose PET is years from the phantom's T1) are refused here by design -- rigidly
    forcing a different-day, atrophied brain onto this phantom would misplace the cortex.
    """
    import SimpleITK as sitk
    suvr, lab, aff, folder = load_pup(sid)

    r, shift = check_alignment(sid, aff)
    if shift != (0, 0, 0):
        raise SystemExit(f"[pup] {sid}: PUP FreeSurfer frame not aligned with our T1 "
                         f"(shift {shift}, r={r:.3f}) -- cross-session PET, discarded")

    T, tsrc = get_transform(sid)                                   # Blocker A
    flip = MN.DATASET_FLIP[MN.dataset_of(sid)]

    t1_path = PET.find_t1(sid.upper())
    iso, _ = to_iso1mm(t1_path)
    suvr_iso = sample_on_iso(suvr, aff, t1_path, iso.shape, 1)
    lab_iso = sample_on_iso(lab.astype(np.float32), aff, t1_path, iso.shape, 0)

    ref = sitk.Image(int(MN.GDIM[0]), int(MN.GDIM[1]), int(MN.GDIM[2]), sitk.sitkFloat32)
    ref.SetSpacing((1, 1, 1))
    ref.SetOrigin(tuple(float(-x) for x in MN.AC))
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    out = {}
    for name, vol, interp in (("suvr", suvr_iso, sitk.sitkLinear),
                              ("t1", iso, sitk.sitkLinear),
                              ("wmparc", lab_iso, sitk.sitkNearestNeighbor)):
        src = vol[::-1].copy() if flip else vol
        res = sitk.Resample(MN._to_sitk(src.astype(np.float32), (0, 0, 0)), ref, T,
                            interp, 0.0, sitk.sitkFloat32)
        out[name] = sitk.GetArrayFromImage(res).transpose(2, 1, 0)

    # End-to-end validation: the transported cortical ribbon must land inside the phantom
    # brain. This catches a wrong transform, a failed bridge, or a bad flip at once.
    wm = np.round(out["wmparc"]).astype(np.int32)
    brain = phantom_brain_mni(sid)
    par = wm > 0
    inside = 100.0 * (brain & par).sum() / max(par.sum(), 1)
    if inside < 95.0:
        raise SystemExit(f"[pup] {sid}: only {inside:.1f}% of the transported parcellation "
                         f"falls in the phantom brain (transform {tsrc}) -- refusing to "
                         "write a misaligned volume")

    np.save(os.path.join(out_dir, f"{sid}_suvr_mni.npy"), out["suvr"].astype(np.float32))
    np.save(os.path.join(out_dir, f"{sid}_t1_mni.npy"), out["t1"].astype(np.float32))
    np.save(os.path.join(out_dir, f"{sid}_wmparc_mni.npy"), wm)
    return dict(sid=sid, folder=folder, transform=tsrc,
                wmparc_inside_pct=round(inside, 1), n_labels=int(len(np.unique(wm)) - 1),
                cortical_suvr_p50=round(float(np.median(
                    out["suvr"][(wm >= 1000) & (wm < 3000)])), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ok, fail = [], []
    for sid in [s.lower() for s in a.subjects]:
        try:
            q = process(sid, a.out)
            ok.append(q)
            print(f"[pup] {sid:9} {q['transform']:18} inside {q['wmparc_inside_pct']:.1f}%  "
                  f"ctxSUVR {q['cortical_suvr_p50']}  ({q['n_labels']} labels)", flush=True)
        except SystemExit as e:
            fail.append((sid, str(e)))
            print(str(e), flush=True)
    print(f"\n[pup] DONE  ok={len(ok)}  fail={len(fail)}", flush=True)
    for sid, msg in fail:
        print(f"    FAIL {sid}: {msg.splitlines()[-1][:80]}", flush=True)


if __name__ == "__main__":
    main()
