#!/usr/bin/env python3
"""
v11 rigid MNI normalization: bring every phantom into ONE common MNI-oriented grid
WITHOUT distorting tissue-layer thickness (RIGID only), so all phantoms share the
same size + a single aim origin O = AC voxel, and the 19 EEG 10-20 sources are a
single fixed set of MNI directions snapped per-subject to the scalp.

Why rigid (not affine): we simulate photon transport through real tissue layers;
affine scale/shear would change scalp/skull/CSF thickness. Rigid is an isometry ->
thicknesses preserved. Because heads are already orientation-standardized (canonical
axes ~ MNI axes), the rigid transform is mostly translation + a small rotation, so
NN resampling is near-lossless.

Pipeline per head:
  brain pseudo-T1 (from optical rows) --RIGID(Mattes MI)--> MNI152 template;
  resample the tissue LABEL (NN) into the common grid, remap label->optical.
L-R handedness (our canonical frame left the L-R sign don't-care) is pinned
PER-DATASET (constant within scb / bw / sh): with RIGID the mirror cannot be hidden
by shear, so register-as-is vs register-X-flipped has a LARGE MI gap. See
DATASET_FLIP (fill via `--determine`).

O = AC voxel (fixed); 19 source dirs = normalize(MNE standard_1020 MNI coords).
"""
import os, sys, json, numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import repro_config as RC
from optical_tissue_lut import _P0, para5_row

PROP_KEY = "vol_prop_eye_aseg"
DATASET_SCBBW = RC.STANDARD_DATASET_DIR
DATASET_SHARM = RC.SHARM_DATASET_DIR
# OASIS-3 T1-derived phantoms. Overridable via FM_OASIS_DATASET so a second OASIS
# cohort (e.g. the 125-subject domain-adaptation set in dataset_oasis810_da126) can be
# normalized without touching the locked AD-vs-HC set or duplicating this module.
DATASET_OASIS = os.environ.get("FM_OASIS_DATASET", RC.OASIS_DATASET_DIR)
OUT_DATASET = RC.MNI_DATASET_DIR
# The 9-row _P0 table is authoritative and untouched -- it is what the phantom generators used.
# But scatterBrains phantoms do NOT store a table row for their outer layer: ScatterBrains
# segments a SINGLE extra-cranial tissue, and the generator (optical_tissue_lut.para5_row, label 1)
# gives it the VOLUME-WEIGHTED BLEND of skin/muscle/fat -- a computed value, not a row. So a
# round-trip that only knows the 9 rows can never match it, and every scb scalp voxel silently fell
# through to label 0 = AIR (948k voxels in scb15, the largest tissue after air), after which its
# MCX ground truth was simulated on a bare-skull head. Append the blend as an extra label so the
# round-trip is lossless. bw and sh keep their separate skin/fat/muscle rows -- their phantoms
# genuinely resolve those layers, and we are not flattening real anatomy to make the sources match.
_P0F = np.vstack([np.array(_P0["F810"]), para5_row("F810", 1)])     # row 9 = scb scalp blend

# ---- common MNI-oriented grid ----
# Sized from measured head extents relative to AC: vertex up to ~97 mm above AC
# (neck-less BrainWeb heads sit highest), neck down to ~152 mm below AC. Z gives
# 125 above / 175 below AC with margin so nothing clips at the vertex (Cz).
GDIM = (224, 256, 300)                    # padded to hold whole head+neck
AC = np.array([112, 148, 175])            # MNI origin (AC) -> this voxel; grid origin = -AC
NAMES = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T3","C3","Cz","C4","T4",
         "T5","P3","Pz","P4","T6","O1","O2"]

# ---- per-dataset L-R flip (True = stored mirrored vs MNI) ----
# A fixed no-flip convention is used because intensity-based registration against
# the near-symmetric MNI152 template does not robustly determine handedness.
DATASET_FLIP = {"scb": False, "bw": False, "sh": False, "oa": False}   # OASIS T1 stored RAS

_MNI = {}


def _load_mni():
    if _MNI: return _MNI
    from nilearn.datasets import load_mni152_template
    import mne; mne.set_log_level("ERROR")
    t = load_mni152_template(resolution=1)
    arr = np.asarray(t.dataobj).astype(np.float32); arr /= (arr.max() or 1)
    mont = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    _MNI["arr"] = arr; _MNI["origin"] = t.affine[:3, 3]
    _MNI["dirs"] = {n: (np.array(mont[n]) * 1000.0) / np.linalg.norm(np.array(mont[n]) * 1000.0)
                    for n in NAMES}
    return _MNI


def dataset_of(head):
    if head.startswith("oa"): return "oa"
    return "sh" if head.startswith("sh") else ("bw" if head.startswith("bw") else "scb")


def std_mat(head):
    d = (DATASET_OASIS if head.startswith("oa") else
         DATASET_SHARM if head.startswith("sh") else DATASET_SCBBW)
    return os.path.join(d, f"{head}_copmri_withHermiteF810.mat")


def optical_to_label(prop):
    """Property volume -> label, for the NN resampling round-trip.

    This round-trip (optical -> label -> optical) is NEW in v11: v10 consumed the baked property
    volumes directly and never needed a table. It is LOSSY by construction -- any (mu_a, mu_s) not
    in _P0F silently stays at label 0, i.e. AIR -- and it did exactly that: the scb phantoms come
    from a different generator and use SCALP (0.03057, 11.694) where bw/sh use SKIN (0.045,
    19.818). _P0F was written for the sh/bw convention, so every scb head lost its entire scalp
    layer (948k voxels in scb15, the largest tissue after air) to air, and its MCX ground truth was
    then simulated on a scalp-less head with the source sitting on bare skull.

    The assert below is the check that would have caught it on the first scb phantom. A lossy
    conversion without a no-loss assertion is how data disappears quietly.
    """
    mua, mus = prop[..., 0], prop[..., 1]; lab = np.zeros(prop.shape[:3], np.uint8)
    for r in range(len(_P0F)):
        lab[np.isclose(mua, _P0F[r, 0]) & np.isclose(mus, _P0F[r, 1])] = r
    tissue = mus > 0                                  # anything that scatters is NOT air
    lost = tissue & (lab == 0)
    if lost.any():
        bad = np.unique(np.stack([mua[lost], mus[lost]], 1), axis=0)[:5]
        raise ValueError(
            f"optical_to_label: {int(lost.sum()):,} scattering voxels matched no row of _P0F and "
            f"would silently become AIR. Unmatched (mu_a, mu_s): {bad.tolist()}. "
            f"Add them to _P0 in optical_tissue_lut.py -- do NOT let them fall through.")
    return lab


def label_to_optical(lab):
    return _P0F[lab].astype(np.float32)           # (X,Y,Z,4)


def pseudo_t1(lab):
    img = np.zeros(lab.shape, np.float32); img[lab == 3] = 0.2; img[lab == 2] = 0.6; img[lab == 1] = 1.0
    return img


def _to_sitk(arr, origin):
    import SimpleITK as sitk
    im = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))
    im.SetSpacing((1, 1, 1)); im.SetOrigin(tuple(float(o) for o in origin))
    im.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1)); return im


def _rigid_once(fixed, moving, seed, init_mode="GEOMETRY"):
    import SimpleITK as sitk
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
        getattr(sitk.CenteredTransformInitializerFilter, init_mode))
    R = sitk.ImageRegistrationMethod(); R.SetMetricAsMattesMutualInformation(64)
    R.SetMetricSamplingStrategy(R.REGULAR); R.SetMetricSamplingPercentage(0.4, seed=seed)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=500,
        convergenceMinimumValue=1e-7, convergenceWindowSize=15)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([6, 4, 2, 1]); R.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
    R.SetInitialTransform(init, inPlace=False)
    T = R.Execute(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32))
    return T, R.GetMetricValue()


def _rigid(fixed, moving_arr):
    """Multi-start rigid (keep best MI) for robustness against bad local optima.

    Starts span BOTH centred initialisers, not just GEOMETRY. On the OASIS cohort,
    GEOMETRY-only starts put 3/84 heads in a bad local optimum (MI ~ -0.04..-0.09 vs
    a -0.37 cohort mean) which pushed the head partly outside the common grid and
    silently discarded up to 38% of its tissue. MOMENTS centres on the intensity
    centre of mass instead of the bounding box, which is robust to how much neck the
    FOV happens to include -- the thing that differs between these scans.
    """
    import SimpleITK as sitk
    moving = _to_sitk(moving_arr, (0, 0, 0))
    best_T, best_m = None, np.inf
    for init_mode in ("GEOMETRY", "MOMENTS"):
        for seed in (1, 42):
            try:
                T, m = _rigid_once(fixed, moving, seed, init_mode)
            except RuntimeError as e:
                # MOMENTS can place the moving image entirely off the fixed one for some
                # heads ("All samples map outside moving image buffer"). That is a bad
                # START, not a bad head -- the other starts are still valid. Without this
                # guard the exception propagated and killed the whole batch: a 12-head
                # re-registration silently stopped after 2, leaving 5 heads untouched
                # while the run looked like it had merely "not improved" them.
                print(f"  [rigid] start {init_mode}/seed{seed} failed "
                      f"({str(e).strip().splitlines()[-1][:60]}) -- skipping", flush=True)
                continue
            if m < best_m:
                best_T, best_m = T, m
    if best_T is None:
        raise RuntimeError("all rigid starts failed")
    return best_T, best_m


def _fixed():
    M = _load_mni(); return _to_sitk(M["arr"], M["origin"])


def register_head(head, flip):
    """Rigid register; return (transform, metric). flip -> X-mirror native first."""
    prop = sio.loadmat(std_mat(head))[PROP_KEY].astype(np.float32)
    lab = optical_to_label(prop)
    nat = pseudo_t1(lab[::-1].copy() if flip else lab)
    return _rigid(_fixed(), nat), lab


def determine_flip(heads):
    """For each head compare as-is vs flipped rigid MI; return per-head + majority."""
    res = {}
    for h in heads:
        (_, m0), _ = register_head(h, False)
        (_, mf), _ = register_head(h, True)
        res[h] = dict(m0=m0, mf=mf, flip=bool(mf < m0), gap=float(abs(mf - m0)))
        print(f"  {h}: as-is={m0:.4f} flip={mf:.4f} gap={abs(mf-m0):.4f} -> "
              f"{'FLIP' if mf<m0 else 'as-is'}", flush=True)
    return res


def snap_scalp(head_mask, direction):
    d = direction / np.linalg.norm(direction); last = AC.astype(float)
    for r in np.arange(0, float(np.linalg.norm(GDIM)), 1.0):
        p = np.round(AC + r * d).astype(int)
        if not all(0 <= p[i] < GDIM[i] for i in range(3)): break
        if head_mask[p[0], p[1], p[2]]: last = p.astype(float)
    return last


def normalize_head(head, force=False):
    """Rigid-resample the phantom into the common grid; save optical + sources. Returns paths."""
    import SimpleITK as sitk
    ds = dataset_of(head); flip = DATASET_FLIP[ds]
    if flip is None:
        raise RuntimeError(f"DATASET_FLIP['{ds}'] unset; run --determine first")
    os.makedirs(OUT_DATASET, exist_ok=True)
    outp = os.path.join(OUT_DATASET, f"{head}_v11mni_F810.mat")
    if os.path.isfile(outp) and not force:
        return outp
    (T, metric), lab = register_head(head, flip)
    lab_src = lab[::-1].copy() if flip else lab
    ref = sitk.Image(int(GDIM[0]), int(GDIM[1]), int(GDIM[2]), sitk.sitkFloat32)
    ref.SetSpacing((1, 1, 1)); ref.SetOrigin(tuple(float(-x) for x in AC))
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    res = sitk.Resample(_to_sitk(lab_src.astype(np.float32), (0, 0, 0)), ref, T,
                        sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32)
    labC = np.round(sitk.GetArrayFromImage(res).transpose(2, 1, 0)).astype(np.uint8)
    prop = label_to_optical(labC)                      # (224,256,288,4) common grid
    head_mask = labC > 0
    M = _load_mni()
    src = {n: snap_scalp(head_mask, M["dirs"][n]) for n in NAMES}
    occ = np.argwhere(head_mask); clip = bool((occ.min(0) <= 0).any() or (occ.max(0) >= np.array(GDIM) - 1).any())
    # PERSIST THE TRANSFORM. The registration is NOT reproducible: _rigid multi-starts
    # with SimpleITK random metric sampling, and re-running register_head on the same
    # phantom yields a different transform each time (observed metrics -0.303/-0.314/
    # -0.315 vs the -0.273 stored here, ~16 voxel centroid drift, Dice 0.80). Anything
    # that must land in the SAME MNI space as this phantom (e.g. a co-registered PET)
    # has to REUSE these parameters, never recompute them.
    sitk.WriteTransform(T, os.path.join(OUT_DATASET, f"{head}_v11mni_rigid.tfm"))
    sio.savemat(outp, {PROP_KEY: prop, "AC": AC, "flip": flip, "metric": metric,
                       "tfm_params": np.array(T.GetParameters(), dtype=np.float64),
                       "tfm_fixed": np.array(T.GetFixedParameters(), dtype=np.float64),
                       "srcpos": np.array([src[n] for n in NAMES]),
                       "electrodes": NAMES}, do_compression=True)
    print(f"[norm] {head} flip={flip} metric={metric:.4f} clip={clip} -> {os.path.basename(outp)}", flush=True)
    return outp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--determine", nargs="+", help="heads to determine per-dataset L-R flip")
    ap.add_argument("--normalize", nargs="+", help="heads to rigid-resample into common grid")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.determine:
        res = determine_flip(a.determine)
        # aggregate by dataset
        byds = {}
        for h, r in res.items():
            byds.setdefault(dataset_of(h), []).append(r["flip"])
        print("\nper-dataset majority flip:")
        for ds, fl in byds.items():
            print(f"  {ds}: {sum(fl)}/{len(fl)} flip -> {'True' if sum(fl)>len(fl)/2 else 'False'}")
    if a.normalize:
        for h in a.normalize:
            normalize_head(h, force=a.force)
