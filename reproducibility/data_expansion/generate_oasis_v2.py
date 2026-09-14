#!/usr/bin/env python3
"""OASIS-3 T1w + TOF-angio + GRACE -> 9-tissue head phantom (FM-INR native format).

HYBRID segmentation, one source per tissue, each chosen on its merits:

  CSF / GM / WM        <- ANTs Atropos (EM + MRF) inside the deepbet brain mask.
      Kept deliberately instead of GRACE's brain classes: GRACE over-calls CSF on
      this elderly cohort (CSF 628k ~ GM 622k > WM 407k on OAS30884) and CSF's
      mu_a is ~1/10 of GM's, so that bias would land straight in photon transport.

  skull / skin / fat / muscle / air / eyes  <- GRACE (UNETR trained on OLDER ADULTS,
      github.com/lab-smile/GRACE). Replaces two things that had no anatomical prior:
      the Otsu+6mm-distance skull heuristic, and the "everything left over is scalp"
      catch-all. GRACE resolves cortical vs cancellous bone and skin/fat/muscle;
      the two bone classes are collapsed to the single SKULL LUT row (no validated
      diploe optical value available) and eyes -> CSF (vitreous ~99% water, matching
      the existing SHARM vitreous->CSF precedent).

  vessel               <- TOF angio threshold. A dedicated angiographic sequence
      beats GRACE's incidental `blood` class (~5.7k voxels vs TOF's ~16k).

labels {0 air, 1 skin, 2 skull, 3 CSF, 4 GM, 5 WM, 6 vessel, 7 fat, 8 muscle}
-> optical via the FULL 9-row F810 table (same scale as SHARM/BrainWeb).

GRACE runs its own canonical-RAS + 1mm preprocessing, which lands on the SAME grid
to_iso1mm produces -- verified across heads (identical shapes, head-mask Dice
0.97-0.98, GRACE brain 100% inside our head mask), so no resampling is needed. The
grid and Dice are both re-checked at runtime and the run aborts on mismatch.

T1 and TOF share a world frame (same session/scanner), so the TOF is mapped onto the
T1 grid analytically through the nibabel affines -- no registration. The TOF covers
only a slab (~circle of Willis), so vessels exist only where it imaged: honest, not a bug.

  python generate_oasis_v2.py --t1 <T1w.nii.gz> --tof <angio.nii.gz> \
         --grace-seg <grace_seg.nii.gz> --sid oa01
"""
import os, sys, argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import repro_config as RC
from optical_tissue_lut import (oasis9_labels_to_property_volume,
                                OASIS9_LABEL_NAME, GRACE_TO_OASIS9)
import v2_manifest as M
import generate_oasis as G          # reuse to_iso1mm / head_mask / N4 / deepbet / kmeans

OUT_DIR = RC.OASIS_DATASET_DIR
SKULL_MM = 6.0
CSF_MM = 2.0
SKIN_MM = 2.0        # outermost head rim treated as skin
VESSEL_PCT = 99.3    # TOF intensity percentile (within its FOV) above which = vessel
VESSEL_MIN_CC = 20   # drop vessel blobs smaller than this many voxels (noise)


def resample_to_iso_grid(src_path, t1_path, iso_shape):
    """Resample a NIfTI onto the T1 iso-1mm grid produced by G.to_iso1mm().

    CAREFUL: G.to_iso1mm builds its sitk image with spacing only (origin 0,
    identity direction), so the reference lives in a FAKE world frame offset from
    the NIfTI affine by the T1 origin (~100 mm here). Resampling the TOF through
    sitk against that reference silently translates it out of the head. So we map
    analytically through the real nibabel affines instead.

    G.to_iso1mm resamples with out-spacing 1 and the source spacing = zoom, hence
        iso index (p,q,r)  <->  canonical-T1 voxel (p/zx, q/zy, r/zz)
    and therefore
        world = A_t1 @ [p/zx, q/zy, r/zz, 1],   tof_vox = inv(A_tof) @ world
    """
    t1 = nib.as_closest_canonical(nib.load(t1_path))
    src = nib.as_closest_canonical(nib.load(src_path))
    zoom = np.array(t1.header.get_zooms()[:3], dtype=float)

    p, q, r = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in iso_shape], indexing="ij")
    t1vox = np.stack([p / zoom[0], q / zoom[1], r / zoom[2], np.ones_like(p)], axis=-1)
    Mx = np.linalg.inv(src.affine) @ t1.affine                 # T1 voxel -> src voxel
    srcvox = t1vox @ Mx.T
    coords = np.stack([srcvox[..., 0], srcvox[..., 1], srcvox[..., 2]], axis=0)
    arr = np.asarray(src.dataobj, dtype=np.float32)
    out = ndi.map_coordinates(arr, coords, order=1, mode="constant", cval=0.0)
    return out.astype(np.float32)


def segment_vessels(tof, head):
    """TOF angio -> boolean vessel mask on the T1 grid.

    Threshold is taken INSIDE the TOF field of view only; outside it we simply do
    not know, and claiming 'no vessel' everywhere else is the honest default.
    """
    fov = (tof > 0) & head
    if fov.sum() < 1e4:
        print("    [vessel] TOF FOV too small -- no vessels", flush=True)
        return np.zeros_like(head), 0.0
    thr = float(np.percentile(tof[fov], VESSEL_PCT))
    ves = fov & (tof >= thr)
    lab, n = ndi.label(ves)
    if n:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        keep = np.isin(lab, np.flatnonzero(sizes >= VESSEL_MIN_CC))
        ves = keep
    cov = float(fov.sum())
    print(f"    [vessel] TOF FOV {int(cov):,} vox  thr={thr:.1f}  "
          f"vessels {int(ves.sum()):,} ({100*ves.sum()/max(cov,1):.2f}% of FOV)", flush=True)
    return ves, cov


def atropos_brain(v, brain):
    """CSF/GM/WM inside the brain mask via ANTs Atropos (EM + MRF).

    Returns a uint8 array coded 3=CSF, 4=GM, 5=WM (0 elsewhere). Atropos numbers
    its classes 1..3 in ascending intensity, which on T1 is exactly CSF<GM<WM.
    """
    import ants
    seg = ants.atropos(a=ants.from_numpy(np.ascontiguousarray(v)),
                       x=ants.from_numpy(np.ascontiguousarray(brain.astype(np.float32))),
                       i="kmeans[3]", m="[0.2,1x1x1]", c="[5,0]")
    S = seg["segmentation"].numpy()
    out = np.zeros(v.shape, np.uint8)
    out[S == 1] = 3; out[S == 2] = 4; out[S == 3] = 5
    return out


def segment9(v, tof, head, brain, grace):
    """HYBRID -> uint8 {0 air,1 skin,2 skull,3 CSF,4 GM,5 WM,6 vessel,7 fat,8 muscle}.

    One source per tissue, chosen on its merits:
      * CSF/GM/WM  -- ANTs Atropos inside the deepbet brain mask (classic EM+MRF).
        NOT GRACE: GRACE over-calls CSF on this elderly cohort, and CSF's mu_a is ~1/10
        of GM's, so that error would propagate straight into photon transport.
      * skull / skin / fat / muscle / air / eyes -- GRACE (UNETR trained on older adults),
        replacing the old Otsu+distance skull heuristic and the "everything left over is
        scalp" catch-all, neither of which had any anatomical prior.
      * vessel -- TOF angio (a dedicated angiographic sequence beats an incidental label).
    """
    lab = np.zeros(v.shape, np.uint8)

    # --- 1. brain interior: Atropos (unchanged classic path) ---
    lab_b = atropos_brain(v, brain)
    lab[brain] = lab_b[brain]
    n = {k: int((lab == k).sum()) for k in (3, 4, 5)}
    print(f"    [atropos] CSF {n[3]:,}  GM {n[4]:,}  WM {n[5]:,}   "
          f"CSF/GM={n[3]/max(n[4],1):.2f}  GM/WM={n[4]/max(n[5],1):.2f}", flush=True)

    # --- 2. everything outside the brain mask: GRACE ---
    outside = head & ~brain
    gmapped = np.zeros(v.shape, np.uint8)
    for gl, ol in GRACE_TO_OASIS9.items():
        if ol is None:
            continue
        gmapped[grace == gl] = ol
    lab[outside] = gmapped[outside]
    ng = {k: int(((lab == k) & outside).sum()) for k in (1, 2, 7, 8)}
    print(f"    [grace]  skin {ng[1]:,}  skull {ng[2]:,}  fat {ng[7]:,}  muscle {ng[8]:,}",
          flush=True)

    # --- 3. fallback for head voxels GRACE called background (head masks differ ~3%) ---
    gap = outside & (lab == 0) & (grace == 0)
    if gap.any():
        src = (lab > 0)
        _, (ix, iy, iz) = ndi.distance_transform_edt(~src, return_indices=True)
        lab[gap] = lab[ix[gap], iy[gap], iz[gap]]      # nearest already-labelled tissue
        print(f"    [fill]   {int(gap.sum()):,} head voxels filled from nearest label",
              flush=True)

    # --- 4. vessels last: they override whatever tissue they run through ---
    ves, _ = segment_vessels(tof, head)
    lab[ves] = 6
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True); ap.add_argument("--tof", required=True)
    ap.add_argument("--grace-seg", required=True,
                    help="GRACE 11-class label NIfTI for this subject (external/GRACE/infer_single.py)")
    ap.add_argument("--sid", required=True); ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--save-labels", action="store_true", help="also write the uint8 label volume")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"[oasis9] {a.sid} <- T1 {os.path.basename(a.t1)} + TOF {os.path.basename(a.tof)} "
          f"+ GRACE {os.path.basename(a.grace_seg)}", flush=True)

    v, iso = G.to_iso1mm(a.t1)
    print(f"  native iso1mm shape={v.shape}", flush=True)
    head = G.head_mask(v); print(f"  head voxels   {int(head.sum()):,}", flush=True)

    n4 = sitk.N4BiasFieldCorrectionImageFilter(); n4.SetMaximumNumberOfIterations([50, 50, 30])
    hm = sitk.GetImageFromArray(np.ascontiguousarray(head.astype(np.uint8).transpose(2, 1, 0)))
    hm.CopyInformation(iso)
    v = sitk.GetArrayFromImage(n4.Execute(sitk.Cast(iso, sitk.sitkFloat32), hm)) \
            .transpose(2, 1, 0).astype(np.float32)
    print("  N4 bias-corrected (head-masked)", flush=True)

    tof = resample_to_iso_grid(a.tof, a.t1, v.shape)
    # Alignment check on SIGNAL-BEARING voxels only. `tof>0` is useless here: linear
    # interpolation smears TOF background noise across the whole FOV box, which
    # includes air around the head (~60% inside-head even when perfectly aligned).
    nz = tof > 0
    sig = tof >= G._otsu(tof[nz])
    ov = float((sig & head).sum()) / max(sig.sum(), 1)
    print(f"  TOF resampled onto T1 grid; signal voxels {int(sig.sum()):,}, "
          f"{100*ov:.1f}% inside head", flush=True)
    if ov < 0.90:
        raise SystemExit(f"[oasis9] TOF/T1 ALIGNMENT FAILED: only {100*ov:.1f}% of TOF signal "
                         "falls inside the head mask (expected >90%). Refusing to write.")

    brain = G.brain_mask_deepbet(a.t1, iso) & head & (v > 0)
    print(f"  brain voxels  {int(brain.sum()):,}", flush=True)

    # GRACE runs its own canonical-RAS + 1mm preprocessing, which lands on the SAME grid
    # to_iso1mm produces (verified: identical shapes, head-mask Dice 0.97-0.98, GRACE brain
    # 100% inside our head mask). A shape mismatch means that assumption broke -- refuse
    # rather than silently mis-assign every extracranial tissue.
    grace = np.asarray(nib.load(a.grace_seg).dataobj).astype(np.uint8)
    if grace.shape != v.shape:
        raise SystemExit(f"[oasis9] GRACE GRID MISMATCH: grace {grace.shape} vs ours {v.shape}. "
                         "Refusing to write.")
    gh = grace > 0
    dice = 2 * (gh & head).sum() / max(gh.sum() + head.sum(), 1)
    print(f"  GRACE seg loaded; head Dice vs ours = {dice:.3f}", flush=True)
    if dice < 0.85:
        raise SystemExit(f"[oasis9] GRACE/our head masks disagree (Dice {dice:.3f} < 0.85). "
                         "Refusing to write.")

    lab = segment9(v, tof, head, brain, grace)
    nv = {k: int((lab == k).sum()) for k in OASIS9_LABEL_NAME if k}
    for k, n in OASIS9_LABEL_NAME.items():
        if k: print(f"    {k} {n:7} {nv[k]:>9,}", flush=True)

    # ---- sanity gate ----
    # NOTE the CSF/GM band is deliberately WIDE (0.3-1.5): this cohort is 77 +- 8 y
    # and CSF/GM rises steeply with age and atrophy (SHARM at age 32 sits at 0.46,
    # a healthy 86 y-old here at 0.82, an AD 86 y-old at 0.90). A tight band tuned
    # on young heads would reject perfectly good elderly segmentations.
    hv = int(head.sum()); bv = int(brain.sum()); bad = []
    ratio = nv[3] / max(nv[4], 1)
    if not (1.0e6 <= hv <= 8.0e6):        bad.append(f"head {hv:,} outside 1-8M")
    if not (0.8e6 <= bv <= 2.2e6):        bad.append(f"brain {bv:,} outside 0.8-2.2M")
    if not (0.3e6 <= nv[5] <= 0.9e6):     bad.append(f"WM {nv[5]:,} outside 0.3-0.9M")
    if not (0.3e6 <= nv[4] <= 1.1e6):     bad.append(f"GM {nv[4]:,} outside 0.3-1.1M")
    if nv[5] < 0.4 * nv[4]:               bad.append(f"WM/GM={nv[5]/max(nv[4],1):.2f} <0.4 (classes swapped?)")
    if not (0.3 <= ratio <= 1.5):         bad.append(f"CSF/GM={ratio:.2f} outside 0.3-1.5")
    if not (0.15e6 <= nv[2] <= 1.2e6):    bad.append(f"skull {nv[2]:,} outside 0.15-1.2M")
    soft = nv[1] + nv[7] + nv[8]
    if soft < 0.3e6:                      bad.append(f"soft tissue skin+fat+muscle {soft:,} <0.3M")
    for k in (1, 7, 8):
        if nv[k] == 0:                    bad.append(f"{OASIS9_LABEL_NAME[k]} empty (GRACE mapping broken?)")
    if nv[6] == 0:                        bad.append("NO vessels segmented (TOF misaligned/empty?)")
    if nv[6] > 0.15e6:                    bad.append(f"vessel {nv[6]:,} >150k (threshold too low?)")
    if bad:
        raise SystemExit("[oasis9] SEGMENTATION SANITY FAILED -- not written:\n   " + "\n   ".join(bad))

    prop = oasis9_labels_to_property_volume(lab, "F810")
    import scipy.io as sio
    dst = os.path.join(a.out, f"{a.sid}_copmri_withHermiteF810.mat")
    sio.savemat(dst, {M.PROP_KEY: prop.astype(np.float32)}, do_compression=True)
    print(f"[oasis9] wrote {dst}  shape={prop.shape}", flush=True)
    # stats line for the batch QC sweep to grep
    print(f"[stats] {a.sid} head={hv} brain={bv} CSF={nv[3]} GM={nv[4]} WM={nv[5]} "
          f"skin={nv[1]} fat={nv[7]} muscle={nv[8]} skull={nv[2]} vessel={nv[6]} csf_gm={ratio:.4f} "
          f"gm_wm={nv[4]/max(nv[5],1):.4f}", flush=True)
    if a.save_labels:
        lp = os.path.join(a.out, f"{a.sid}_labels9.npy"); np.save(lp, lab)
        np.save(os.path.join(a.out, f"{a.sid}_t1_iso.npy"), v.astype(np.float32))
        print(f"[oasis9] wrote {lp} (+ T1) for QC", flush=True)


if __name__ == "__main__":
    main()
