#!/usr/bin/env python3
"""OASIS-3 T1w -> 5-layer head phantom in the FM-INR native format.

WHY an approximate segmentation is scientifically valid here: the purpose is to test surrogate
GENERALISATION to an unseen head geometry. The MCX simulation run on whatever phantom we build IS
the ground truth for that phantom, so the surrogate is judged on reproducing MC on a novel, layered,
anatomically-plausible head -- it does not require the segmentation to be a perfect reconstruction
of the real subject's skull. (Stated explicitly because no FSL/FreeSurfer/SimNIBS is available here;
only nibabel/SimpleITK/nilearn/scipy.)

Pipeline: T1 -> canonical RAS -> 1 mm iso -> head mask -> MNI-propagated brain mask ->
CSF/GM/WM by intensity inside brain -> skull shell + scalp outside -> labels {1 scalp, 2 skull,
3 CSF, 4 GM, 5 WM} -> optical (F810 5-layer LUT) -> {sid}_copmri_withHermiteF810.mat (PROP_KEY).
Downstream (mni_normalize -> illumination -> MCX) is unchanged.

  python generate_oasis.py --t1 <T1.nii.gz> --sid oa01
"""
import os, sys, argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import repro_config as RC
from optical_tissue_lut import labels_to_property_volume
import v2_manifest as M

OUT_DIR = RC.OASIS_DATASET_DIR
SKULL_MM = 6.0          # max distance from the brain surface a dark voxel may be and still be skull
CSF_MM = 2.0            # subarachnoid CSF rim grown just outside the brain-tissue mask


def to_iso1mm(t1_path):
    """Canonical RAS + 1 mm isotropic resample -> (arr, sitk image)."""
    im = nib.as_closest_canonical(nib.load(t1_path))
    arr = np.asarray(im.dataobj, dtype=np.float32)
    zoom = np.array(im.header.get_zooms()[:3], dtype=float)
    s = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))
    s.SetSpacing(tuple(float(z) for z in zoom))
    new_size = [int(round(sz * sp)) for sz, sp in zip(s.GetSize(), s.GetSpacing())]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing((1.0, 1.0, 1.0)); r.SetSize(new_size)
    r.SetOutputDirection(s.GetDirection()); r.SetOutputOrigin(s.GetOrigin())
    r.SetInterpolator(sitk.sitkLinear)
    out = r.Execute(s)
    return sitk.GetArrayFromImage(out).transpose(2, 1, 0).astype(np.float32), out


def head_mask(v):
    """Whole head (scalp outer surface): Otsu + close + largest CC + fill.

    Per-slice hole filling along all three axes closes narrow channels that are
    not enclosed in 3D; a final three-dimensional pass fills enclosed cavities.
    """
    thr = _otsu(v[v > 0])
    m = v > thr * 0.45                                   # generous: keep dim scalp/marrow
    m = ndi.binary_closing(m, ndi.generate_binary_structure(3, 1), iterations=3)
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    for ax in range(3):                                  # fill interior air pockets slice-wise
        filled = np.empty_like(m)
        for i in range(m.shape[ax]):
            sl = [slice(None)] * 3; sl[ax] = i
            filled[tuple(sl)] = ndi.binary_fill_holes(m[tuple(sl)])
        m = m | filled
    m = ndi.binary_fill_holes(m)
    return m


def _otsu(x, nbins=256):
    h, e = np.histogram(x, bins=nbins)
    c = h.cumsum(); mids = (e[:-1] + e[1:]) / 2
    w0 = c / c[-1]; w1 = 1 - w0
    m0 = np.cumsum(h * mids) / np.maximum(c, 1)
    mt = (h * mids).sum() / c[-1]
    m1 = (mt - m0 * w0) / np.maximum(w1, 1e-9)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(mids[np.nanargmax(var)])


def _nib2sitk(n):
    a = np.asarray(n.dataobj, dtype=np.float32)
    s = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)))
    s.SetSpacing(tuple(float(z) for z in n.header.get_zooms()[:3]))
    return s


def brain_mask_deepbet(t1_path, ref_iso):
    """Learned brain extraction (deepbet), resampled onto our 1 mm native grid.

    Hand-rolled morphology and template registration both failed here (the mask spanned the whole
    head, starving the WM class); skull-stripping needs learned priors, so use a proper tool.
    """
    import tempfile
    from deepbet import run_bet
    with tempfile.TemporaryDirectory() as td:
        mpath = os.path.join(td, "mask.nii.gz")
        run_bet([t1_path], mask_paths=[mpath], threshold=.5, n_dilate=0, no_gpu=True)
        m = nib.as_closest_canonical(nib.load(mpath))
        ms = _nib2sitk(m)
        ms = sitk.Resample(ms, ref_iso, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0,
                           ms.GetPixelID())
        return sitk.GetArrayFromImage(ms).transpose(2, 1, 0) > 0.5


def brain_mask_morph(v, head):
    """Registration-free brain extraction from T1 contrast.

    In T1 the skull is a DARK ring separating bright brain from bright scalp/fat. So: erode the head
    to drop scalp+skull, keep bright voxels, take the largest connected component (the brain core),
    then close+fill to recover ventricles/sulcal CSF. This replaced template registration, which
    proved fragile in both directions (the brain-only mask scaled onto the whole head).
    """
    thr = _otsu(v[head])
    core = ndi.binary_erosion(head, iterations=10)          # ~10 mm: past scalp+skull
    seed = (v > thr) & core
    lab, n = ndi.label(seed)
    if n == 0:
        return np.zeros_like(head)
    brain = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    brain = ndi.binary_closing(brain, iterations=4)
    brain = ndi.binary_fill_holes(brain)
    brain = ndi.binary_dilation(brain, iterations=2)
    # The brain must sit well INSIDE the scalp surface. Without this the mask leaks through gaps in
    # the dark skull ring (skull base / face) and swallows scalp+neck -- which is exactly what
    # starved the WM class (WM/GM 0.19) in the unconstrained version.
    inner = ndi.binary_erosion(head, iterations=6)
    return ndi.binary_fill_holes(brain & inner)


def brain_mask_native(iso_img):
    """Warp the MNI152 brain mask into the subject's NATIVE 1 mm space (rigid -> affine).

    Native space is used (not MNI) because the nilearn MNI152 template FOV is brain-focused and
    CROPS the neck/lower head -- which photon transport needs. Downstream, mni_normalize resamples
    into the pipeline's common grid (224,256,300, neck included) exactly as for every other dataset.
    Both stages resample rather than compose transforms (SimpleITK returns a CompositeTransform,
    which the metric cannot take a Jacobian of). Returns the brain mask on the subject grid.
    """
    from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
    tpl = load_mni152_template(resolution=1); bm = load_mni152_brain_mask(resolution=1)
    fix = sitk.Cast(iso_img, sitk.sitkFloat32)                 # SUBJECT is fixed (keeps full FOV)
    mov = sitk.Cast(_nib2sitk(tpl), sitk.sitkFloat32)          # MNI template moves
    movm = _nib2sitk(bm)                                       # MNI brain mask rides along

    def _reg(tx, fixed, moving, init_tf=None):
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(50)
        R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.20, seed=7)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=300,
                                        convergenceMinimumValue=1e-7, convergenceWindowSize=12)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4, 2, 1]); R.SetSmoothingSigmasPerLevel([2, 1, 0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInitialTransform(init_tf if init_tf is not None else tx, inPlace=False)
        out = R.Execute(fixed, moving)
        return out, R.GetMetricValue(), R.GetOptimizerIteration()

    # STAGE 1 rigid, then RESAMPLE, then STAGE 2 affine from identity in the aligned space.
    # Resampling between stages avoids composing transforms entirely (SimpleITK returns a
    # CompositeTransform, which the metric cannot take a Jacobian of).
    init = sitk.CenteredTransformInitializer(fix, mov, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    rig, m1, i1 = _reg(sitk.Euler3DTransform(), fix, mov, init)
    print(f"  [mni] rigid  MI={m1:.4f} iters={i1}", flush=True)
    stage1 = sitk.Resample(mov, fix, rig, sitk.sitkLinear, 0.0, mov.GetPixelID())
    mask1 = sitk.Resample(movm, fix, rig, sitk.sitkNearestNeighbor, 0.0, movm.GetPixelID())
    try:
        aff, m2, i2 = _reg(sitk.AffineTransform(3), fix, sitk.Cast(stage1, sitk.sitkFloat32),
                           sitk.AffineTransform(3))
        print(f"  [mni] affine MI={m2:.4f} iters={i2}", flush=True)
        maskf = sitk.Resample(mask1, fix, aff, sitk.sitkNearestNeighbor, 0.0, mask1.GetPixelID()) \
            if m2 < m1 else mask1
    except RuntimeError as e:
        print(f"  [mni] affine FAILED ({str(e).splitlines()[-1][:60]}) -> keeping rigid", flush=True)
        maskf = mask1
    return sitk.GetArrayFromImage(maskf).transpose(2, 1, 0) > 0.5


def _kmeans1d(x, k=3, iters=40, seed=0):
    """1-D k-means on intensities -> sorted centres (robust 3-class CSF/GM/WM split)."""
    rng = np.random.default_rng(seed)
    c = np.percentile(x, np.linspace(10, 90, k))
    for _ in range(iters):
        d = np.abs(x[:, None] - c[None, :]); a = d.argmin(1)
        nc = np.array([x[a == j].mean() if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c, atol=1e-4):
            break
        c = nc
    return np.sort(c)


def segment(v, head, brain):
    """-> uint8 labels {0 air,1 scalp,2 skull,3 CSF,4 GM,5 WM}."""
    lab = np.zeros(v.shape, np.uint8)
    # --- inside the brain: CSF / GM / WM by intensity (T1: CSF dark < GM < WM) ---
    # Fit the 3 classes on the brain INTERIOR only: the outer shell is dominated by partial-volume
    # (CSF/GM edge) voxels which drag the cluster centres and starve the WM class.
    interior = ndi.binary_erosion(brain, iterations=3)
    bv = v[interior if interior.sum() > 5e4 else brain]
    idx = np.random.default_rng(0).choice(bv.size, size=min(bv.size, 200000), replace=False)
    c = _kmeans1d(bv[idx], 3)                             # centres: CSF < GM < WM
    b1, b2 = (c[0] + c[1]) / 2, (c[1] + c[2]) / 2         # midpoint decision boundaries
    print(f"    [tissue] centres CSF/GM/WM = {c[0]:.1f}/{c[1]:.1f}/{c[2]:.1f}  "
          f"bounds {b1:.1f},{b2:.1f}", flush=True)
    lab[brain & (v <= b1)] = 3                            # CSF
    lab[brain & (v > b1) & (v <= b2)] = 4                 # GM
    lab[brain & (v > b2)] = 5                             # WM
    # --- subarachnoid CSF rim just outside the brain mask ---
    rim = ndi.binary_dilation(brain, iterations=int(CSF_MM)) & head & (lab == 0)
    lab[rim] = 3
    # --- skull: dark voxels within SKULL_MM of the brain/CSF envelope ---
    env = ndi.binary_dilation(brain | (lab == 3), iterations=1)
    dist = ndi.distance_transform_edt(~env)
    outer = head & (lab == 0)
    dark = v <= _otsu(v[outer]) if outer.any() else np.zeros_like(outer)
    lab[outer & dark & (dist <= SKULL_MM)] = 2
    # --- everything else in the head is scalp/soft tissue ---
    lab[head & (lab == 0)] = 1
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True); ap.add_argument("--sid", required=True)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"[oasis] {a.sid} <- {os.path.basename(a.t1)}", flush=True)
    v, iso = to_iso1mm(a.t1)
    print(f"  native iso1mm shape={v.shape}", flush=True)
    head = head_mask(v);   print(f"  head voxels   {int(head.sum()):,}", flush=True)
    # N4 bias-field correction inside the head. MRI intensity inhomogeneity otherwise makes a single
    # global intensity threshold invalid across the volume (it starved the WM class before this).
    n4 = sitk.N4BiasFieldCorrectionImageFilter(); n4.SetMaximumNumberOfIterations([50, 50, 30])
    hm = sitk.GetImageFromArray(np.ascontiguousarray(head.astype(np.uint8).transpose(2, 1, 0)))
    hm.CopyInformation(iso)
    corr = n4.Execute(sitk.Cast(iso, sitk.sitkFloat32), hm)
    v = sitk.GetArrayFromImage(corr).transpose(2, 1, 0).astype(np.float32)
    print(f"  N4 bias-corrected (head-masked)", flush=True)
    brain = brain_mask_deepbet(a.t1, iso) & head & (v > 0)
    print(f"  brain voxels  {int(brain.sum()):,}", flush=True)
    lab = segment(v, head, brain)
    names = {1: "scalp", 2: "skull", 3: "CSF", 4: "GM", 5: "WM"}
    for k, n in names.items():
        print(f"    {k} {n:6} {int((lab == k).sum()):>9,}", flush=True)
    # ---- sanity gate: a silently-broken segmentation must NOT be written ----
    nv = {k: int((lab == k).sum()) for k in names}
    hv = int(head.sum())
    bad = []
    if not (1.0e6 <= hv <= 8.0e6):            bad.append(f"head {hv:,} outside 1-8M")
    if not (0.8e6 <= int(brain.sum()) <= 2.2e6): bad.append(f"brain {int(brain.sum()):,} outside 0.8-2.2M")
    if not (0.3e6 <= nv[5] <= 0.9e6):         bad.append(f"WM {nv[5]:,} outside 0.3-0.9M")
    if not (0.3e6 <= nv[4] <= 1.1e6):         bad.append(f"GM {nv[4]:,} outside 0.3-1.1M")
    if nv[5] < 0.4 * nv[4]:                   bad.append(f"WM/GM={nv[5]/max(nv[4],1):.2f} <0.4 (T1 classes swapped?)")
    if not (0.15e6 <= nv[2] <= 1.2e6):        bad.append(f"skull {nv[2]:,} outside 0.15-1.2M")
    if nv[1] < 0.3e6:                         bad.append(f"scalp {nv[1]:,} <0.3M")
    if bad:
        raise SystemExit("[oasis] SEGMENTATION SANITY FAILED -- not written:\n   " + "\n   ".join(bad))
    prop = labels_to_property_volume(lab, "F810")
    dst = os.path.join(a.out, f"{a.sid}_copmri_withHermiteF810.mat")
    import scipy.io as sio
    sio.savemat(dst, {M.PROP_KEY: prop.astype(np.float32)}, do_compression=True)
    print(f"[oasis] wrote {dst}  shape={prop.shape}", flush=True)


if __name__ == "__main__":
    main()
