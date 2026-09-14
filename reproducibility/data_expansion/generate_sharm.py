#!/usr/bin/env python3
"""
Ingest the SHARM head dataset (Rashed et al., IXI-derived, 256^3, 1 mm iso, 16
tissue labels) into the FM-INR pipeline as native F810 property volumes -- the
SHARM analogue of generate_brainweb.py.

Per subject:
  1. load `model` (uint8 16-label) from SHARMxxx.mat (nested MATLAB struct),
  2. auto-orient native -> canonical (+X=L-R, +Y=anterior, +Z=superior) by an
     anatomical fingerprint (bilateral-symmetry / brain-vs-neck / eye COMs), as a
     PROPER rotation (det=+1). SHARM is already canonical, so this is normally the
     identity -- but it is computed+verified per head, catching any outlier,
  3. crop to the head bounding box, pad uniformly with PAD air voxels (matches scb/
     bw: trims excess air + guarantees MCX source standoff),
  4. map the 16 labels -> (mu_a,mu_s,g,n) via the FULL F810 9-tissue LUT
     (optical_tissue_lut.sharm_labels_to_property_volume) -- same optical scale as
     the existing scb/bw heads,
  5. write the standardized property volume in the exact repo layout
     ({sid}_copmri_withHermiteF810.mat, key PROP_KEY, (X,Y,Z,4) float32, compressed)
     into dataset_sharm810/ so it flows straight through run_sharm_mcx.py.

Because the orientation transform is the identity, the written volumes are already
"standardized" (native == canonical); no separate standardize_orientation pass is
needed. A sharm_manifest.json (head ids + a stratified train/val/test split) is
emitted alongside.

Idempotent.  Run:  python generate_sharm.py           (all 196)
                   python generate_sharm.py --only SHARM026
"""
import os, sys, glob, json, argparse
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import repro_config as RC
from optical_tissue_lut import (sharm_labels_to_property_volume,
                                 SHARM_LABEL_TO_P0ROW, assert_lut_in_phys_range)
from generate import assert_in_range
import v2_manifest as M

PAD    = 32                    # air voxels each side (>=10 clear + 15 margin standoff)
WLKEY  = "F810"
PROP_KEY = M.PROP_KEY          # "vol_prop_eye_aseg" -- same key the MCX core reads
SHARM_SRC = RC.SHARM_DIR
SHARM_DATASET = RC.SHARM_DATASET_DIR


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_model(path):
    """Return the uint8 16-label `model` volume from a SHARM .mat (nested struct)."""
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    d = {k: v for k, v in d.items() if not k.startswith("__")}
    (s,) = d.values()                                   # single top-level struct
    return np.asarray(s.model).astype(np.uint8)


# --------------------------------------------------------------------------- #
# orientation: native -> canonical (+X=L-R, +Y=anterior, +Z=superior), det=+1
# --------------------------------------------------------------------------- #
def native_axis_roles(m):
    """Detect (perm, sign) mapping native axes -> canonical from anatomy.

    perm[c] = native axis that becomes canonical axis c (0=X/LR,1=Y/AP,2=Z/SI);
    sign[c] = +1/-1 flip so +Y=anterior, +Z=superior. LR sign is chosen last to
    force a PROPER rotation (det=+1).
    """
    occ = m > 0
    sym = [np.logical_and(occ, np.flip(occ, ax)).sum()
           / np.logical_or(occ, np.flip(occ, ax)).sum() for ax in range(3)]
    lr = int(np.argmax(sym))                            # bilateral-symmetry axis
    def com(mask):
        idx = np.argwhere(mask)
        return idx.mean(0) if len(idx) else np.full(3, np.nan)
    brain = com((m == 10) | (m == 11))                  # cerebrum -> superior
    neck  = com(m == 3)                                 # muscle   -> inferior
    eyes  = com((m == 14) | (m == 15))                  # vitreous+lens -> anterior
    rest = [a for a in range(3) if a != lr]
    # SI = the remaining axis where brain/neck separate most; AP = the other
    si = rest[int(np.argmax([abs(brain[a] - neck[a]) for a in rest]))]
    ap = rest[0] if rest[1] == si else rest[1]
    si_sign = 1 if brain[si] - neck[si] > 0 else -1     # +Z = superior
    ap_sign = 1 if eyes[ap] - brain[ap] > 0 else -1     # +Y = anterior
    perm = [lr, ap, si]                                 # canonical X,Y,Z <- native
    sign = [1, ap_sign, si_sign]
    # force det=+1: parity(perm) * prod(sign) must be +1; LR sign is free
    parity = _perm_parity(perm)
    if parity * (sign[1] * sign[2]) < 0:
        sign[0] = -1
    return perm, sign


def _perm_parity(p):
    seen = [False] * 3; par = 1
    for i in range(3):
        if seen[i]:
            continue
        j, ln = i, 0
        while not seen[j]:
            seen[j] = True; j = p[j]; ln += 1
        if ln % 2 == 0:
            par = -par
    return par


def reorient(vol, perm, sign):
    """Apply (perm, sign) to a (X,Y,Z[,C]) array -> canonical frame."""
    extra = tuple(range(3, vol.ndim))
    out = np.transpose(vol, tuple(perm) + extra)
    for c in range(3):
        if sign[c] < 0:
            out = np.flip(out, axis=c)
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- #
# crop + pad (identical policy to generate_brainweb)
# --------------------------------------------------------------------------- #
def crop_pad(label):
    occ = label > 0
    bb = np.argwhere(occ); lo = bb.min(0); hi = bb.max(0) + 1
    c = label[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return np.pad(c, PAD, mode="constant", constant_values=0)


# --------------------------------------------------------------------------- #
# id / split
# --------------------------------------------------------------------------- #
def sid_to_head(sid):
    """'SHARM026' -> 'sh026' (repo head-id style, disjoint from scb/bw)."""
    return "sh" + sid.replace("SHARM", "").zfill(3)


def std_mat(head, out_dir=SHARM_DATASET):
    return os.path.join(out_dir, f"{head}_copmri_withHermiteF{M.WL}.mat")


def stratified_split(heads, val_frac=0.1, test_frac=0.1, seed=0):
    """Deterministic subject-disjoint train/val/test split over SHARM heads."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(heads)); rng.shuffle(idx)
    n = len(heads); nval = max(1, int(round(val_frac * n)))
    ntest = max(1, int(round(test_frac * n)))
    test = sorted(heads[i] for i in idx[:ntest])
    val  = sorted(heads[i] for i in idx[ntest:ntest + nval])
    train = sorted(heads[i] for i in idx[ntest + nval:])
    return train, val, test


# --------------------------------------------------------------------------- #
def process_one(path, out_dir, overwrite=False, verbose=True):
    sid = os.path.splitext(os.path.basename(path))[0]      # SHARM026
    head = sid_to_head(sid)                                # sh026
    out = std_mat(head, out_dir)
    if os.path.isfile(out) and not overwrite:
        if verbose: print(f"[skip] {head} (exists)")
        return head, None
    m = load_model(path)
    perm, sign = native_axis_roles(m)
    m = reorient(m, perm, sign)
    lab = crop_pad(m)
    prop = sharm_labels_to_property_volume(lab, WLKEY)     # (X,Y,Z,4) physical
    assert_in_range(prop, head)
    os.makedirs(out_dir, exist_ok=True)
    sio.savemat(out, {PROP_KEY: prop.astype(np.float32)}, do_compression=True)
    occ = lab > 0; ext = (np.argwhere(occ).max(0) - np.argwhere(occ).min(0) + 1)
    ident = (perm == [0, 1, 2] and sign == [1, 1, 1])
    if verbose:
        print(f"[write] {head} <- {sid}  shape={prop.shape[:3]} head_ext={ext.tolist()} "
              f"orient={'identity' if ident else f'perm{perm} sign{sign}'}", flush=True)
    return head, dict(sid=sid, shape=list(prop.shape[:3]),
                      orient_perm=list(perm), orient_sign=list(sign))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SHARM_SRC, help="dir of SHARMxxx.mat")
    ap.add_argument("--out-dir", default=SHARM_DATASET)
    ap.add_argument("--only", default=None, help="single id e.g. SHARM026")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    assert_lut_in_phys_range()
    files = sorted(glob.glob(os.path.join(a.src, "SHARM*.mat")))
    if a.only:
        files = [f for f in files if a.only in f]
    heads, info = [], {}
    for f in files:
        head, meta = process_one(f, a.out_dir, a.overwrite)
        heads.append(head)
        if meta: info[head] = meta
    # (re)write manifest over the full set present in out_dir
    all_heads = sorted(sid_to_head(os.path.splitext(os.path.basename(p))[0])
                       for p in glob.glob(os.path.join(a.src, "SHARM*.mat")))
    train, val, test = stratified_split(all_heads)
    manifest = dict(wavelength=M.WL, prop_key=PROP_KEY, n_heads=len(all_heads),
                    dataset_dir=a.out_dir, labels=SHARM_LABEL_TO_P0ROW,
                    heads=all_heads, split=dict(train=train, val=val, test=test),
                    processed_this_run=info)
    mpath = os.path.join(a.out_dir, "sharm_manifest.json")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[sharm] DONE  processed={len(info)} present={len(all_heads)}  "
          f"split train/val/test = {len(train)}/{len(val)}/{len(test)}  -> {mpath}")


if __name__ == "__main__":
    main()
