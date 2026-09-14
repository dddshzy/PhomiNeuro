#!/usr/bin/env python3
"""
Stage-2: ingest BrainWeb-20 into the dataset as native property volumes.

Per subject: load the crisp model, downsample 0.5 mm -> 1 mm (match scb), crop to
the head bounding box and pad uniformly with PAD voxels of air (both trims excess
air and guarantees the MCX source standoff ~25 vox, since BrainWeb's A-P occupancy
fills the native FOV), map the 12 tissue codes -> (mu_a,mu_s,g,n) via the FULL
F810 9-tissue LUT (vessel/fat/muscle/skin kept distinct), and write the native
property volume in the same layout as scb (dataset/bwNN_..._F810.mat).

These NATIVE volumes are then orientation-standardized by standardize_orientation.py
(BW_ORIENT) into dataset_v2_head810/, exactly like scb.

Idempotent.  Run:  python generate_brainweb.py   (optionally --only bw01)
"""
import os, sys, argparse
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import fetch_brainweb as FB
from optical_tissue_lut import bw_labels_to_property_volume, assert_lut_in_phys_range
from generate import assert_in_range
import v2_manifest as M

PAD = 32                      # air voxels on every side (>= 10 clear + 15 margin standoff)
WLKEY = "F810"


def crop_pad(label):
    """Crop to occupancy bbox, then pad uniformly with PAD voxels of air (code 0)."""
    occ = label > 0
    bb = np.argwhere(occ); lo = bb.min(0); hi = bb.max(0) + 1
    c = label[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return np.pad(c, PAD, mode="constant", constant_values=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="single head id e.g. bw01")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    assert_lut_in_phys_range()
    files = FB.subject_files(); ids = FB.bw_head_ids()
    n = 0
    for sid, f in zip(ids, files):
        if a.only and sid != a.only:
            continue
        out = M.src_mat(sid)                              # dataset/bwNN_..._F810.mat
        if os.path.isfile(out) and not a.overwrite:
            print(f"[skip] {sid} (exists)"); continue
        lab = crop_pad(FB.load_label_1mm(f))             # (X,Y,Z) 1 mm, padded
        prop = bw_labels_to_property_volume(lab, WLKEY)  # (X,Y,Z,4) physical
        assert_in_range(prop, sid)
        sio.savemat(out, {M.PROP_KEY: prop.astype(np.float32)}, do_compression=True)
        occ = lab > 0; ext = (np.argwhere(occ).max(0) - np.argwhere(occ).min(0) + 1)
        print(f"[write] {sid} -> {os.path.basename(out)}  shape={prop.shape[:3]} "
              f"head_ext={ext.tolist()}", flush=True)
        n += 1
    print(f"[brainweb] DONE {n} native property volumes")


if __name__ == "__main__":
    main()
