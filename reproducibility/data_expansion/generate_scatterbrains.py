"""
Ingest the scatterBrains 16-subject head database into the dataset.

Per subject: load the 256^3 labelled volume (0 air / 1 scalp / 2 skull / 3 CSF /
4 GM / 5 WM), map labels -> (mu_a,mu_s,g,n) via the authoritative F-series LUT for
each wavelength, write the property volume in the discover_scenes layout, then run
MCX for the structured STARTER_GRID of sources (placed on the head surface) to get
fluence GT.  These heads join the FM fine-tuning + INR training sets.

Idempotent (skips existing).  Run:
    python -m data_expansion.generate_scatterbrains
"""
import os
import sys
import json

import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "INR_design"))
import repro_config as RC
import inr_dataset as D
from data_expansion.optical_tissue_lut import labels_to_property_volume, assert_lut_in_phys_range
from data_expansion.illumination import STARTER_GRID, place_source, illum_tag
from data_expansion.solvers import solve_mcx
from data_expansion.generate import assert_in_range

SCB_DIR = RC.SCATTERBRAINS_DIR
WL_MAP = {"810": "F810", "1064": "F1064"}     # 起步集: 2 wavelengths
NPHOTON = 5e6
GPUID = 1


def _write_property(sid, wlnum, prop):
    p = os.path.join(D.DATASET_DIR, f"{sid}_copmri_withHermiteF{wlnum}.mat")
    sio.savemat(p, {"vol_prop_eye_aseg": prop.astype(np.float32)}, do_compression=True)
    return p


def _scene_io(sid, wlnum, tag):
    wld = os.path.join(D.SIM_DIR, f"wl{wlnum}")
    os.makedirs(wld, exist_ok=True)
    flu = os.path.join(wld, f"fluence_{sid}_F{wlnum}_{tag}.mat")
    meta = os.path.join(wld, f"meta_{sid}_F{wlnum}_{tag}.json")
    return flu, meta


def main():
    assert_lut_in_phys_range()
    n_prop, n_flu = 0, 0
    for i in range(1, 17):
        sid = f"scb{i:02d}"
        volf = os.path.join(SCB_DIR, f"Subject{i:02d}", f"Subject{i:02d}_volume.mat")
        if not os.path.isfile(volf):
            print(f"[scb] {sid}: volume missing, skip", flush=True)
            continue
        vol = sio.loadmat(volf)["vol"]                       # (256,256,256) labels 0..5
        occupancy = (vol > 0)
        for wlnum, wlkey in WL_MAP.items():
            prop = labels_to_property_volume(vol, wlkey)     # (256,256,256,4) physical
            assert_in_range(prop, f"{sid}_F{wlnum}")
            _write_property(sid, wlnum, prop); n_prop += 1
            for c in STARTER_GRID:
                tag = illum_tag(c["theta"], c["phi"], c["radius"])
                flu, meta = _scene_io(sid, wlnum, tag)
                if os.path.isfile(flu) and os.path.isfile(meta):
                    continue
                illum = place_source(occupancy, c["theta"], c["phi"], c["radius"])
                phi = solve_mcx(prop, illum, nphoton=NPHOTON, gpuid=GPUID)
                sio.savemat(flu, {"fluence": phi.astype(np.float32)}, do_compression=True)
                with open(meta, "w") as f:
                    json.dump(dict(sample_id=sid, wavelength=wlnum,
                                   srcpos=[float(x) for x in illum.srcpos],
                                   srcdir=[float(x) for x in illum.dir_unit()],
                                   radius_mm=float(illum.radius_mm),
                                   theta=c["theta"], phi=c["phi"], kind="scatterbrains"), f, indent=2)
                n_flu += 1
                print(f"[scb] {sid} F{wlnum} {tag}: phi max={phi.max():.2e} "
                      f"src={np.round(illum.srcpos,0).tolist()}", flush=True)
    print(f"\n[scb] DONE — {n_prop} property volumes, {n_flu} fluence scenes", flush=True)


if __name__ == "__main__":
    main()
