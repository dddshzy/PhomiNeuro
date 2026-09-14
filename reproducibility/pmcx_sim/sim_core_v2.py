#!/usr/bin/env python3
"""
v2 single-source fluence simulation core (810 nm, anatomical angle parametrization).

Reuses the validated building blocks from mcx_fluence_simv2 (mask fill, exact
per-voxel cfg, outward source search, pmcxcl) but:
  - reads the ORIENTATION-STANDARDIZED volume from dataset_v2_head810,
  - computes srcdir from anatomical angles (A elevation, B azimuth) in the shared
    canonical frame (anat_angles), so the same (A,B) means the same scalp location
    on every head,
  - fixes disk radius = 5 voxels (mm) and wavelength = 810 nm,
  - caches the per-head cfg once (angle-independent) under CACHE_V2,
  - saves fluence_{head}_F810_{Atag} .mat + meta.json carrying A, B, srcpos, srcdir.
"""
import os
import sys
import json
import time
import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_expansion"))
sys.path.insert(0, os.path.join(ROOT, "pmcx_sim"))

import v2_manifest as M
import anat_angles as AA
import mcx_fluence_simv2 as S        # loads pmcxcl .so; provides building blocks

WL = "810"
RADIUS_MM = 5.0
TIME_GATE = S.TIME_GATE
SEED = 29012392


def _load_std_vol(head):
    mat = sio.loadmat(M.std_mat(head))
    return mat[M.PROP_KEY].astype(np.float64)


def get_cfg(head, force_rebuild=False):
    """Per-head (angle-independent) cfg_vol / cfg_prop / mask / origin, cached."""
    os.makedirs(M.CACHE_V2, exist_ok=True)
    path = os.path.join(M.CACHE_V2, f"{head}_F{WL}_cfg.npz")
    if os.path.isfile(path) and not force_rebuild:
        d = np.load(path)
        return d["cfg_vol"], d["cfg_prop"], d["filled_mask"], d["origin"]
    vol4d = _load_std_vol(head)
    mask = S.build_filled_mask(vol4d)
    # V10: discrete few-media cfg (fixes the pmcxcl large-media-table bug). Set
    # MCX_DISCRETE=0 to fall back to the legacy buggy 1:1 builder for comparison.
    builder = S.build_cfg_exact if os.environ.get("MCX_DISCRETE") == "0" else S.build_cfg_discrete
    cfg_vol, cfg_prop, _ = builder(vol4d, mask, WL)
    origin = S.compute_mask_centroid(mask)
    np.savez_compressed(path, cfg_vol=cfg_vol, cfg_prop=cfg_prop,
                        filled_mask=mask, origin=origin)
    print(f"  [cache] built cfg -> {path}")
    return cfg_vol, cfg_prop, mask, origin


def run_one(head, A_deg, B_deg, nphoton, outdir,
            radius_mm=RADIUS_MM, gpu_id=0, force_rebuild=False, save=True, tag=None):
    """Run one (head, A, B) simulation. Returns dict with fluence + geometry meta."""
    t0 = time.time()
    cfg_vol, cfg_prop, mask, origin = get_cfg(head, force_rebuild)
    vol_size = cfg_vol.shape

    srcdir, n_out = AA.angles_to_srcdir(A_deg, B_deg)         # canonical frame
    src_pos = S.find_source_position(cfg_vol, vol_size, srcdir, origin)

    cfg = {
        "vol": cfg_vol, "prop": cfg_prop,
        "srcpos": src_pos.astype(float).tolist(),
        "srcdir": srcdir.tolist(),
        "srctype": "disk", "srcparam1": [float(radius_mm), 0.0, 0.0, 0.0],
        "tstart": 0.0, "tend": float(TIME_GATE), "tstep": float(TIME_GATE),
        "issrcfrom0": 1, "seed": SEED, "nphoton": int(nphoton), "unitinmm": 1.0,
        "outputtype": "fluence", "isreflect": 0, "isspecular": 0,
        "gpuid": gpu_id, "autopilot": 1,
    }
    res = S.pmcx.run(cfg)
    flux = res["flux"]
    if flux.ndim == 4 and flux.shape[3] == 1:
        flux = flux[:, :, :, 0]
    fluence = (flux * 1000.0).astype(np.float32)             # mJ/mm^2

    tag = tag or AA.angle_tag(A_deg, B_deg)
    dt = time.time() - t0
    if save:
        os.makedirs(outdir, exist_ok=True)
        base = f"{head}_F{WL}_{tag}_r{int(radius_mm)}"
        sio.savemat(os.path.join(outdir, f"fluence_{base}.mat"),
                    {"fluence": fluence, "srcpos": src_pos, "srcdir": srcdir,
                     "origin": origin, "A_deg": A_deg, "B_deg": B_deg,
                     "vol_size": np.array(vol_size)}, do_compression=True)
        meta = {"sample_id": head, "wavelength": WL, "A_deg": float(A_deg),
                "B_deg": float(B_deg), "radius_mm": float(radius_mm),
                "srcpos": src_pos.tolist(), "srcdir": srcdir.tolist(),
                "origin": origin.tolist(), "nphoton": int(nphoton),
                "sim_seconds": round(dt, 1)}
        with open(os.path.join(outdir, f"meta_{base}.json"), "w") as f:
            json.dump(meta, f, indent=2)
    nz = fluence[fluence > 0]
    print(f"  [{head} A{A_deg:.0f} B{B_deg:.0f}] src={src_pos.tolist()} "
          f"max={fluence.max():.3e} nz={nz.size} ({dt:.1f}s)")
    return dict(fluence=fluence, srcpos=src_pos, srcdir=srcdir, origin=origin,
                mask=mask, cfg_vol=cfg_vol, cfg_prop=cfg_prop, sim_seconds=dt)


if __name__ == "__main__":
    # smoke test: one cheap sim on ADT1
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="ADT1")
    ap.add_argument("--A", type=float, default=75.0)
    ap.add_argument("--B", type=float, default=0.0)
    ap.add_argument("--nphoton", type=float, default=1e6)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    run_one(a.head, a.A, a.B, a.nphoton, M.SIM_V2_PRETEST, gpu_id=a.gpu)
