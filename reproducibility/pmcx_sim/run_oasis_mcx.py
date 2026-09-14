#!/usr/bin/env python3
"""
Drive the validated v2/v10 MCX fluence core (sim_core_v2) on the SHARM heads
without touching the frozen scb/bw pipeline.

SHARM standardized property volumes (from data_expansion/generate_oasis_v2.py) live
in dataset_oasis810/ and use the SAME layout + PROP_KEY + F810 optical scale as
scb/bw. The ONLY thing sim_core_v2 needs is to look there instead of
dataset_v2_head810/, so we redirect two module globals (DATASET_V2, CACHE_V2) at
runtime -- std_mat()/get_cfg() read them at call time, so no edits to sim_core_v2
or v2_manifest are required. Everything else (anatomical (A,B) angle -> srcdir,
disk source standoff, discrete-media cfg that fixes the pmcxcl media-table bug,
810 nm, time gate, seed) is reused verbatim, so SHARM fluences are directly
comparable to the existing dataset.

Examples
  # smoke test: one (A,B) on one head
  python run_sharm_mcx.py --head oa_test01 --A 75 --B 0 --nphoton 1e7
  # full anatomical angle grid on one head
  python run_sharm_mcx.py --head oa_test01 --grid --nphoton 5e7
  # a batch of heads (e.g. from the manifest train split)
  python run_sharm_mcx.py --heads oa_test01 --grid --nphoton 5e7
"""
import os, sys, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_expansion"))
sys.path.insert(0, os.path.join(ROOT, "pmcx_sim"))
import repro_config as RC

import v2_manifest as M
import anat_angles as AA

# ---- redirect the MCX core onto the SHARM dataset (call-time globals) ----
OASIS_DATASET = RC.OASIS_DATASET_DIR
OASIS_SIM_ROOT = os.path.join(RC.SIM_ROOT, "sim_oasis_810")
M.DATASET_V2 = OASIS_DATASET
M.CACHE_V2   = os.path.join(OASIS_SIM_ROOT, "cache")
OASIS_SIM_DATA = os.path.join(OASIS_SIM_ROOT, "data", "wl810")

import sim_core_v2 as SC        # imports M (already patched); loads pmcxcl .so


def heads_present():
    import glob
    fs = glob.glob(os.path.join(OASIS_DATASET, f"*_copmri_withHermiteF{M.WL}.mat"))
    return sorted(os.path.basename(f).split("_")[0] for f in fs)


def run_head(head, angles, nphoton, outdir, gpu=0, force_rebuild=False):
    matp = M.std_mat(head)
    if not os.path.isfile(matp):
        print(f"[error] missing standardized volume for {head}: {matp} "
              f"(run generate_oasis_v2.py first)"); return
    for (A, B) in angles:
        tag = AA.angle_tag(A, B)
        base = f"{head}_F{M.WL}_{tag}_r5"
        done = os.path.join(outdir, f"meta_{base}.json")
        if os.path.isfile(done):
            print(f"  [skip] {head} {tag} (already done)"); continue
        SC.run_one(head, A, B, nphoton, outdir, gpu_id=gpu, force_rebuild=force_rebuild)
        force_rebuild = False        # cfg cache built once per head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None)
    ap.add_argument("--heads", nargs="+", default=None)
    ap.add_argument("--A", type=float, default=75.0)
    ap.add_argument("--B", type=float, default=0.0)
    ap.add_argument("--grid", action="store_true", help="use the anatomical (A,B) grid")
    ap.add_argument("--nphoton", type=float, default=1e7)
    ap.add_argument("--gpu", type=int, default=1)     # pmcxcl gpuid is 1-based (1..N)
    ap.add_argument("--out-dir", default=OASIS_SIM_DATA)
    ap.add_argument("--force-rebuild", action="store_true")
    a = ap.parse_args()

    if a.heads:
        heads = a.heads
    elif a.head:
        heads = [a.head]
    else:
        heads = heads_present()
        print(f"[oasis-mcx] no --head given; {len(heads)} heads present in {OASIS_DATASET}")
    angles = M.angle_grid() if a.grid else [(a.A, a.B)]
    print(f"[oasis-mcx] heads={len(heads)} angles/head={len(angles)} nphoton={a.nphoton:.0e} "
          f"-> {a.out_dir}")
    for h in heads:
        run_head(h, angles, a.nphoton, a.out_dir, gpu=a.gpu,
                 force_rebuild=a.force_rebuild)
    print("[oasis-mcx] DONE")


if __name__ == "__main__":
    main()
