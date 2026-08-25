#!/usr/bin/env python3
"""The four ways to query the surrogate: one coordinate, one time, all ten gates, the whole head.

    python examples/01_quickstart.py --head scb16 --electrode Cz

First run extracts the head's FM feature pyramid (~1 min on a GPU, ~4 GB) and caches it; later runs
reuse the cache. Extraction is deterministic, so the cache changes nothing but the wall clock.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phomineuro import GATE_NS, ROOT, WEIGHTS, Predictor, Scene, load_ensemble   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--head", default="scb16")
ap.add_argument("--electrode", default="Cz")
ap.add_argument("--seed", type=int, default=0, help="which of the three trained seeds")
ap.add_argument("--all-seeds", action="store_true", help="run all three and report the spread")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--cache", default=os.path.join(ROOT, "pyramid_cache"))
ap.add_argument("--vista3d", default=os.environ.get("VISTA3D_CKPT", ""))
a = ap.parse_args()

dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
print(f"device: {dev}\n")

pred = Predictor(os.path.join(WEIGHTS, f"phomineuro_s{a.seed}.pt"), dev)
scene = Scene(a.head, a.electrode, ROOT, dev, src_ref=pred.src_ref, cache_dir=a.cache,
              vista3d_ckpt=a.vista3d, weights_dir=os.path.join(WEIGHTS, "fm_encoder"))

# ================================================================================ 1. ONE COORDINATE
# A single voxel, all ten gates. The model is a field: nothing is evaluated that you did not ask for,
# so this costs microseconds after the pyramid is in memory.
print("\n" + "=" * 78 + "\n1. SINGLE COORDINATE, all 10 gates")
xyz = [[120.0, 130.0, 170.0]]                       # voxel units on the 224x256x300 MNI grid
v = pred.at_points(scene, xyz)[0]                   # (10,)
print(f"   point {xyz[0]}  (1 mm voxels, AC at [112, 148, 175])")
for k, (t, lg) in enumerate(zip(GATE_NS, v.tolist())):
    print(f"     gate {k}  t = {t:4.2f} ns   log10 fluence = {lg:8.4f}")

# ================================================================================ 2. ONE TIME POINT
# The network is continuous in t, so a time BETWEEN gate centres is a legal query -- something a
# model that outputs a fixed 10-channel grid cannot do at all.
print("\n" + "=" * 78 + "\n2. SINGLE TIME POINT (including one off the training grid)")
pts = torch.tensor([[120.0, 130.0, 170.0],
                    [150.0, 130.0, 168.0],
                    [190.0, 133.0, 166.0]], device=dev)
for t in (0.3, 0.4, 0.6, 1.1):
    on_grid = any(abs(t - g) < 1e-9 for g in GATE_NS)
    lab = "training gate" if on_grid else "INTERPOLATED between gates"
    got = pred.at_points(scene, pts, t_ns=t)[:, 0]
    print(f"   t = {t:4.2f} ns  [{lab}]   log10 fluence = "
          f"{np.array2string(got.cpu().numpy(), precision=4)}")
print("   Off-gate values are not free: the interpolation cost is largest around t = 0.4-0.6 ns,\n"
      "   where a 10-sample curve is at its most curved. Treat them as interpolation, not as data.")

# ================================================================================ 3. ALL TEN GATES
print("\n" + "=" * 78 + "\n3. ALL 10 GATES over a set of points")
xyz3 = scene.tissue_coords()[::5000]
t0 = time.time()
g10 = pred.at_points(scene, xyz3)
print(f"   {xyz3.shape[0]} points x 10 gates in {time.time()-t0:.2f} s -> {tuple(g10.shape)}")
print(f"   per-gate mean over these points (log10):")
print("     " + "  ".join(f"{t:.1f}ns:{m:7.3f}" for t, m in zip(GATE_NS, g10.mean(0).tolist())))

# ================================================================================ 4. WHOLE HEAD
print("\n" + "=" * 78 + "\n4. WHOLE HEAD, one gate")
t0 = time.time()
vol = pred.volume(scene, t_ns=GATE_NS[1])            # gate 1, t = 0.3 ns
dt = time.time() - t0
n_tissue = int(scene.tissue.sum())
print(f"   {tuple(vol.shape)}  ({n_tissue:,} tissue voxels) in {dt:.1f} s")
print(f"   log10 fluence over tissue: peak {float(vol.max()):.3f}, "
      f"5th percentile {float(torch.quantile(vol[scene.tissue].flatten().float(), 0.05)):.3f}")
print("   Air is not evaluated. It is outside the training distribution entirely, so a value there\n"
      "   would be extrapolation dressed up as a result.")

# ================================================================================ SEED SPREAD
if a.all_seeds:
    print("\n" + "=" * 78 + "\n5. ALL THREE SEEDS")
    preds = load_ensemble(WEIGHTS, dev, verbose=False)
    stack = torch.stack([p.at_points(scene, xyz3) for p in preds])       # (3,N,10)
    sd_ = stack.std(0)
    print(f"   seed-to-seed std of log10 fluence: median {float(sd_.median()):.4f}, "
          f"p95 {float(torch.quantile(sd_.flatten().float(), 0.95)):.4f}")
    print("   Report the spread rather than one seed. Choosing the 'best' seed on the same data you\n"
          "   then quote biases the number, whether the choice is made on validation or on test.")

print("\ndone.")
