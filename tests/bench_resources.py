#!/usr/bin/env python3
"""Measure what this actually needs: peak GPU memory, host RAM, and wall time, stage by stage.

    python tests/bench_resources.py --gpu 0 --vista3d $VISTA3D_CKPT

The two stages have very different profiles and the README must not quote one number for both:

  EXTRACTION  runs a 217M-parameter encoder over a 224x256x300 volume by sliding window and
              accumulates a 48-channel full-resolution output. This is the peak.
  INFERENCE   holds the finished pyramid and evaluates the 1.18M-parameter surrogate. Much smaller
              compute, but the pyramid has to live somewhere.

Peaks are read from torch.cuda.max_memory_reserved, i.e. what the allocator actually took from the
card -- not max_memory_allocated, which under-reports what a card must have free.
"""
import argparse
import gc
import os
import resource
import sys
import time

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phomineuro import GATE_NS, ROOT, WEIGHTS, Predictor, Scene            # noqa: E402
from phomineuro.encoder import build_fm_encoder, extract_pyramid, normalise_volume   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--head", default="scb16")
ap.add_argument("--electrode", default="Cz")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--cpu", action="store_true", help="measure the CPU-only path instead")
ap.add_argument("--chunk", type=int, default=200_000,
                help="points per forward pass. The single biggest lever on inference peak memory: "
                     "measured 9.93 GB at 200k against 5.73 GB at 20k, for ~1 s more on a whole head.")
ap.add_argument("--vista3d", default=os.environ.get("VISTA3D_CKPT", ""))
ap.add_argument("--cap-gb", type=float, default=None,
                help="hard-cap the process to this many GB of card memory. This is how the README's "
                     "minimum-VRAM figure was established: max_memory_reserved on a 95 GB card "
                     "reports what a generous caching allocator TOOK, not what a small card NEEDS.")
a = ap.parse_args()

dev = torch.device("cpu") if a.cpu else torch.device(f"cuda:{a.gpu}")
CUDA = dev.type == "cuda"
GB = 1024 ** 3
if CUDA and a.cap_gb:
    total = torch.cuda.get_device_properties(dev).total_memory / GB
    torch.cuda.set_per_process_memory_fraction(min(a.cap_gb / total, 1.0), dev)
    print(f"[cap] process limited to {a.cap_gb:.1f} GB of {total:.0f} GB\n")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)   # ru_maxrss is KiB


def gpu_peak_gb():
    return torch.cuda.max_memory_reserved(dev) / GB if CUDA else float("nan")


def reset():
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


rows = []


def stage(name, fn):
    reset()
    t0 = time.time()
    out = fn()
    if CUDA:
        torch.cuda.synchronize(dev)
    rows.append((name, time.time() - t0, gpu_peak_gb(), rss_gb()))
    return out


print(f"device        : {torch.cuda.get_device_name(dev) if CUDA else 'CPU'}")
if CUDA:
    p = torch.cuda.get_device_properties(dev)
    print(f"card memory   : {p.total_memory / GB:.1f} GB")
print(f"host RAM      : {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / GB:.0f} GB")
print(f"CPU threads   : {os.cpu_count()} (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','unset')})")
print(f"torch         : {torch.__version__}\n")

head_path = os.path.join(ROOT, "demo_heads", f"{a.head}_v11mni_F810.mat")
prop = np.transpose(sio.loadmat(head_path)["vol_prop_eye_aseg"].astype(np.float32), (3, 0, 1, 2))

enc = stage("build encoder", lambda: build_fm_encoder(
    a.vista3d, os.path.join(WEIGHTS, "fm_encoder"), dev, verbose=False))
pyr = stage("extract pyramid (5 levels)",
            lambda: extract_pyramid(enc, normalise_volume(prop), dev, verbose=False))
del enc
reset()

pred = stage("load surrogate", lambda: Predictor(
    os.path.join(WEIGHTS, "phomineuro_s0.pt"), dev, verbose=False))
scene = stage("build scene (pyramid -> device)", lambda: Scene(
    a.head, a.electrode, ROOT, dev, src_ref=pred.src_ref, pyramid=pyr, verbose=False))
stage("query 1 point, 10 gates", lambda: pred.at_points(scene, [[120.0, 130.0, 170.0]]))
stage("query 100k points, 10 gates",
      lambda: pred.at_points(scene, scene.tissue_coords()[:100_000], chunk=a.chunk))
stage("whole head, 1 gate", lambda: pred.volume(scene, t_ns=GATE_NS[1], chunk=a.chunk))
stage("whole head, all 10 gates", lambda: pred.volume(scene, chunk=a.chunk))

print(f"{'stage':<34} {'wall/s':>8} {'GPU peak/GB':>12} {'host RSS/GB':>12}")
for n, t, g, r in rows:
    print(f"{n:<34} {t:>8.1f} {g:>12.2f} {r:>12.2f}")

ext = max(r[2] for r in rows[:2]) if CUDA else float("nan")
inf = max(r[2] for r in rows[2:]) if CUDA else float("nan")
print(f"\nGPU peak, extraction stage : {ext:.2f} GB")
print(f"GPU peak, inference stage  : {inf:.2f} GB")
print(f"host RSS peak overall      : {max(r[3] for r in rows):.2f} GB")
print(f"pyramid on disk            : "
      f"{sum(p.numel() * p.element_size() for p in pyr) / GB:.2f} GB")
