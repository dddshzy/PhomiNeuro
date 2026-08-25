#!/usr/bin/env python3
"""Check that your installation reproduces our reference field.

    python examples/03_verify_install.py

This compares the model against the Monte-Carlo field shipped with the demo head and prints the
agreement gate by gate. Use it to confirm the pieces are wired together correctly -- the right
backbone, the right encoder seed, an intact pyramid.

WHAT THIS IS NOT: a measure of how well the model works. It is one illumination site on one head,
and the shipped reference is itself a finite-photon Monte-Carlo run, not truth. Accuracy, benchmarks
and the evaluation protocol are in the paper; please quote them from there.

A healthy install shows agreement that is high at early gates and falls off at late ones -- late
gates carry the fewest photons, and the reference is noisiest exactly where the model is hardest.
What matters here is that your numbers match the ones printed below the table, not that they are
high.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phomineuro import GATE_NS, ROOT, WEIGHTS, Predictor, Scene   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default=None, metavar="HEAD/ELEC",
                help="scene to check, e.g. scb16/Cz. Default: every shipped reference.")
ap.add_argument("--decades", type=float, default=6.0,
                help="compare only where MC is within this many decades of that gate's own peak")
ap.add_argument("--points", type=int, default=200_000)
ap.add_argument("--chunk", type=int, default=200_000, help="lower to 20000 on a small card")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--cache", default=os.path.join(ROOT, "pyramid_cache"))
ap.add_argument("--vista3d", default=os.environ.get("VISTA3D_CKPT", ""))
a = ap.parse_args()

dev = torch.device(f"cuda:{a.gpu}" if a.gpu >= 0 and torch.cuda.is_available() else "cpu")
scene_kw = dict(cache_dir=a.cache, vista3d_ckpt=a.vista3d,
                weights_dir=os.path.join(WEIGHTS, "fm_encoder"))

shipped = sorted(os.path.basename(p)[len("fluence_"):-len("_r5_t10.mat")].replace("_F810_", "/")
                 for p in glob.glob(os.path.join(ROOT, "demo_heads", "mc_reference", "*.mat")))
scenes = [a.scene] if a.scene else shipped
if not scenes:
    sys.exit("no Monte-Carlo reference shipped; each is ~18 MB, so only one is carried.")

preds = [Predictor(os.path.join(WEIGHTS, f"phomineuro_s{s}.pt"), dev, verbose=(s == 0))
         for s in range(3)]

for spec in scenes:
    head, elec = spec.split("/")
    scene = Scene(head, elec, ROOT, dev, src_ref=preds[0].src_ref, **scene_kw)
    mc = scene.mc_reference(ROOT)
    if mc is None:
        print(f"skipping {spec}: no reference shipped")
        continue

    rng = np.random.default_rng(0)
    tis = torch.nonzero(scene.tissue, as_tuple=False).cpu().numpy()
    sel = tis[rng.choice(tis.shape[0], min(a.points, tis.shape[0]), replace=False)]
    xyz = torch.tensor(sel, dtype=torch.float32, device=dev)

    P = torch.stack([p.at_points(scene, xyz, chunk=a.chunk) for p in preds]).cpu().numpy()
    G = mc[sel[:, 0], sel[:, 1], sel[:, 2], :]                               # (N,10) linear

    print(f"\n{'='*66}\n{spec}   {sel.shape[0]:,} tissue voxels   3 seeds")
    print(f"agreement on log10 fluence, where MC is within {a.decades:g} decades of its peak\n")
    print(f"{'gate':>4} {'t/ns':>5} {'n':>8} {'agreement':>11} {'offset':>9} {'med|err|':>9}")
    for k in range(10):
        g = G[:, k]
        m = g > g.max() * 10 ** (-a.decades)
        gl = np.log10(g[m])
        rows = []
        for s in range(3):
            pl = P[s][m, k]
            ss = ((gl - gl.mean()) ** 2).sum()
            rows.append((1 - ((pl - gl) ** 2).sum() / ss, (pl - gl).mean(),
                         np.median(np.abs(pl - gl))))
        r2, bias, mae = np.array(rows).mean(0)
        print(f"{k:>4} {GATE_NS[k]:>5.2f} {int(m.sum()):>8,} "
              f"{r2:>11.4f} {bias:>9.4f} {mae:>9.4f}")

print("\nA correct install on scb16/Cz gives roughly 0.97 at gate 0 falling to 0.73 at gate 9, with")
print("offsets under 0.2. If yours are far from that, the usual causes are a mismatched VISTA3D")
print("checkpoint or a pyramid cached by an older version -- delete pyramid_cache/ and rerun.")
print("\nThese are installation reference values, not model performance. See the paper for that.")
