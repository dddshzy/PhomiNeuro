#!/usr/bin/env python3
"""Does this standalone package reproduce the production pipeline exactly?

The package transcribes feature assembly out of a much larger research tree. Transcription errors do
not raise -- they degrade the prediction quietly, which is the failure mode this project has hit
repeatedly (a nearest-voxel path gather instead of trilinear; source-referenced features fed to an
entry-trained model). So the transcription is not trusted, it is CHECKED: same head, same site, same
points, both code paths, compared element by element.

Runs only on the machine that holds the research tree; it is a provenance test, not a user test.
Everything a user needs is checked by examples/ instead.
"""
import os
import sys

import numpy as np
import torch

PROD = "/iopsstor/scratch/cscs/sdong/FM_INR"
if not os.path.isdir(PROD):
    sys.exit(f"SKIP: production tree not present at {PROD} (this test only runs at CSCS)")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROD)
sys.path.insert(0, os.path.join(PROD, "INR_design"))

HEAD, ELEC, N = "scb16", "Cz", 20_000
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "weights", "phomineuro_s0.pt")
dev = torch.device(f"cuda:{os.environ.get('GPU', '3')}" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- production path
import train_inr_v11 as T                                                        # noqa: E402
from eval_benchmark_v14cw import build_predictor_v14                             # noqa: E402
from eval_bench_pergate import predict_gates                                     # noqa: E402

pred, _ = build_predictor_v14(CKPT, dev)       # pins T.SRC_REF from cfg -- must run before the store
store = T.SceneStore(dev)
sc = dict(head=HEAD, tag=ELEC, geom_key=HEAD,
          pyramid=os.path.join(PROD, "INR_design", "extracted_pyramids_v16",
                               f"{HEAD}_v11mni_F810_pyramid.pt"),
          prop=os.path.join(PROD, "dataset_v11_mni", f"{HEAD}_v11mni_F810.mat"),
          fluence=os.path.join("/iopsstor/scratch/cscs/sdong/pmcx_sim/sim_v11_mni_810/data/wl810",
                               f"fluence_{HEAD}_F810_{ELEC}_r5_t10.mat"),
          meta=os.path.join("/iopsstor/scratch/cscs/sdong/pmcx_sim/sim_v11_mni_810/data/wl810",
                            f"meta_{HEAD}_F810_{ELEC}_r5_t10.json"))
sd = store.get(sc)
print(f"[prod] SRC_REF={T.SRC_REF}  srcpos={[round(float(v),3) for v in sd['srcpos']]}")

g = torch.Generator(device="cpu").manual_seed(12345)
pool = torch.nonzero(torch.from_numpy(sd["tissue_np"]), as_tuple=False)
xyz = pool[torch.randint(0, pool.shape[0], (N,), generator=g)].float().to(dev)
ref = predict_gates(pred, sd, xyz)                                        # (N,10)

# ---------------------------------------------------------------- standalone path
from phomineuro import Scene, Predictor, ROOT                                       # noqa: E402

p2 = Predictor(CKPT, dev)
s2 = Scene(HEAD, ELEC, ROOT, dev, src_ref=p2.src_ref, pyramid=torch.load(sc["pyramid"],
                                                                        map_location="cpu")["pyramid"])
got = p2.at_points(s2, xyz)

# ---------------------------------------------------------------- compare
fail = 0
d_src = float((s2.srcpos - sd["srcpos"]).abs().max())
d_light = float((s2.light - sd["light"]).abs().max())
print(f"srcpos  max|diff| = {d_src:.3e}")
print(f"light   max|diff| = {d_light:.3e}")
d = (got - ref).abs()
print(f"log10 fluence over {N} points x 10 gates:")
print(f"  max|diff|  = {float(d.max()):.3e}")
print(f"  mean|diff| = {float(d.mean()):.3e}")
print(f"  range of the reference: [{float(ref.min()):.3f}, {float(ref.max()):.3f}]")
for name, val, tol in (("srcpos", d_src, 1e-5), ("light", d_light, 1e-6),
                       ("fluence", float(d.max()), 1e-4)):
    ok = val <= tol
    fail += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name} within {tol:g}")
sys.exit(1 if fail else 0)
