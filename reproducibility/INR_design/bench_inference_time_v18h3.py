#!/usr/bin/env python3
"""Wall-clock cost of one forward inference, for the finalised model set.

WHAT ONE INFERENCE IS. Producing the ten-gate log-fluence field for one scene over EVERY tissue
voxel of that head. That is the unit a user actually asks for, and it is the only definition under
which the three architecture families can be compared at all: a grid model emits the whole volume in
a single forward pass, while our INR and the coordinate baselines are queried point-by-point, so
"one forward pass of the network" means different amounts of delivered field in each family.

WHAT IS INSIDE THE TIMED REGION. Everything the model needs to turn a scene into a field:

  grid   : build_grid_inputs -> one model(volume) forward -> grid_sample at the query points.
           pred.reset() before every repetition, otherwise the volume forward is cached after the
           first one and the measurement collapses to the cost of a trilinear gather.
  ours   : per-chunk optical sampling, source features, path integrals, pyramid sampling, then ten
           forwards (one per gate).
  coord  : the same minus the pyramid.

WHAT IS OUTSIDE IT. Reading the scene from disk, and loading the 4.4 GB pyramid from disk. Those are
I/O against a Lustre filesystem shared with other users; they are not properties of the model, and
including them would measure the cluster.

OURS IS REPORTED TWICE, which is the honest way to present a model whose input is precomputed:

  cached pyramid   the FM pyramid for this head already exists (it is xyz-only and static, so it is
                   computed once per head and reused by every electrode, every gate and every later
                   query on that head)
  + encoder        plus the one-off encoder pass that produces it, charged in full to this single
                   inference -- the worst case, a cold head queried exactly once

The truth for a deployment sits between the two and depends on how many queries a head receives; the
table gives both ends rather than a number that quietly assumes one of them.

  python bench_inference_time_v18h3.py --gpu 0 --reps 100
"""
import os, sys, json, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import repro_config as RC
import train_inr_v11 as T
from eval_benchmark_v14cw import build_predictor_v14
from eval_bench_pergate import predict_gates

CK3 = RC.INR_CHECKPOINT_DIR
CK11 = RC.INR_CHECKPOINT_DIR
OUT = os.path.join(RC.RESULTS_DIR, "infer_time_v18h3.json")

# seed 1 for the baselines and seed 0 for ours, the seed each family's table row uses
MODELS = [
    ("v18hero3",    "Ours",      f"{CK11}/inr_v18hero3_s0_base.pt"),
    ("vit",         "ViT",       f"{CK3}/vit3_frozen/inr_v18vitH_s1_bench_unetr.pt"),
    ("dynunet",     "DynUNet",   f"{CK3}/inr_v17gh_s1_bench_dynunet.pt"),
    ("segresnet",   "SegResNet", f"{CK3}/inr_v17gh_s1_bench_segresnet.pt"),
    ("unet",        "UNet",      f"{CK3}/inr_v17gh_s1_bench_unet.pt"),
    ("fno",         "FNO",       f"{CK3}/inr_v17gh_s1_bench_fno.pt"),
    ("coordrff2",   "Coord-RFF", f"{CK3}/inr_v18c2h_coord_rff_ft3_s1.pt"),
    ("coordsiren2", "SIREN",     f"{CK3}/inr_v18c2h_coord_siren_ft3_s1.pt"),
]


def timeit(fn, reps, warmup=3):
    """-> (mean_s, sd_s, n). CUDA is asynchronous: without the synchronise the timer measures how
    long it took to QUEUE the work, which for the grid models is close to zero."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts)
    return float(a.mean()), float(a.std(ddof=1)), len(a)


def encoder_time(head, dev, reps):
    """Cost of the FM pyramid for one head: the encoder pass alone, volume already in memory."""
    os.environ.setdefault("FM_CKPT_DIR", str(RC.FM_ADAPTER_DIR))
    import extract_pyramids as EP
    mat = os.path.join(RC.MNI_DATASET_DIR, f"{head}_v11mni_F810.mat")
    if not os.path.exists(mat):
        return None
    vol = EP.load_and_normalize(mat)                    # I/O, outside the timed region
    model = EP.build_model(dev)
    npar = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        m, s, n = timeit(lambda: EP.extract_pyramid(model, vol, dev), reps, warmup=1)
    del model
    torch.cuda.empty_cache()
    return dict(mean_s=m, sd_s=s, n=n, params=int(npar), head=head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--head", default=None, help="default: the first held-out OASIS head")
    ap.add_argument("--elec", default="Cz")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--enc-reps", type=int, default=5,
                    help="encoder repetitions; it is ~1000x the cost of a field query, so 100 of "
                         "them would dominate the run for no gain in precision")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    split = os.environ.get("EVAL_SPLIT", "test")

    scenes = [s for s in T.discover_scenes()
              if T.label_of(s["head"]) == split and s["tag"] == a.elec
              and s["head"].startswith("oas")]
    assert scenes, f"no held-out OASIS scene at {a.elec}"
    sc = next((s for s in scenes if s["head"] == a.head), sorted(scenes, key=lambda s: s["head"])[0])
    print(f"[time] scene {sc['head']}/{sc['tag']}  reps={a.reps}  gpu={a.gpu}", flush=True)

    res = {"scene": f"{sc['head']}/{sc['tag']}", "reps": a.reps, "device": torch.cuda.get_device_name(a.gpu)
           if torch.cuda.is_available() else "cpu", "models": {}}

    for key, label, fp in MODELS:
        assert os.path.exists(fp), f"missing checkpoint {fp}"
        pred, npar = build_predictor_v14(fp, dev)
        store = T.SceneStore(dev)          # after the predictor: SRC_REF is pinned from cfg
        sd = store.get(sc)
        idx = sd["valid_idx"].float()
        n_pts = int(idx.shape[0])

        def one():
            pred.reset()                   # grid: forces the volume forward every repetition
            with torch.no_grad():
                predict_gates(pred, sd, idx)

        m, s, n = timeit(one, a.reps)
        res["models"][key] = dict(label=label, family=pred.family, params=int(npar),
                                  n_points=n_pts, mean_s=m, sd_s=s, n=n,
                                  ckpt=os.path.basename(fp))
        print(f"  {label:10s} {pred.family:9s} {npar/1e6:7.2f} M  "
              f"{m*1e3:8.1f} ± {s*1e3:5.1f} ms   x100 = {m*100:7.2f} s", flush=True)
        del pred, store, sd, idx
        torch.cuda.empty_cache()

    print("[time] encoder pass (FM pyramid, one head)", flush=True)
    enc = encoder_time(sc["head"], dev, a.enc_reps)
    res["encoder"] = enc
    if enc:
        print(f"  encoder    {enc['params']/1e6:7.2f} M  {enc['mean_s']:8.2f} ± {enc['sd_s']:5.2f} s"
              f"   ({enc['n']} reps)", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
