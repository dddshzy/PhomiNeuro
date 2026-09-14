#!/usr/bin/env python3
"""Assert the two prediction paths agree, for every family, on the same points.

WHAT THIS CATCHES, AND WHY IT EXISTS.
Two evaluators build the model's input features independently:

    eval_bench_pergate.predict_gates  -> (N, 10) per-gate log10   (used by eval_steadymask, the
                                         benchmark and the ablation)
    eval_benchmark_v14cw.predict_field -> (N,)  log10 CW          (used by the CW benchmark)

Duplicated feature construction is exactly the kind of code that drifts. It did: predict_gates
called path_features() without forwarding cfg["path_tri"], so a model trained with trilinear path
sampling was scored with nearest-neighbour gathers. Measured over the full 504-scene held-out set,
that mismatch cost V17 +0.0023 R2 and -0.0040 RMSE -- small, but a silent train/eval inconsistency
that nothing in the pipeline would have flagged.

The invariant that makes a test possible: for the time-resolved families, predict_field is exactly
integ_t(predict_gates), because it queries the same ten time codes and log-sum-exps them. So

    integ_t(predict_gates(model, sd, x))  ==  predict_field(model, sd, x)

must hold to floating-point tolerance for ANY checkpoint. If the two paths ever build a feature
differently, this breaks -- regardless of which feature, which family, or which flag was forgotten.

    INR_SPLIT=v16_split EVAL_SPLIT=test python test_eval_consistency.py --gpu 0
"""
import os, sys, glob, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "data_expansion")):
    sys.path.insert(0, p)
import train_inr_v11 as T
from eval_benchmark_v14cw import build_predictor_v14, predict_field
from eval_bench_pergate import predict_gates
from eval_v11_level2 import predict as predict_l2

CK11 = f"{HERE}/inr_checkpoints_v11"
CK3 = f"{HERE}/inr_checkpoints_v3"
# one representative per family, chosen so every feature path is exercised: ours with path features
# and trilinear sampling, an ablation with path_dim=0, a coord baseline, a grid baseline.
CASES = [
    ("ours / V17 K=3000 (path_tri=True)", f"{CK11}/inr_v17k3_s0_base.pt"),
    ("ours / V16 hero2 (path_tri unset)", f"{CK11}/inr_v16hero2_s1_base.pt"),
    ("ablation / no path features",       f"{CK11}/inr_v17a_nopath_base.pt"),
    ("coord / RFF entry+hinge",           f"{CK3}/inr_v17hin_coord_rff_s1.pt"),
    ("grid / DynUNet entry",              f"{CK3}/inr_v17gh_s1_bench_dynunet.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--only", default=None, help="run a single checkpoint (used by the driver)")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max |integ_t(per-gate) - CW|. Both paths run the same modules in the same "
                         "order, so agreement should be at float32 round-off; a loose tolerance "
                         "would let a real feature mismatch through.")
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu); dev = torch.device(f"cuda:{a.gpu}")

    # build_predictor_v14 PINS src_ref for the whole process and refuses to mix conventions, which
    # is correct -- so each case must run in its own process. Without --only, this dispatches one
    # subprocess per case and aggregates; with --only it is that subprocess.
    if a.only is None:
        import subprocess
        rc = 0
        for lab, ck in CASES:
            if not os.path.exists(ck):
                print(f"  SKIP  {lab}: checkpoint absent"); continue
            r = subprocess.run([sys.executable, __file__, "--gpu", str(a.gpu), "--only", ck,
                                "--points", str(a.points), "--tol", str(a.tol)],
                               capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.strip().startswith(("PASS", "FAIL"))]
            print(f"  {line[0].strip() if line else 'ERROR'}  {lab}")
            if r.returncode != 0:
                rc = 1
                if not line:
                    print("     " + (r.stderr.strip().splitlines() or ["?"])[-1])
        print()
        print("all families agree: the two prediction paths build identical features" if rc == 0
              else "MISMATCH -- see above; compare predict_gates against predict_field line by line")
        raise SystemExit(rc)

    fails = []
    for lab, ck in [(os.path.basename(a.only), a.only)]:
        if not os.path.exists(ck):
            print(f"  SKIP  {lab}: checkpoint absent"); continue
        # Each checkpoint pins its own src_ref convention, and build_predictor_v14 refuses to mix
        # them inside one process -- so every case gets a fresh SceneStore and scene.
        T.SRC_REF = "source"
        pred = build_predictor_v14(ck, dev)[0]
        store = T.SceneStore(dev)
        s = [x for x in T.discover_scenes()
             if x["head"].startswith(("oa", "oas")) and x["tag"] == "Cz"][0]
        sd = store.get(s)
        idx = sd["valid_idx"].float()
        g = torch.Generator(device=idx.device); g.manual_seed(0)
        x = idx[torch.randperm(idx.shape[0], device=idx.device, generator=g)[:a.points]]

        pred.reset()
        with torch.no_grad():
            pg = predict_gates(pred, sd, x)
        pred.reset()
        with torch.no_grad():
            cw = predict_field(pred, sd, x)
        got = T.integ_t(pg).squeeze(-1)
        d = float((got - cw).abs().max())
        # THIRD path: eval_v11_level2.predict, which checkerboard_metric uses. It returns the same
        # (N,10) per-gate array as predict_gates, so they must agree elementwise. It was added to
        # this test only after it was found to have the SAME missing cfg["path_tri"] -- two paths
        # were checked against each other while a third quietly disagreed with both.
        d2 = 0.0
        if pred.family == "ours":
            with torch.no_grad():
                pl2 = predict_l2(pred.model, pred.cfg, sd, x, dev)
            d2 = float((pl2 - pg).abs().max())
            d = max(d, d2)
        ok = d <= a.tol
        print(f"{'PASS' if ok else 'FAIL'}  max|Δ| = {d:.3e} (l2 {d2:.1e})  "
              f"path_tri={pred.cfg.get('path_tri')} path_dim={pred.cfg.get('path_dim')}")
        if not ok:
            fails.append((lab, d))
        del pred, store
        torch.cuda.empty_cache()

    print()
    if fails:
        for lab, d in fails:
            print(f"MISMATCH  {lab}: {d:.3e} > {a.tol:.0e}")
        print("\nThe two evaluators built different features for the same points. Compare\n"
              "  eval_bench_pergate.predict_gates   (per-gate path)\n"
              "  eval_benchmark_v14cw.predict_field (CW path)\n"
              "line by line -- every cfg key one of them reads, the other must read too.")
        raise SystemExit(1)
    print("all families agree: the two prediction paths build identical features")


if __name__ == "__main__":
    main()
