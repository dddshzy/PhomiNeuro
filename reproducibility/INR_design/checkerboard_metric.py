#!/usr/bin/env python3
"""Quantify the dark-zone CHECKERBOARD artefact, separating it from smooth over-floor leak.

Real deep fluence is one smooth, connected, monotonically fading blob. The checkerboard artefact is
a scatter of isolated, regularly spaced above-floor specks in the region that should be dark. Three
numbers, computed in the DARK zone (voxels where the 5e7 GT is censored, i.e. <= floor):

  (1) ISOLATION  -- connected-component analysis of the above-floor voxels (26-connectivity):
        speck_frac = fraction of above-floor voxels living in SMALL components (< SMALL voxels)
        n_speck    = number of such small components (per 1e5 dark voxels)
      Smooth extrapolation = a few LARGE components -> speck_frac ~ 0. Checkerboard = many small
      isolated blobs -> speck_frac high. THIS is the direct checkerboard signature.
  (2) ROUGHNESS  -- mean |discrete Laplacian| of the predicted field over the dark zone. A smooth
      field is flat; the checkerboard's bright-dark-bright alternation is spatially jagged -> high.
  (3) LEAK%      -- fraction of dark-zone voxels predicted above floor. Mixes (a) smooth leak past
      the reachable boundary [hinge/offset's job] and (b) the specks [jitter/PlanA/B's job]; only
      meaningful next to (1)-(2).

Runs on VALIDATION heads by default (no test leakage). Usage:
  python checkerboard_metric.py <ckpt> [--heads sh006 ...] [--gates 2 4 6] [--small 50]
"""
import os, sys, argparse
import numpy as np, torch
from scipy.ndimage import label as cc_label

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "data_expansion")):
    sys.path.insert(0, _p)
import train_inr_v11 as T
from eval_v11_level2 import load, predict, SIM7
from eval_benchmark_v14cw import build_predictor_v14
from eval_bench_pergate import predict_gates

VAL_DEFAULT = ["sh006", "sh023", "bw17", "scb08"]     # a few validation heads
STRUCT = np.ones((3, 3, 3), dtype=int)                 # 26-connectivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--heads", nargs="+", default=VAL_DEFAULT)
    ap.add_argument("--elec", default="C3"); ap.add_argument("--gates", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--small", type=int, default=50)   # components below this many voxels = specks
    ap.add_argument("--pred-floor", choices=["common", "self"], default="self",
                    help="threshold used to call a PREDICTED voxel 'above floor'. common = the "
                         "ground truth's floor, which is not available at deployment and lets a "
                         "globally dim model look artefact-free; self = the model's own peak minus "
                         "the same number of decades, which is scale-invariant and uses only what "
                         "the model itself produces. The dark REGION is still defined by the ground "
                         "truth -- that is the question being asked, not information leaked into "
                         "the answer.")
    ap.add_argument("--deep-margin", type=float, default=2.0)  # deep zone = floor + this (pure-artefact region)
    ap.add_argument("--out", default=None,
                    help="where to write the JSON summary. Default (<ckpt>_checker.json) is keyed "
                         "on the CHECKPOINT alone, so six electrodes of one model all write the "
                         "same path -- harmless while only the stdout text is parsed, but it is a "
                         "concurrent write, and this project lost a day to one. Pass a per-task "
                         "path when running electrodes in parallel. Omitting it reproduces the old "
                         "behaviour exactly, so every existing caller is unaffected.")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    ck = torch.load(a.ckpt, map_location="cpu"); cfg = ck["cfg"]
    # V17 CONSISTENCY GUARD. Unlike the scorers this script builds the model itself and reads scenes
    # straight from T.SceneStore, so build_predictor_v14's guard never runs. SceneStore is where
    # srcpos is defined, so an entry-referenced checkpoint scored with the default "source" setting
    # gets features built around a point 15 voxels off -- the same silent mismatch that made the
    # raw-encoder ablation read -2.16 (V16) and -40382 (V17) before it was caught.
    _ref = cfg.get("src_ref", "source")
    if T.SRC_REF != _ref:
        print(f"  [src_ref] SceneStore -> '{_ref}' (from cfg; was '{T.SRC_REF}')", flush=True)
        T.SRC_REF = _ref
    # PYRAMID PROVENANCE, for the same reason and with the same failure mode. eval_benchmark_v14cw
    # gained this guard after the b7 row read R2 = -11.06 from a healthy model that had simply been
    # scored against the wrong pyramid directory -- but that guard lives in build_predictor_v14,
    # which this script bypasses, so until now the dark-zone numbers had no such protection at all.
    # An encoder-ablation checkpoint (buffer `..._raw` / `..._random`) MUST be read with the matching
    # INR_PYRDIR; feeding it the fine-tuned pyramids is a silent train/eval feature mismatch.
    if cfg.get("use_pyramid", True):
        _buf = os.path.basename(str(cfg.get("buffer", "")))
        _pyr = os.environ.get("INR_PYRDIR", "")
        for _sfx in ("_raw", "_random"):
            if _buf.endswith(_sfx) != _pyr.rstrip("/").endswith(_sfx):
                raise SystemExit(
                    f"pyramid provenance mismatch: {os.path.basename(a.ckpt)} trained on buffer "
                    f"'{_buf}' but INR_PYRDIR='{_pyr}'. A '{_sfx}' buffer must be scored against a "
                    f"'{_sfx}' pyramid directory and vice versa. Set INR_PYRDIR and rerun.")
    # BUILD THROUGH build_predictor_v14, NOT T.build_model. The direct call constructs an INRv7 and
    # therefore silently restricts this script to our own family -- the grid baselines (DynUNet,
    # UNet, SegResNet, FNO, UNETR) and the coordinate baselines could not be scored at all, so the
    # dark-zone comparison had no baseline column anywhere in the paper. build_predictor_v14
    # dispatches on the checkpoint's family and returns a Predictor that predict_gates understands.
    # For our own checkpoints the two paths are line-for-line the same computation, and that is
    # verified rather than assumed: see scratch_jobs/dark_pathcheck.json.
    pred, _ = build_predictor_v14(a.ckpt, dev)
    cfg = pred.cfg
    store = T.SceneStore(dev)

    agg = {k: [] for k in ("speck_frac", "n_speck", "rough", "leak", "ndark")}
    for h in a.heads:
        f7 = load(os.path.join(SIM7, f"fluence_{h}_F810_{a.elec}_r5_t10.mat"))
        sc = [s for s in T.discover_scenes() if s["head"] == h and s["tag"] == a.elec][0]
        sd = store.get(sc); vs = tuple(int(v) for v in sd["vol_shape"])
        vi = sd["valid_idx"].float()
        # reset() is load-bearing for the grid families: predict_gates caches the whole predicted
        # volume on the Predictor, so without it every head after the first would be scored against
        # the FIRST head's field.
        pred.reset()
        pr = predict_gates(pred, sd, vi).cpu().numpy()                # (V,10) log10, UNclipped
        idx = sd["valid_idx"].round().long().cpu().numpy()
        peak7 = float(f7.max())
        floor = (np.log10(peak7) if a.pred_floor == "common" else float(np.nanmax(pr))) \
            - T.FLU_FLOOR_DECADES
        for k in a.gates:
            a7 = f7[idx[:, 0], idx[:, 1], idx[:, 2], k]
            dark = a7 <= peak7 * 10 ** (-T.FLU_FLOOR_DECADES)         # 5e7 censored here (should be dark)
            if dark.sum() < 1000:
                continue
            # dense prediction volume (NaN outside tissue) for connected components + Laplacian
            vol = np.full(vs, np.nan, np.float32)
            vol[idx[:, 0], idx[:, 1], idx[:, 2]] = pr[:, k]
            darkvol = np.zeros(vs, bool)
            di = idx[dark]; darkvol[di[:, 0], di[:, 1], di[:, 2]] = True
            above = darkvol & (vol > floor)                          # above-floor in the dark zone
            nabove = int(above.sum()); ndark = int(darkvol.sum())
            # (1) isolation
            lab, n = cc_label(above, structure=STRUCT)
            if n:
                sizes = np.bincount(lab.ravel())[1:]
                small = sizes[sizes < a.small]
                speck_frac = float(small.sum()) / max(nabove, 1)
                n_speck = len(small) / max(ndark, 1) * 1e5
            else:
                speck_frac, n_speck = 0.0, 0.0
            # (2) roughness: mean |6-neighbour Laplacian| of the prediction, restricted to the dark zone
            v0 = np.nan_to_num(vol, nan=floor)
            pad = np.pad(v0, 1, mode="edge")
            lap6 = (pad[2:, 1:-1, 1:-1] + pad[:-2, 1:-1, 1:-1] + pad[1:-1, 2:, 1:-1] + pad[1:-1, :-2, 1:-1]
                    + pad[1:-1, 1:-1, 2:] + pad[1:-1, 1:-1, :-2] - 6 * v0)
            rough = float(np.abs(lap6[darkvol]).mean())
            leak = 100.0 * nabove / max(ndark, 1)
            # DEEP zone = >= floor + margin below the peak: physically ~0 fluence, so ANY above-floor
            # there is pure artefact (no legitimate smooth extrapolation reaches this deep). This is
            # the cleanest checkerboard signal -- it is not diluted by the near-boundary smooth leak.
            deep = a7 <= peak7 * 10 ** (-(T.FLU_FLOOR_DECADES + a.deep_margin))
            deepvol = np.zeros(vs, bool); dj = idx[deep]; deepvol[dj[:, 0], dj[:, 1], dj[:, 2]] = True
            deep_above = deepvol & (vol > floor)
            deep_leak = 100.0 * int(deep_above.sum()) / max(int(deepvol.sum()), 1)
            for key, val in (("speck_frac", 100 * speck_frac), ("n_speck", n_speck),
                             ("rough", rough), ("leak", leak), ("deep_leak", deep_leak), ("ndark", ndark)):
                agg.setdefault(key, []).append(val)

    M = {k: float(np.mean(v)) for k, v in agg.items() if v}
    print(f"checkerboard metric | {os.path.basename(a.ckpt)} | heads {a.heads} gates {a.gates}\n")
    print(f"  (1) ISOLATION  speck_frac = {M['speck_frac']:6.2f} %   (above-floor voxels in <{a.small}-vox specks)")
    print(f"                 n_speck    = {M['n_speck']:6.2f}     (small components per 1e5 dark voxels)")
    print(f"  (2) ROUGHNESS  |lap|      = {M['rough']:6.4f}     (dark-zone spatial jaggedness; lower=smoother)")
    print(f"  (3) LEAK%      above floor= {M['leak']:6.2f} %   (whole dark zone: smooth leak + specks)")
    print(f"  (4) DEEP-LEAK% above floor= {M['deep_leak']:6.2f} %   (>= floor+{a.deep_margin} dec: PURE artefact, cleanest signal)")
    print(f"\n  lower deep_leak / speck_frac / n_speck / rough  ==  less checkerboard.")
    out = a.out or a.ckpt.replace(".pt", "_checker.json")
    import json; json.dump(M, open(out, "w"), indent=1)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
