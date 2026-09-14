#!/usr/bin/env python3
"""WHOLE-VOLUME per-gate scoring that also penalises DARK-ZONE OVERFLOW.

Why: the reachable-set R2 in eval_bench_pergate.py is computed only where the GT is above the floor,
which is 0.3% of tissue voxels at gate 0 and 16% at gate 9 -- i.e. 84-99.7% of the volume carries NO
penalty, so a model that floods the dark zone scores just as well as one that does not. (That is how
a baseline can hold R2 0.87 at gate 3 while its absorbed energy diverges 18x by gate 9.)

Primary metric -- CENSORED (Tobit-style) error over ALL tissue voxels, per gate:
    floor_k = log10(max_k GT) - D
    bright (GT > floor):  e = pred - gt            (two-sided: over- AND under-prediction penalised)
    dark   (GT <= floor): e = max(0, pred - floor) (ONE-sided: only "inventing light" is penalised)
    R2_cens   = 1 - sum(e^2) / sum((gt_clamped - mean)^2)     <- denominator is GT-only => model-independent
    RMSE_cens = sqrt(mean(e^2))                               <- absolute, in log10 decades
The one-sided dark term is deliberate: the MC dark zone is LEFT-CENSORED ("true value <= floor",
a detection limit), so predicting below the floor is legitimate extrapolation, not an error. It also
mirrors the training hinge, so the metric is not tuned to favour our model.

Secondary -- the UNION-set view (GT-bright OR model-bright), reported for cross-check. Note its point
set differs per model, so its R2 denominator is model-dependent; read it together with leak/miss.

Peak reference: the gate's own max. Verified not to be an MC noise spike (p99.99 is 0.92-0.99 of it).
p99 is NOT used: it is the p99 of the NON-ZERO voxels, and the zero fraction swings from 99.7% to 84%
across gates, so a p99 threshold would drift with the gate.

  INR_SPLIT=v16_split EVAL_SPLIT=test python eval_censored.py --gpu 0 \
      --models ours=inr_checkpoints_v11/inr_v16ls_lin03dmid70_base.pt --out out.json
"""
import os, sys, math, json, argparse, zlib
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "data_expansion"),
          os.path.join(os.path.dirname(HERE), "evaluation")):
    sys.path.insert(0, p)
import train_inr_v11 as T
from eval_v11_level2 import load, SIM7
from eval_benchmark_v14cw import build_predictor_v14
from eval_bench_pergate import predict_gates


def category(h):
    return "oasis" if h.startswith(("oa", "oas")) else ("bw" if h.startswith("bw") else
           ("scb" if h.startswith("scb") else "sh"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--elecs", nargs="+", default=["Cz"])
    ap.add_argument("--heads", nargs="+", default=None)
    ap.add_argument("--oasis", nargs="+", default=None)
    ap.add_argument("--floor", type=float, default=8.0)
    ap.add_argument("--points", type=int, default=0,
                    help="0 = every tissue voxel (exact). >0 = evaluate on this many RANDOMLY SAMPLED "
                         "tissue voxels instead. All the metrics here are MEANS over voxels, so a large "
                         "uniform subsample is an unbiased estimate of the whole-volume value at a "
                         "fraction of the cost; the sample is crc32-seeded per scene so every model sees "
                         "the identical voxels.")
    ap.add_argument("--chunk", type=int, default=200000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu); dev = torch.device(f"cuda:{a.gpu}")
    D = a.floor
    split = os.environ.get("EVAL_SPLIT", "test")

    P = {}
    for spec in a.models:
        nm, paths = spec.split("=", 1)
        P[nm] = [build_predictor_v14(p, dev)[0] for p in paths.split(",")]
        print(f"  {nm}: {len(P[nm])} seed(s), family={P[nm][0].family}", flush=True)

    scenes = [s for s in T.discover_scenes() if T.label_of(s["head"]) == split and s["tag"] in a.elecs]
    if a.heads:
        scenes = [s for s in scenes if s["head"] in a.heads]
    if a.oasis is not None:
        keep = set(a.oasis)
        scenes = [s for s in scenes if category(s["head"]) != "oasis" or s["head"] in keep]
    print(f"[censored] split={split} scenes={len(scenes)} D={D:g}", flush=True)

    KEYS = ("R2_cens", "RMSE_cens", "R2_reach", "R2_union", "RMSE_union", "leak_frac", "miss_frac", "dark_frac",
            "R2_censG", "RMSE_censG", "brightG_frac")
    acc = {nm: {k: {m: [] for m in KEYS} for k in range(T.N_STEP)} for nm in P}
    store = T.SceneStore(dev)
    for si, s in enumerate(scenes):
        sd = store.get(s)
        f7 = load(os.path.join(SIM7, f"fluence_{s['head']}_F810_{s['tag']}_r5_t10.mat"))
        idx = sd["valid_idx"]                                   # every tissue voxel
        if a.points and idx.shape[0] > a.points:
            _g = torch.Generator(device=idx.device)
            _g.manual_seed(zlib.crc32((s["head"] + s["tag"]).encode()) % (2 ** 31))
            idx = idx[torch.randperm(idx.shape[0], device=idx.device, generator=_g)[:a.points]]
        jj = idx.round().long().cpu().numpy()
        gt_lin = torch.from_numpy(np.ascontiguousarray(
            f7[jj[:, 0], jj[:, 1], jj[:, 2], :T.N_STEP])).to(dev).float()      # (Nvox,10)
        lgmax = math.log10(float(f7[..., :T.N_STEP].max()))     # GLOBAL peak (space AND time; ~gate0)
        flG = lgmax - D                                          # convention B: one absolute floor

        for nm, preds in P.items():
            pg = torch.zeros_like(gt_lin)
            for pred in preds:
                pred.reset()
                parts = []
                for i in range(0, idx.shape[0], a.chunk):
                    parts.append(predict_gates(pred, sd, idx[i:i + a.chunk].float()))
                pg += torch.cat(parts, 0) / len(preds)                          # (Nvox,10) log10
            for k in range(T.N_STEP):
                pk = float(f7[..., k].max())
                if pk <= 0:
                    continue
                lpk = math.log10(pk); fl = lpk - D
                g = torch.clamp(torch.log10(gt_lin[:, k].clamp_min(10.0 ** (lpk - 30))), min=fl)
                p = pg[:, k]
                bright = gt_lin[:, k] > 10.0 ** fl                              # GT above floor
                dark = ~bright
                # ---- censored error over ALL tissue voxels ----
                e = torch.where(bright, p - g, torch.relu(p - fl))
                sse = float((e ** 2).sum()); n = e.numel()
                sst = float(((g - g.mean()) ** 2).sum())
                acc[nm][k]["R2_cens"].append(1 - sse / sst if sst > 0 else float("nan"))
                acc[nm][k]["RMSE_cens"].append(math.sqrt(sse / n))
                # ---- classic reachable-set R2 (for continuity with the existing tables) ----
                if int(bright.sum()) >= 50:
                    pb, gb = p[bright], g[bright]
                    ss = float(((gb - gb.mean()) ** 2).sum())
                    acc[nm][k]["R2_reach"].append(1 - float(((pb - gb) ** 2).sum()) / ss if ss > 0 else float("nan"))
                # ---- UNION set (GT-bright OR model-bright), the alternative you proposed ----
                mb = p > fl
                U = bright | mb
                if int(U.sum()) >= 50:
                    pu, gu = p[U], g[U]
                    ssu = float(((gu - gu.mean()) ** 2).sum())
                    acc[nm][k]["R2_union"].append(1 - float(((pu - gu) ** 2).sum()) / ssu if ssu > 0 else float("nan"))
                    acc[nm][k]["RMSE_union"].append(float(((pu - gu) ** 2).mean().sqrt()))
                # ---- convention B: ONE absolute floor (global peak - D) for every gate ----
                # A (per-gate floor) gives every gate a full D-decade two-sided window; B fixes the
                # floor in absolute fluence, so a late gate whose own peak already sits 5 decades down
                # only gets ~3 decades of two-sided test and the rest becomes one-sided (leak-only).
                gG = torch.clamp(torch.log10(gt_lin[:, k].clamp_min(10.0 ** (lgmax - 30))), min=flG)
                brightG = gt_lin[:, k] > 10.0 ** flG
                eG = torch.where(brightG, p - gG, torch.relu(p - flG))
                sseG = float((eG ** 2).sum()); sstG = float(((gG - gG.mean()) ** 2).sum())
                acc[nm][k]["R2_censG"].append(1 - sseG / sstG if sstG > 0 else float("nan"))
                acc[nm][k]["RMSE_censG"].append(math.sqrt(sseG / eG.numel()))
                acc[nm][k]["brightG_frac"].append(float(brightG.float().mean()))
                nb = max(int(bright.sum()), 1)
                acc[nm][k]["leak_frac"].append(float(int((mb & dark).sum()) / nb))   # false-bright / GT-bright
                acc[nm][k]["miss_frac"].append(float(int((~mb & bright).sum()) / nb))
                acc[nm][k]["dark_frac"].append(float(dark.float().mean()))
        if (si + 1) % 5 == 0:
            print(f"  {si+1}/{len(scenes)}", flush=True)

    out = {}
    RATIO = {"leak_frac", "miss_frac"}          # ratios -> median across scenes
    for nm in P:
        out[nm] = {"per_gate": {}}
        for k in range(T.N_STEP):
            out[nm]["per_gate"][k] = {m: (float(np.median(v)) if m in RATIO else float(np.mean(v)))
                                      for m, v in acc[nm][k].items() if v}
    for nm in out:
        print(f"\n### {nm}")
        print(f"  {'gate':>4}{'R2_cens':>10}{'RMSE_cens':>11}{'R2_censG':>10}{'RMSEcensG':>11}{'R2_reach':>10}{'R2_union':>10}{'leak%':>9}")
        for k in range(T.N_STEP):
            d = out[nm]["per_gate"][k]
            print(f"  {k:>4}{d.get('R2_cens',float('nan')):>10.4f}{d.get('RMSE_cens',float('nan')):>11.4f}"
                  f"{d.get('R2_censG',float('nan')):>10.4f}{d.get('RMSE_censG',float('nan')):>11.4f}"
                  f"{d.get('R2_reach',float('nan')):>10.4f}{d.get('R2_union',float('nan')):>10.4f}"
                  f"{d.get('leak_frac',float('nan'))*100:>9.2f}")
        ms = [out[nm]["per_gate"][k] for k in range(T.N_STEP)]
        print(f"  {'MEAN':>4}{np.mean([m['R2_cens'] for m in ms]):>10.4f}"
              f"{np.mean([m['RMSE_cens'] for m in ms]):>11.4f}"
              f"{np.mean([m['R2_censG'] for m in ms]):>10.4f}"
              f"{np.mean([m['RMSE_censG'] for m in ms]):>11.4f}"
              f"{np.mean([m['R2_reach'] for m in ms]):>10.4f}"
              f"{np.mean([m['R2_union'] for m in ms]):>10.4f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
