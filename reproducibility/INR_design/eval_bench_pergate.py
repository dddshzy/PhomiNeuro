#!/usr/bin/env python3
"""PER-GATE benchmark: score the time-resolved models on EACH of the 10 gates separately, instead of
on the time-integrated (CW) view.

Why this is a different question. The CW view integrates the 10 gates before scoring, so a model can
be right about the total while getting the temporal SHAPE wrong (early/late energy traded off). Only
per-gate scoring tests whether the model reproduces the time-resolved field itself -- which is the
whole point of a time-resolved surrogate.

Floor convention (this is the one place D=8 is correct): the per-gate GT really is clipped at
peak - 8 decades, where `peak` here is the PER-GATE peak (each gate normalised to its own maximum),
so no log10(10) integration lift applies.

  EVAL_SPLIT=test python eval_bench_pergate.py --gpu 0 \
      --models v14=snap_v14hero_hero_ep120.pt tr_dynunet=inr_v14tr_bench_dynunet.pt
"""
import os, sys, math, json, argparse, zlib
import numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data_expansion"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, p)
import train_inr_v11 as T
import optical_config as OC
import inr_dataset as D1
from train_inr_v3 import xyz_to_norm, sample_raw
from make_pathfeat import path_features
from eval_v11_level2 import load, SIM7
from vol_common import build_grid_inputs
from eval_benchmark_v14cw import build_predictor_v14, r2
from eval_benchmark_full import resolve

GRID = 128
FLOOR_DEC = 8.0          # per-gate GT floor: 8 decades below that gate's own peak (no integ lift)


@torch.no_grad()
def predict_gates(pred, sd, xyz, chunk=200000):
    """-> (N, 10) log10 per-gate prediction, for either family."""
    model, fam, cfg, dev = pred.model, pred.family, pred.cfg, pred.dev
    vs = sd["vol_shape"]
    if fam == "voxel_tr":
        gi = build_grid_inputs(sd, GRID, with_light=(cfg.get("in_c", 14) == 14))
        if pred._cache is None:
            pred._cache = model(gi)                                     # (1,10,g,g,g)
        xn = xyz_to_norm(xyz, vs)
        g = torch.stack([xn[:, 2], xn[:, 1], xn[:, 0]], -1).view(1, 1, 1, -1, 3)
        out = F.grid_sample(pred._cache, g, mode="bilinear", align_corners=True)   # (1,10,1,1,N)
        return out.view(10, -1).transpose(0, 1)                          # (N,10) -- NO integ_t
    # per-point families (our time-resolved INR, and the time-resolved coord baselines):
    # query each gate's time code. coord_tr takes t as a separate kwarg; ours puts it in the coord.
    is_coord = (fam == "coord_tr")
    out = torch.empty(xyz.shape[0], T.N_STEP, device=dev)
    pd = cfg.get("path_dim", 0)
    for i in range(0, xyz.shape[0], chunk):
        x = xyz[i:i + chunk]; xn = xyz_to_norm(x, vs)
        opt = OC.normalize_points(D1.sample_volume(sd["prop"], x, vs))
        sf = T.src_features(x, sd["srcpos"], sd["srcdir"])
        # cfg["path_tri"] MUST be forwarded. Omitting it left `trilinear` at its False default, so a
        # model trained with trilinear path sampling (every V17 checkpoint) was evaluated with
        # nearest-neighbour gathers -- a train/eval feature mismatch that silently penalised it.
        # eval_benchmark_v14cw.predict_field:100 always passed it; this path did not.
        pf = (path_features(x, sd["srcpos"], sd["prop"], vs, cfg.get("path_nseg", 0),
                            cfg.get("path_tri", False)) if pd else None)
        raw = (None if is_coord else
               (sample_raw(None, xn, sd["pyramid"]).float() if cfg.get("use_pyramid", True) else None))
        for k, tt in enumerate(T.T_ENC):
            tcol = torch.full((xn.shape[0], 1), float(tt), device=dev)
            if is_coord:
                out[i:i + chunk, k] = model.forward_feats(None, xn, opt, sd["light"], sf,
                                                          path=pf, t=tcol).squeeze(1)
            else:
                xt = torch.cat([xn, tcol], 1)
                out[i:i + chunk, k] = model.forward_feats(raw, xt, opt, sd["light"], sf, pf).squeeze(1)
    return out


def linear_metrics(pl, gl):
    """pl, gl: LINEAR fluence, already normalised to the gate peak (so gl in (10^-D, 1])."""
    out = {}
    e = pl - gl
    ss = float(((gl - gl.mean()) ** 2).sum())
    out["R2"] = float(1 - (e ** 2).sum() / ss) if ss > 0 else np.nan
    out["RMSE"] = float(np.sqrt((e ** 2).mean()))
    out["medAE"] = float(np.median(np.abs(e)))
    rel = np.abs(e) / np.clip(gl, 1e-30, None)
    out["relMedAE"] = float(np.median(rel))
    out["relP90"] = float(np.percentile(rel, 90))
    out["energyRatio"] = float(pl.sum() / gl.sum()) if gl.sum() > 0 else np.nan
    rp = np.argsort(np.argsort(pl)); rg = np.argsort(np.argsort(gl))
    out["Spearman"] = float(np.corrcoef(rp, rg)[0, 1])
    # Kish effective sample size of the squared-error weights: how many voxels the metric REALLY uses
    w = gl ** 2
    out["effN"] = float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else np.nan
    out["n_reach"] = int(gl.size)
    return out


def gate_metrics(p, g, D):
    """p,g: (N,) log10 relative to that gate's peak, already clamped to -D."""
    d = p - g
    reach = g > -D
    res = dict(n=int(p.size))
    if reach.sum() >= 50:
        pr, gr = p[reach], g[reach]
        res["R2"] = r2(pr, gr)
        res["RMSE"] = float(np.sqrt(((pr - gr) ** 2).mean()))
        res["medAE"] = float(np.median(np.abs(pr - gr)))
        rp = np.argsort(np.argsort(pr)); rg = np.argsort(np.argsort(gr))
        res["Spearman"] = float(np.corrcoef(rp, rg)[0, 1])
        res["gt_std"] = float(gr.std())
    A = p > -D; B = g > -D
    ni = int((A & B).sum())
    res["DICE"] = 2 * ni / max(int(A.sum()) + int(B.sum()), 1)
    res["leak"] = int((A & ~B).sum()) / max(int(B.sum()), 1)
    dark = ~B
    if dark.sum() >= 50:
        res["darkExcess"] = float((p[dark] - g[dark]).mean())
        lf = float((p[dark] > -D + 0.5).mean())
        res["darkLeakFrac"] = lf
        # DEGENERACY 1 -- dark-zone saturation. Once a model predicts above the floor essentially
        # everywhere, the predicted set A becomes the whole volume, so DICE and leak are determined by
        # the ground truth alone and are bit-identical across architectures differing 10x in size.
        # They then say "this model floods", not "this model is better/worse than that one".
        res["saturated"] = float(lf > 0.98)
    # DEGENERACY 2 -- tiny denominator. leak = |A and not B| / |B|; at the earliest gate |B| is only a
    # few hundred voxels, so leak reaches O(100) and its scene-to-scene spread exceeds its mean. Report
    # it with the denominator, and prefer the median across scenes.
    res["n_reach"] = int(B.sum())
    res["leak_illcond"] = float(int(B.sum()) < 2000)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--elecs", nargs="+", default=["Cz", "C3"])
    ap.add_argument("--points", type=int, default=200000)
    ap.add_argument("--heads", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    # REFERENCE FRAME for the per-gate window. This is not cosmetic: it changes which voxels each
    # gate is scored on, and the two frames answer different questions.
    #   pergate (default): every gate is renormalised to ITS OWN peak, floor at peak_k - D. Each gate
    #     therefore gets a DIFFERENT absolute window, and a late gate's window sits far below an early
    #     gate's. Answers "how well is the shape of this gate reproduced, on its own terms".
    #   global: one window for all gates, [LFM_all - D, LFM_all], where LFM_all is the scene's global
    #     max over space AND time -- i.e. exactly the reference frame the model was TRAINED in
    #     (dec = LFM - gt). Answers "how well is the field reproduced where it actually carries dose".
    # Under `global` the natural floor is D=7 rather than 8.
    ap.add_argument("--ref", choices=["pergate", "global"], default="pergate")
    # DEPLOYMENT floor for the PREDICTION. In production the surrogate's output is clamped at
    # (its OWN predicted peak - D) and everything below is discarded -- the model never sees the GT
    # peak. Scoring the prediction clamped at the GT-derived threshold therefore evaluates an object
    # that is never shipped. `self` evaluates the artefact as deployed; it also makes the metric
    # sensitive to PEAK-CALIBRATION error (a mis-estimated peak puts the whole cut line in the wrong
    # place), which the common threshold hides. `common` keeps the old behaviour for comparison.
    ap.add_argument("--pred-floor", choices=["self", "common"], default="self")
    # DOMAIN. log10 is the training/regression domain and weights all 7-8 decades comparably.
    # linear is the DOSE domain -- but there fluence spans 10^7, so a squared-error metric is
    # dominated almost entirely by the handful of brightest near-source voxels: linear R2 is
    # effectively "did you get the peak right", with an effective sample size of order tens, not
    # 10^4. Reported alongside relative-error metrics (relMedAE/relP90), which stay informative
    # across the whole range, plus effN = Kish effective sample size to make the concentration
    # explicit rather than implicit.
    ap.add_argument("--domain", choices=["log", "linear"], default="log")
    ap.add_argument("--floor", type=float, default=None, help="override D (default 8 pergate / 7 global)")
    a = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    split = os.environ.get("EVAL_SPLIT", "test")
    D = a.floor if a.floor is not None else (8.0 if a.ref == "pergate" else 7.0)

    P = {}
    for spec in a.models:
        nm, ck = spec.split("=", 1)
        # multi-seed: "a.pt,b.pt" or a glob -> one predictor per seed, metrics averaged with std
        preds = []
        for fp in resolve(ck):
            pr, npar = build_predictor_v14(fp, dev)
            preds.append(pr)
        P[nm] = (preds, npar)
        print(f"  {nm}: {len(preds)} seed(s), family={preds[0].family}, {npar:,} params", flush=True)

    store = T.SceneStore(dev)
    te = [s for s in T.discover_scenes() if T.label_of(s["head"]) == split and s["tag"] in a.elecs
          and not s["head"].startswith(("oa", "oas"))]
    if a.heads:
        te = [s for s in te if s["head"] in a.heads]
    print(f"[per-gate] split={split} scenes={len(te)} gates={T.N_STEP} ref={a.ref} D={D:g}\n", flush=True)

    acc = {nm: {k: {m: [] for m in ("R2", "RMSE", "medAE", "DICE", "leak", "darkExcess", "darkLeakFrac",
                                    "saturated", "leak_illcond", "n_reach", "Spearman", "gt_std", "gain", "relMedAE", "relP90", "energyRatio", "effN")}
                for k in range(T.N_STEP)} for nm in P}
    for si, s in enumerate(te):
        sd = store.get(s); vs = sd["vol_shape"]
        f7 = load(os.path.join(SIM7, f"fluence_{s['head']}_F810_{s['tag']}_r5_t10.mat"))  # (X,Y,Z,10)
        idx = sd["valid_idx"].float()
        n = min(a.points, idx.shape[0])
        # deterministic point set: seeded per scene, so EVERY model -- including models scored in
        # separate runs/groups -- is measured on the exact same voxels. An unseeded randperm
        # made group-to-group comparisons carry an extra sampling difference.
        _g = torch.Generator(device=idx.device); _g.manual_seed(zlib.crc32((s['head'] + s['tag']).encode()) % (2**31))
        ii = idx[torch.randperm(idx.shape[0], device=idx.device, generator=_g)[:n]]
        jj = ii.round().long().cpu().numpy()
        gt_lin = f7[jj[:, 0], jj[:, 1], jj[:, 2], :T.N_STEP]                  # (N,10) linear
        for nm, (preds, _) in P.items():
          for pred in preds:
            pred.reset()
            pg = predict_gates(pred, sd, ii).cpu().numpy()                     # (N,10) log10 absolute
            lfm_all = math.log10(float(f7[..., :T.N_STEP].max()))   # global max over space AND time
            lfm_pred = float(pg.max())                              # model's OWN peak (space AND time)
            for k in range(T.N_STEP):
                gk = gt_lin[:, k]
                pk_lin = float(f7[..., k].max())
                if pk_lin <= 0:
                    continue
                lpk = lfm_all if a.ref == "global" else math.log10(pk_lin)
                # absolute log10 thresholds; the metric is shift-invariant so the subtraction below
                # is only bookkeeping -- what matters is that each side is cut at its OWN threshold.
                thr_p = (lfm_pred - D) if a.pred_floor == "self" else (lpk - D)
                g = np.log10(np.clip(gk, 10.0 ** (lpk - 12), None)) - lpk       # rel. to chosen ref
                p = np.maximum(pg[:, k], thr_p) - lpk                            # DEPLOYMENT clamp
                g = np.maximum(g, -D)
                res_gain = lfm_pred - lpk
                if a.domain == "linear":
                    rch = g > -D
                    _m = linear_metrics(10.0 ** p[rch], 10.0 ** g[rch]) if rch.sum() >= 50 else {}
                else:
                    _m = gate_metrics(p, g, D)
                _m["gain"] = res_gain
                for mk, v in _m.items():
                    if mk in acc[nm][k]:
                        acc[nm][k][mk].append(v)
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(te)} scenes", flush=True)

    out = {}
    for nm in P:
        out[nm] = {"params": P[nm][1], "family": P[nm][0][0].family,
                   "n_seeds": len(P[nm][0]), "per_gate": {}}
        for k in range(T.N_STEP):
            out[nm]["per_gate"][k] = {m: float(np.nanmedian(v) if m == "leak" else np.nanmean(v))
                                      for m, v in acc[nm][k].items() if v}
            out[nm]["per_gate"][k].update({m + "_std": float(np.nanstd(v))
                                           for m, v in acc[nm][k].items() if v})

    hdr = ["R2", "RMSE", "medAE", "Spearman", "n_reach", "gain"]
    for nm in P:
        print("\n" + "=" * 92)
        print(f"### {nm}  ({out[nm]['family']}, {out[nm]['params']:,} params)   ref={a.ref} D={D:g} predfloor={a.pred_floor} domain={a.domain}")
        print(f"{'gate':>5}{'t(ns)':>8}" + "".join(f"{h:>14}" for h in hdr))
        print("-" * 92)
        for k in range(T.N_STEP):
            g = out[nm]["per_gate"].get(k, {})
            row = "".join(f"{g[h]:>14.4f}" if h in g else f"{'-':>14}" for h in hdr)
            print(f"{k:>5}{0.1 + 0.2 * k:>8.1f}" + row)
        gs = [out[nm]["per_gate"][k] for k in range(T.N_STEP) if out[nm]["per_gate"].get(k)]
        if gs:
            print(f"{'MEAN':>13}" + "".join(
                f"{np.nanmean([x[h] for x in gs if h in x]):>14.4f}" for h in hdr))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
