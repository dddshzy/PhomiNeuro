#!/usr/bin/env python3
"""MULTI-REGION benchmark — the paper-grade scorer. Extends eval_benchmark_v14cw with the region matrix
that makes DARK-ZONE fidelity visible in the headline metrics (R2/SSIM/PSNR/RMSE/median error), plus a
comprehensive metric set and multi-seed mean+-std aggregation.

REGIONS (per scene, anchored to the GT CW peak; D = decades of dynamic range):
  reach{D}  voxels with GT within D decades of the peak (the classic reachable view; pred floor-clamped)
  whole{D}  ALL tissue voxels, BOTH pred and GT clamped to floor = peak-D. A non-leaking surrogate sits
            ON the floor in the dark zone (~0 error); a leaking one pays for every false-bright voxel, so
            dark-zone false light finally shows up in R2/RMSE/SSIM/PSNR instead of being invisible.
  dark{D}   the complement (GT at/below floor) alone -- isolates leak magnitude.

INTEGRITY: the clamp is applied IDENTICALLY to every model, and reach{D} is always reported next to
whole{D}. The dark-inclusive view is an ADDITION, never a replacement -- see the plan's guardrail.

  EVAL_SPLIT=test python eval_benchmark_full.py --gpu 3 --regions 6 8 \
     --models v14=snap_v14hero_hero_ep120.pt ours=snap_v10t2hero_hero_ep120.pt \
              segresnet='inr_v10t2bm_bench_segresnet*.pt' --out ../viz/out/v14bench/full.json
Multi-seed: give a comma-separated list or a glob per model; metrics are reported mean+-std over seeds.
"""
import os, sys, math, json, glob, argparse
import numpy as np, torch
from skimage.metrics import structural_similarity as ssim_fn
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data_expansion"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, p)
import train_inr_v11 as T
import inr_dataset as D1
from eval_comprehensive import gamma2d
from eval_v11_level2 import load, SIM7
from eval_benchmark_v14cw import build_predictor_v14, r2, CKV3, CKV11
from baselines import count_params

GRID = 128
# metrics that are ratios with a possibly-tiny denominator -> aggregate across scenes by median
RATIO_METRICS = {"energy_ratio", "leak", "miss", "relMed", "relP95"}


class RawCWStore:
    """SceneStore + RAW (un-floored) CW ground truth, so any floor D can be applied downstream.
    sim_dir/gate_suffix let the same scorer read the 5e9 high-photon subset (Phase D)."""

    def __init__(self, dev, sim_dir=SIM7, suffix="r5_t10"):
        self.dev = dev; self.S = T.SceneStore(dev); self.sim = sim_dir; self.suf = suffix

    def get(self, s):
        sd = self.S.get(s)
        f = os.path.join(self.sim, f"fluence_{s['head']}_F810_{s['tag']}_{self.suf}.mat")
        cw = load(f).sum(-1)                                            # (X,Y,Z) linear CW
        fmax = float(cw.max())
        sd["cw_lin"] = cw; sd["fmax"] = fmax; sd["lpk"] = math.log10(fmax)
        # log10 with a very low guard floor -- the REAL floor is applied per-region below
        sd["logflu"] = torch.from_numpy(
            np.log10(np.clip(cw, fmax * 1e-12, None)).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0).to(self.dev)
        return sd


def _core(p, t, D, csf=None):
    """p,t: (N,) log10 already clamped to the region floor. Full metric set."""
    d = p - t
    rel = np.abs(10.0 ** np.clip(d, -30, 30) - 1.0)
    out = dict(R2=r2(p, t), RMSE=float(np.sqrt((d ** 2).mean())), MSE=float((d ** 2).mean()),
               MAE=float(np.abs(d).mean()), medAE=float(np.median(np.abs(d))),
               relMed=float(np.median(rel)), relP95=float(np.percentile(rel, 95)),
               bias=float(d.mean()), medBias=float(np.median(d)), n=int(p.shape[0]))
    if csf is not None and csf.any():
        out["csf"] = float(np.median(rel[csf]))
    return out


def _detect(p, t, thr):
    """Reachable-set detection at threshold thr (log10). Mirrors eval_level1_reach.detect."""
    A = p > thr; B = t > thr
    ni = int((A & B).sum()); nA = int(A.sum()); nB = int(B.sum())
    return dict(DICE=2 * ni / max(nA + nB, 1), IoU=ni / max(int((A | B).sum()), 1),
                leak=int((A & ~B).sum()) / max(nB, 1), miss=int((B & ~A).sum()) / max(nB, 1))


def region_metrics(pr, gt, dec, csf, mua, D):
    """All three region views for one scene at dynamic range D. pr/gt: (N,) log10 raw; dec = lpk-gt.

    FLOOR-CONVENTION HAZARD (this bit us twice -- read before changing --regions):
    A model can only predict as low as the floor its TRAINING TARGET was clipped to. Per-gate targets
    are floored at peak-8, but summing 10 floored gates lifts the CW floor by log10(10) = exactly one
    decade, so ANY model whose output is an integral of per-gate predictions bottoms out at peak-7,
    not peak-8. Scoring such a model in the dark zone at D=8 charges it a full decade of "leak" on
    EVERY sub-floor voxel, which saturates leak/leakFrac/DICE: they stop measuring the model and
    instead return the same constant for every architecture (the tell is bit-identical values across
    models plus excess ~= +1.00). Use D=7 for anything scored on the CW/time-integrated view; D=8 is
    only valid for genuinely per-gate scoring. `dark_saturated` below flags the failure automatically.
    """
    floor = 0.0                                        # work in "decades below peak" space: v = value-lpk
    res = {}
    prc = np.maximum(pr, -D); gtc = np.maximum(gt, -D)  # clamp BOTH to the floor (identical for all models)
    reach = dec < D                                     # GT-defined reachable set
    dark = ~reach
    # --- reach{D}: classic view (GT-reachable voxels only) ---
    if reach.sum() >= 50:
        m = _core(prc[reach], gtc[reach], D, csf[reach] if csf is not None else None)
        m.update(_detect(prc[reach], gtc[reach], -D))
        res[f"reach{D:g}"] = m
    # --- whole{D}: ALL tissue voxels, both clamped -> dark-zone leak becomes visible ---
    m = _core(prc, gtc, D, csf)
    m.update(_detect(prc, gtc, -D))
    Egt = float((mua * 10.0 ** gtc).sum()); Ep = float((mua * 10.0 ** prc).sum())
    m["energy_ratio"] = Ep / Egt if Egt > 0 else float("nan")
    res[f"whole{D:g}"] = m
    # --- dark{D}: the sub-floor complement alone (pure leak magnitude) ---
    if dark.sum() >= 50:
        dm = prc[dark] - gtc[dark]                      # gtc[dark] == -D exactly
        lf = float((prc[dark] > -D + 0.5).mean()); ex = float(dm.mean())
        # Saturation guard: leakFrac pinned at 1 with excess ~= +1 decade means the model's floor sits
        # one decade above D (the per-gate -> integrated CW lift), so the dark-zone metrics are a
        # constant, not a measurement. Flagged per scene and surfaced in the printed table.
        res[f"dark{D:g}"] = dict(RMSE=float(np.sqrt((dm ** 2).mean())), MAE=float(np.abs(dm).mean()),
                                 medAE=float(np.median(np.abs(dm))), leakFrac=lf, excess=ex,
                                 dark_saturated=float(lf > 0.98 and abs(ex - 1.0) < 0.15),
                                 n=int(dark.sum()))
    return res


@torch.no_grad()
def volumetric(pred, sd, dev, Ds, chunk=200000):
    """Dense 3-D metrics on the tissue bounding box for EVERY requested dynamic range Ds.
    The expensive part -- predicting the field over the whole tissue volume -- is done ONCE and
    shared across all D (clamping is the only D-dependent step), instead of re-predicting per D."""
    vs = sd["vol_shape"]; lpk = sd["lpk"]
    tis = (sd["prop"][0, 1] > 0).cpu().numpy()
    xs, ys, zs = np.where(tis)
    x0, x1 = xs.min(), xs.max() + 1; y0, y1 = ys.min(), ys.max() + 1; z0, z1 = zs.min(), zs.max() + 1
    sub = tis[x0:x1, y0:y1, z0:z1]
    gg = np.stack(np.where(sub), 1) + np.array([x0, y0, z0])
    xyz = torch.tensor(gg, dtype=torch.float32, device=dev)
    pr = np.empty(xyz.shape[0], dtype=np.float32)
    pred.reset()
    for i in range(0, xyz.shape[0], chunk):
        pr[i:i + chunk] = pred.predict(sd, xyz[i:i + chunk]).cpu().numpy()
    gt = sd["cw_lin"][gg[:, 0], gg[:, 1], gg[:, 2]]
    gt = np.log10(np.clip(gt, sd["fmax"] * 1e-12, None)).astype(np.float32)
    li = (gg[:, 0] - x0, gg[:, 1] - y0, gg[:, 2] - z0)
    Praw = np.full(sub.shape, -99.0, np.float32); Graw = np.full(sub.shape, -99.0, np.float32)
    Praw[li] = pr - lpk; Graw[li] = gt - lpk           # decades below peak, UNCLAMPED
    return {D: _vol_at(Praw, Graw, sub, D, sd, x0) for D in Ds}


def _vol_at(Praw, Graw, sub, D, sd, x0):
    """Dense metrics at one dynamic range D, from the once-predicted unclamped fields."""
    P = np.maximum(Praw, -D); G = np.maximum(Graw, -D)
    ss, ssmap = ssim_fn(G, P, data_range=D, full=True)
    ss = float(ss)
    mse = float(np.mean((P[sub] - G[sub]) ** 2))
    psnr = 10 * math.log10(D ** 2 / max(mse, 1e-12))
    # ---- STRUCTURE-SENSITIVE metrics -------------------------------------------------------
    # Magnitude-averaged metrics (R2/RMSE/SSIM over the floor-dominated volume) reward a smooth
    # low-amplitude error field and therefore favour blurry grid CNNs, while DICE/gamma reward
    # correct geometry and favour a sharp INR. These probe the geometry directly, in the
    # R2/RMSE family, to test whether the DICE/gamma advantage is reproducible there.
    gP = np.gradient(P); gG = np.gradient(G)
    mgP = np.sqrt(sum(g ** 2 for g in gP)); mgG = np.sqrt(sum(g ** 2 for g in gG))
    gpv, ggv = mgP[sub], mgG[sub]                      # (not `gg` -- that holds voxel coords above)
    ssg = ((ggv - ggv.mean()) ** 2).sum()
    grad_r2 = float(1 - ((gpv - ggv) ** 2).sum() / ssg) if ssg > 0 else float("nan")
    grad_rmse = float(np.sqrt(((gpv - ggv) ** 2).mean()))
    # high-pass (detail) error: remove the smooth component both fields share
    hpP = P - ndi.gaussian_filter(P, 2.0); hpG = G - ndi.gaussian_filter(G, 2.0)
    hp_rmse = float(np.sqrt(((hpP[sub] - hpG[sub]) ** 2).mean()))
    # SSIM restricted to the BRIGHT (reachable) band instead of the floor-dominated whole head.
    # NOTE this one did NOT reproduce the DICE/gamma ordering -- SSIM stays magnitude-dominated even
    # when the region is restricted; kept because negative results belong in the record.
    reach_m = sub & (G > -min(6.0, D))
    ssim_reach = float(ssmap[reach_m].mean()) if reach_m.sum() > 100 else float("nan")
    # multi-level isodose-surface agreement (generalises the single-threshold DICE). Levels beyond
    # the clamp floor are degenerate (everything sits at -D), so only L <= D is emitted.
    iso = {}
    for lv in range(1, 9):
        if lv > D:
            continue
        A = (P > -float(lv)) & sub; B = (G > -float(lv)) & sub
        iso[f"isoDICE{lv}"] = float(2 * (A & B).sum() / max(A.sum() + B.sum(), 1))
    # gamma + DICE on the source-plane slice of the same clamped volumes (2-D, as in the classic panel)
    sx = int(np.clip(int(sd["srcpos"][0].item()) - x0, 0, sub.shape[0] - 1))
    msl = sub[sx]
    if msl.sum() >= 50:
        Pl = np.where(msl, 10.0 ** P[sx], 0.0); Gl = np.where(msl, 10.0 ** G[sx], 0.0)
        g = float(gamma2d(Gl, Pl, msl))
        thr = Gl[msl].max() * 0.1
        mp = (Pl >= thr) & msl; mt = (Gl >= thr) & msl
        dice = 2 * (mp & mt).sum() / max(mp.sum() + mt.sum(), 1)
    else:
        g, dice = float("nan"), float("nan")
    return dict(vSSIM=ss, vPSNR=psnr, gamma=g, sliceDICE=float(dice),
                gradR2=grad_r2, gradRMSE=grad_rmse, hpRMSE=hp_rmse, ssimReach=ssim_reach, **iso)


def resolve(spec):
    """'a.pt,b.pt' or a glob -> list of existing checkpoint paths (multi-seed)."""
    out = []
    for tok in spec.split(","):
        cands = []
        for base in ("", CKV11, CKV3):
            pat = os.path.join(base, tok) if base else tok
            cands += sorted(glob.glob(pat)) if any(c in tok for c in "*?[") else \
                     ([pat] if os.path.exists(pat) else [])
            if cands:
                break
        if not cands:
            raise FileNotFoundError(tok)
        out += cands
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True, help="name=ckpt[,ckpt2|glob] ...")
    ap.add_argument("--regions", type=float, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--elecs", nargs="+", default=["Fp1", "F7", "Fz", "C3", "Cz", "T4", "Pz", "O2"])
    ap.add_argument("--points", type=int, default=150000, help="tissue voxels sampled per scene")
    ap.add_argument("--vol-scenes", type=int, default=1, help="scenes per head getting dense 3-D metrics")
    ap.add_argument("--sim-dir", default=SIM7); ap.add_argument("--suffix", default="r5_t10")
    ap.add_argument("--heads", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    split = os.environ.get("EVAL_SPLIT", "test")
    store = RawCWStore(dev, a.sim_dir, a.suffix)

    # build every (model, seed) predictor once; they stay resident so GT is loaded only once per scene
    P = {}
    for spec in a.models:
        nm, pat = spec.split("=", 1)
        P[nm] = []
        for ck in resolve(pat):
            pr, npar = build_predictor_v14(ck, dev)
            P[nm].append((os.path.basename(ck), pr, npar))
        print(f"  {nm}: {len(P[nm])} seed(s), family={P[nm][0][1].family}, {P[nm][0][2]:,} params", flush=True)

    te = [s for s in T.discover_scenes() if T.label_of(s["head"]) == split and s["tag"] in a.elecs]
    if a.heads:
        te = [s for s in te if s["head"] in a.heads]
    else:
        # OASIS heads are an external-capability DEMO, not benchmark subjects (user directive), and
        # they have no high-photon GT. They are labelled "test" so they can never leak into training,
        # which means they must be excluded here explicitly rather than implicitly.
        te = [s for s in te if not s["head"].startswith("oa")]
    print(f"[full-bench] split={split} scenes={len(te)} regions={a.regions} points={a.points}\n", flush=True)

    acc = {nm: {ck: {} for ck, _, _ in P[nm]} for nm in P}      # acc[model][seed][region][metric] = [vals]
    seen = {}
    for si, s in enumerate(te):
        sd = store.get(s); vs = sd["vol_shape"]; lpk = sd["lpk"]
        idx = sd["valid_idx"].float()
        n = min(a.points, idx.shape[0])
        ii = idx[torch.randperm(idx.shape[0], device=idx.device)[:n]]
        jj = ii.round().long()
        gt = D1.sample_volume(sd["logflu"], ii, vs).squeeze(1).cpu().numpy() - lpk   # rel. to peak
        dec = -gt
        csf = sd["csf"][jj[:, 0], jj[:, 1], jj[:, 2]].cpu().numpy() > 0.5
        mua = D1.sample_volume(sd["prop"], ii, vs)[:, 0].cpu().numpy()
        k = seen.get(s["head"], 0); seen[s["head"]] = k + 1
        for nm in P:
            for ck, pred, _ in P[nm]:
                pred.reset()
                pr = pred.predict(sd, ii).cpu().numpy() - lpk
                for D in a.regions:
                    for rk, m in region_metrics(pr, gt, dec, csf, mua, D).items():
                        for mk, v in m.items():
                            acc[nm][ck].setdefault(rk, {}).setdefault(mk, []).append(v)
                if k < a.vol_scenes:
                    for D, vm in volumetric(pred, sd, dev, a.regions).items():
                        for mk, v in vm.items():
                            acc[nm][ck].setdefault(f"whole{D:g}", {}).setdefault(mk, []).append(v)
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(te)} scenes", flush=True)

    # aggregate: per seed -> mean over scenes; then mean+-std over seeds
    out = {}
    for nm in P:
        per_seed = {}
        for ck, _, _ in P[nm]:
            # RATIO-type metrics (energy_ratio, leak, ...) are unbounded above and can explode on a
            # single scene whose GT total is tiny, so the arithmetic mean across scenes is not robust:
            # one outlier scene drove tr_unet's energy_ratio to 524.8 while a direct recomputation of
            # the SAME checkpoint gives ~0.6. Aggregate those by median, everything else by mean.
            per_seed[ck] = {rk: {mk: float(np.nanmedian(v) if mk in RATIO_METRICS
                                           else np.nanmean(v)) for mk, v in mm.items()}
                            for rk, mm in acc[nm][ck].items()}
        regions = sorted({rk for d in per_seed.values() for rk in d})
        agg = {}
        for rk in regions:
            keys = sorted({mk for d in per_seed.values() for mk in d.get(rk, {})})
            agg[rk] = {mk: dict(mean=float(np.nanmean([per_seed[c][rk][mk] for c in per_seed if mk in per_seed[c].get(rk, {})])),
                                std=float(np.nanstd([per_seed[c][rk][mk] for c in per_seed if mk in per_seed[c].get(rk, {})])))
                       for mk in keys}
        out[nm] = dict(family=P[nm][0][1].family, params=P[nm][0][2], n_seeds=len(per_seed),
                       per_seed=per_seed, agg=agg)

    hdr = ["R2", "RMSE", "medAE", "vSSIM", "ssimReach", "gradR2", "gradRMSE", "hpRMSE",
           "isoDICE1", "isoDICE2", "isoDICE3", "gamma", "leak", "DICE"]
    for D in a.regions:
        for view in (f"reach{D:g}", f"whole{D:g}"):
            print("\n" + "=" * 104); print(f"### {view}")
            print(f'{"model":12}{"seeds":>6} | ' + " ".join(f"{h:>13}" for h in hdr))
            print("-" * 104)
            for nm, r in out.items():
                if view not in r["agg"]:
                    continue
                g = r["agg"][view]
                cells = []
                for h in hdr:
                    if h in g:
                        cells.append(f"{g[h]['mean']:7.3f}±{g[h]['std']:5.3f}")
                    else:
                        cells.append(f"{'-':>13}")
                print(f"{nm:12}{r['n_seeds']:>6} | " + " ".join(cells))
        print("\n" + "-" * 104); print(f"### dark{D:g} (sub-floor leak)")
        print(f'{"model":12} | {"RMSE":>13} {"medAE":>13} {"leakFrac":>13} {"excess":>13}')
        sat = []
        for nm, r in out.items():
            g = r["agg"].get(f"dark{D:g}")
            if not g:
                continue
            flag = ""
            if g.get("dark_saturated", {}).get("mean", 0) > 0.5:
                sat.append(nm); flag = "  <-- SATURATED"
            print(f"{nm:12} | " + " ".join(f"{g[k]['mean']:7.3f}±{g[k]['std']:5.3f}"
                                          for k in ("RMSE", "medAE", "leakFrac", "excess")) + flag)
        if sat:
            print(f"  !! dark{D:g} is INVALID for {sat}: their floor sits ~1 decade above peak-{D:g} "
                  f"(per-gate targets integrated to CW). Score these at D={D-1:g} instead.")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
