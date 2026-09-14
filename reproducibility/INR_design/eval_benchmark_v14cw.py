#!/usr/bin/env python3
"""V14 STEADY-STATE (CW) benchmark — the same 13-metric panel + heat-table as evaluation/eval_benchmark,
but on the V14 MNI/electrode dataset with the CW (time-integrated 5e7) GT, scoring the V10t2 benchmark
panel on IDENTICAL query voxels:

  ours   = INRv7-CW (FM pyramid + path, common decade-MSE loss)   -- our architecture+encoder
  coord  = rff / siren  (+ path; NO pyramid)                       -- fair physical inputs, no encoder
  voxel  = unet / segresnet / fno / dynunet / unetr  (14-ch grid: mu_a,mu_s,src-blob,cosθ,+light10)
  operator = deeponet (14-ch branch)

  EVAL_SPLIT=test python eval_benchmark_v14cw.py --models ours=snap_... rff=inr_v10t2bm_rff.pt ... \
      --out work/results/benchmark.json
"""
import os, sys, math, json, time, argparse
import numpy as np, torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data_expansion"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, p)
import train_inr_v11 as T
import optical_config as OC
import inr_dataset as D1
from make_pathfeat import path_features
from train_inr_v3 import xyz_to_norm, sample_raw
from eval_comprehensive import gamma2d
from eval_v11_level2 import load, SIM7
from vol_common import build_grid_inputs
from baselines import make_coord_model, make_voxel_model, DeepONet, count_params
import repro_config as RC

METRICS =["R2", "RMSE", "MAE", "relMed", "relP95", "bias", "shallow", "deep", "csf",
           "SSIM", "PSNR", "gamma", "DICE"]
FLOOR_DEC = 8.0
GRID = 128
CKV3 = os.path.join(RC.INR_CHECKPOINT_DIR, "v3")
CKV11 = os.path.join(RC.INR_CHECKPOINT_DIR, "v11")


def r2(p, g):
    ss = ((g - g.mean()) ** 2).sum()
    return float(1 - ((p - g) ** 2).sum() / ss) if ss > 0 else float("nan")


class V14CWStore:
    """train_inr_v11.SceneStore + CW ground truth (time-integral of the raw 5e7 gates)."""
    def __init__(self, dev):
        self.dev = dev; self.S = T.SceneStore(dev)

    def get(self, s):
        sd = self.S.get(s)
        f7 = load(os.path.join(SIM7, f"fluence_{s['head']}_F810_{s['tag']}_r5_t10.mat"))  # (X,Y,Z,10) linear
        cw = f7.sum(-1)                                                                    # CW = time-integral
        fmax = float(cw.max()); floor = max(1e-30, fmax * 10.0 ** (-FLOOR_DEC))
        logcw = np.log10(np.clip(cw, floor, None)).astype(np.float32)
        sd["logflu"] = torch.from_numpy(logcw).unsqueeze(0).unsqueeze(0).to(self.dev)      # (1,1,X,Y,Z) CW
        sd["fmax"] = fmax
        return sd


@torch.no_grad()
def predict_field(pred, sd, xyz, chunk=200000):
    """Unified CW predictor -> (N,) log10 CW for every family."""
    model, fam, cfg, dev = pred.model, pred.family, pred.cfg, pred.dev
    vs = sd["vol_shape"]
    if fam in ("voxel", "voxel_tr", "operator"):
        gi = build_grid_inputs(sd, GRID, with_light=(cfg.get("in_c", 14) == 14))   # 4- or 14-ch
        if fam == "voxel":
            if pred._cache is None:
                pred._cache = model(gi)                                                    # (1,1,g,g,g)
            xn = xyz_to_norm(xyz, vs)
            g = torch.stack([xn[:, 2], xn[:, 1], xn[:, 0]], -1).view(1, 1, 1, -1, 3)
            return F.grid_sample(pred._cache, g, mode="bilinear", align_corners=True).view(-1)
        if fam == "voxel_tr":
            # TIME-RESOLVED grid baseline: one forward -> (1,10,g,g,g) per-gate log10; sample all 10
            # gates at the query points, then integ_t to CW so it is scored on the SAME CW GT as the
            # steady models (integ_t = base-10 log-sum-exp; the GT here is the 5e7 time-integral).
            if pred._cache is None:
                pred._cache = model(gi)                                                    # (1,10,g,g,g)
            xn = xyz_to_norm(xyz, vs)
            g = torch.stack([xn[:, 2], xn[:, 1], xn[:, 0]], -1).view(1, 1, 1, -1, 3)
            gts = F.grid_sample(pred._cache, g, mode="bilinear", align_corners=True)       # (1,10,1,1,N)
            gts = gts.view(10, -1).transpose(0, 1)                                          # (N,10)
            return T.integ_t(gts).squeeze(-1)                                              # (N,) log10 CW
        if pred._cache is None:
            pred._cache = model.branch_code(gi)                                            # (1,K)
        out = torch.empty(xyz.shape[0], device=dev)
        for i in range(0, xyz.shape[0], chunk):
            out[i:i + chunk] = model(pred._cache, xyz_to_norm(xyz[i:i + chunk], vs)).view(-1)
        return out
    # per-point families: coord baseline (+path), ours (INRv7-CW, +path +pyramid), or v10t1
    # (INRv6-CW: FM pyramid, NO path, NO time -- the V10 hero architecture on the V14-CW data)
    is_ours = (fam == "ours"); is_v10t1 = (fam == "v10t1"); pd = cfg.get("path_dim", 0)
    # (coord / coord_tr / v10t1 / ours all fall through to the shared per-point loop below)
    out = torch.empty(xyz.shape[0], device=dev)
    for i in range(0, xyz.shape[0], chunk):
        x = xyz[i:i + chunk]; xn = xyz_to_norm(x, vs)
        opt = OC.normalize_points(D1.sample_volume(sd["prop"], x, vs))
        sf = T.src_features(x, sd["srcpos"], sd["srcdir"])
        pf = (path_features(x, sd["srcpos"], sd["prop"], vs, cfg.get("path_nseg", 0),
                            cfg.get("path_tri", False)) if pd else None)
        if is_ours:
            raw = sample_raw(None, xn, sd["pyramid"]).float() if cfg.get("use_pyramid", True) else None
            if cfg.get("cw"):                                       # CW (steady-state): one fixed-t forward
                xt = torch.cat([xn, torch.full((xn.shape[0], 1), T.T_CW, device=dev)], 1)
                out[i:i + chunk] = model.forward_feats(raw, xt, opt, sd["light"], sf, pf).squeeze(1)
            else:                                                  # TIME-RESOLVED: query 10 gates -> integ to CW
                gts = []
                for tt in T.T_ENC:
                    xt = torch.cat([xn, torch.full((xn.shape[0], 1), float(tt), device=dev)], 1)
                    gts.append(model.forward_feats(raw, xt, opt, sd["light"], sf, pf).squeeze(1))
                out[i:i + chunk] = T.integ_t(torch.stack(gts, -1)).squeeze(-1)   # (n,10)->(n,) log10 CW
        elif is_v10t1:
            raw = sample_raw(None, xn, sd["pyramid"]).float()      # INRv6: 3D xn, pyramid, no path/time
            out[i:i + chunk] = model.forward_feats(raw, xn, opt, sd["light"], sf).squeeze(1)
        elif fam == "coord_tr":
            # time-resolved coord baseline: query all 10 gates, then integ_t to CW (same treatment
            # as our time-resolved INR, so both are scored against the identical CW ground truth)
            gts = []
            for tt in T.T_ENC:
                tcol = torch.full((xn.shape[0], 1), float(tt), device=dev)
                gts.append(model.forward_feats(None, xn, opt, sd["light"], sf, path=pf,
                                               t=tcol).squeeze(1))
            out[i:i + chunk] = T.integ_t(torch.stack(gts, -1)).squeeze(-1)
        else:
            out[i:i + chunk] = model.forward_feats(None, xn, opt, sd["light"], sf, path=pf).squeeze(1)
    return out


class Predictor:
    def __init__(self, model, family, cfg, dev):
        self.model = model; self.family = family; self.cfg = cfg; self.dev = dev; self._cache = None

    def predict(self, sd, xyz):
        return predict_field(self, sd, xyz)

    def reset(self):
        self._cache = None


def build_predictor_v14(path, dev):
    for cand in (path, os.path.join(CKV11, path), os.path.join(CKV3, path)):
        if os.path.exists(cand):
            fp = cand; break
    else:
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ck = torch.load(fp, map_location=dev); fam = ck.get("family"); arch = ck.get("arch")
    cfg = ck.get("cfg", {}) or {}
    # V17 CONSISTENCY GUARD. The source-dependent features (light[0:3], src_features, path_features)
    # are built from sd["srcpos"], which SceneStore defines according to T.SRC_REF. A checkpoint
    # trained with src_ref="entry" scored against source-referenced features is silently wrong -- the
    # same class of mistake that made the raw-encoder ablation read -2.16 before it was caught. So we
    # pin T.SRC_REF from the checkpoint and refuse to mix conventions in one run.
    _ref = cfg.get("src_ref", "source")
    _prev = getattr(build_predictor_v14, "_src_ref", None)
    if _prev is not None and _prev != _ref:
        raise SystemExit(f"src_ref mismatch: this run already pinned '{_prev}' but {path} wants "
                         f"'{_ref}'. Score the two conventions in SEPARATE runs.")
    build_predictor_v14._src_ref = _ref
    if T.SRC_REF != _ref:
        print(f"  [src_ref] SceneStore switched to '{_ref}' (was '{T.SRC_REF}')", flush=True)
        T.SRC_REF = _ref

    # PYRAMID-PROVENANCE GUARD, same failure class as the src_ref guard above and it has already
    # fired for real. Every evaluator RE-SAMPLES the FM pyramid at scoring time from $INR_PYRDIR;
    # the checkpoint only records which BUFFER it trained on. Score a model trained on
    # bufmm_*_k1000_raw (un-fine-tuned encoder) while INR_PYRDIR points at the fine-tuned pyramids
    # and the network gets a feature set it has never seen: the V18 b7 ablation read R2 = -11.06,
    # energy ratio 35.9 and top-K relerr inf, while its own training was healthy (val 0.0868 against
    # the reference model's 0.0826). Nothing errored -- the numbers were simply wrong, and would have
    # been published as "the un-fine-tuned encoder is catastrophic". The true cost is -0.0100 R2.
    # The buffer name carries the provenance suffix, so require the pyramid directory to match it.
    if cfg.get("use_pyramid", True):
        _buf = os.path.basename(str(cfg.get("buffer", "")))
        _pyr = os.environ.get("INR_PYRDIR", "")
        for _suffix in ("_raw", "_random"):
            if _buf.endswith(_suffix) != _pyr.rstrip("/").endswith(_suffix):
                raise SystemExit(
                    f"pyramid provenance mismatch: {os.path.basename(fp)} trained on buffer "
                    f"'{_buf}' but INR_PYRDIR='{_pyr}'. A '{_suffix}' buffer must be scored against "
                    f"a '{_suffix}' pyramid directory and vice versa, or the model is fed features "
                    f"it never saw. Set INR_PYRDIR accordingly and rerun.")
    if fam in ("coord", "coord_tr"):
        td = int(ck.get("time_dim", 0))
        # num_freq_t is absent on every pre-V16 coord checkpoint -> None -> raw scalar t, exactly
        # as they were trained; V16 checkpoints carry the band limit they were trained with.
        # drop_feats MUST be forwarded. The input-aligned baselines (coordrff2 / coordsiren2) drop
        # the optical block and the sin/cos angles, so rebuilding without it produces a model that is
        # 8 columns too wide -- load_state_dict then raises on the shape, which is at least loud, but
        # the flag has to be here for those checkpoints to load at all.
        m = make_coord_model(arch, path_dim=int(ck.get("path_dim", 0)), time_dim=td,
                             num_freq_t=ck.get("num_freq_t"),
                             drop_feats=ck.get("drop_feats", ())).to(dev)
        m.load_state_dict(ck["model"]); m.eval()
        cfg = dict(cfg); cfg["time_dim"] = td
        return Predictor(m, "coord_tr" if td else "coord", cfg, dev), count_params(m)
    if fam == "voxel":
        ic = int(ck.get("in_c", 14)); oc = int(ck.get("out_c", 1))
        m = make_voxel_model(arch, in_c=ic, out_c=oc).to(dev)
        m.load_state_dict(ck["model"]); m.eval()
        cfg = dict(cfg); cfg["in_c"] = ic          # so predict_field builds the matching channels
        # a 10-gate grid model is scored via the time-resolved branch (grid_sample all gates -> integ_t)
        fam2 = "voxel_tr" if (ck.get("timeres") or oc == 10) else "voxel"
        return Predictor(m, fam2, cfg, dev), count_params(m)
    if fam == "operator":
        ic = int(ck.get("in_c", 14))
        m = DeepONet(in_c=ic).to(dev)
        m.load_state_dict(ck["model"]); m.eval()
        cfg = dict(cfg); cfg["in_c"] = ic
        return Predictor(m, "operator", cfg, dev), count_params(m)
    if "pinn_case" in ck:                                          # V10 INRv6 checkpoint (v10t1)
        from train_inr_0620v6 import INRv6
        m = INRv6(clean=True, use_pyramid=bool(ck.get("pyramid", 1)),
                  mlp_width=int(cfg.get("mlp_width", 512)), mlp_extra=int(cfg.get("mlp_extra", 0))).to(dev)
        m.load_state_dict(ck["model"]); m.eval()
        return Predictor(m, "v10t1", cfg, dev), count_params(m)
    # else: ours = INRv7-CW
    m = T.build_model(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    return Predictor(m, "ours", cfg, dev), count_params(m)


@torch.no_grad()
def slice_metrics(pred, sd, dev):
    vs = sd["vol_shape"]; sp = sd["srcpos"].cpu().numpy().astype(int)
    sx = int(np.clip(sp[0], 0, vs[0].item() - 1))
    msl = (sd["prop"][0, 1] > 0)[sx].cpu().numpy()
    if msl.sum() < 50:
        return {}
    ys, zs = np.where(msl)
    xyz = torch.tensor(np.stack([np.full_like(ys, sx), ys, zs], 1), dtype=torch.float32, device=dev)
    pred.reset(); pr = pred.predict(sd, xyz).cpu().numpy()
    gt = D1.sample_volume(sd["logflu"], xyz, vs).squeeze(1).cpu().numpy()
    lm = math.log10(sd["fmax"]); floorv = lm - FLOOR_DEC
    P = np.full(msl.shape, floorv); Tg = np.full(msl.shape, floorv)
    P[ys, zs] = np.clip(pr, floorv, None); Tg[ys, zs] = np.clip(gt, floorv, None)
    dr = lm - floorv
    ss = ssim_fn(Tg, P, data_range=dr)
    mse = np.mean((P[msl] - Tg[msl]) ** 2); psnr = 10 * math.log10(dr ** 2 / max(mse, 1e-12))
    Pl = np.where(msl, 10.0 ** P, 0.0); Tl = np.where(msl, 10.0 ** Tg, 0.0)
    g = gamma2d(Tl, Pl, msl)
    thr = Tl[msl].max() * 0.1; mp = (Pl >= thr) & msl; mt = (Tl >= thr) & msl
    dice = 2 * (mp & mt).sum() / max(mp.sum() + mt.sum(), 1)
    return dict(SSIM=float(ss), PSNR=float(psnr), gamma=float(g), DICE=float(dice))


@torch.no_grad()
def eval_head(pred, store, scenes, dev, npts=40000, max_dec=8.0, clip_floor=True):
    P, Tg, DC, CF = [], [], [], []; sm = []
    for k, s in enumerate(scenes):
        sd = store.get(s); vs = sd["vol_shape"]; idx = sd["valid_idx"].float()
        ii = idx[torch.randint(0, idx.shape[0], (npts,), device=idx.device)]
        pred.reset(); pr = pred.predict(sd, ii)
        if clip_floor:                                     # DEPLOYMENT-consistent: sub-floor is clamped
            pr = torch.clamp(pr, min=math.log10(sd["fmax"]) - FLOOR_DEC)   # up to the floor (GT already is)
        gt = D1.sample_volume(sd["logflu"], ii, vs).squeeze(1)
        dec = math.log10(sd["fmax"]) - gt
        jj = ii.long(); cf = sd["csf"][jj[:, 0], jj[:, 1], jj[:, 2]].float()
        P.append(pr.cpu().numpy()); Tg.append(gt.cpu().numpy())
        DC.append(dec.cpu().numpy()); CF.append(cf.cpu().numpy())
        if k < 2:
            sm.append(slice_metrics(pred, sd, dev))
    p, t, dc, cf = (np.concatenate(x) for x in (P, Tg, DC, CF))
    if max_dec < 99:                                   # restrict to the reachable (dec<max_dec) band
        keep = dc < max_dec; p, t, dc, cf = p[keep], t[keep], dc[keep], cf[keep]
    d = p - t; rel = np.abs(10.0 ** np.clip(d, -30, 30) - 1.0)
    sh = dc < 3; dp = dc >= 3; cm = cf > 0.5
    md = lambda m: float(np.median(rel[m])) if m.any() else np.nan
    sm = [x for x in sm if x]; agg = lambda key: float(np.nanmean([x[key] for x in sm])) if sm else np.nan
    md_metrics = dict(R2=r2(p, t), RMSE=float(np.sqrt((d ** 2).mean())), MAE=float(np.abs(d).mean()),
                      relMed=float(np.median(rel)), relP95=float(np.percentile(rel, 95)),
                      bias=float(np.median(d)), shallow=md(sh), deep=md(dp), csf=md(cm),
                      SSIM=agg("SSIM"), PSNR=agg("PSNR"), gamma=agg("gamma"), DICE=agg("DICE"))
    return md_metrics, p.astype(np.float32), t.astype(np.float32)   # + raw pairs for the POOLED aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True, help="name=ckpt.pt ...")
    ap.add_argument("--elecs", nargs="+", default=["Fp1", "F7", "Fz", "C3", "Cz", "T4", "Pz", "O2"])
    ap.add_argument("--points", type=int, default=40000)
    ap.add_argument("--max-dec", type=float, default=8.0,
                    help="restrict R2/RMSE/etc to the reachable band (dec<max_dec); 99=whole field.")
    ap.add_argument("--no-clip-floor", dest="clip_floor", action="store_false",
                    help="disable the deployment-consistent floor clip on predictions (default: clip).")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    split = os.environ.get("EVAL_SPLIT", "test")
    store = V14CWStore(dev)

    sc = [s for s in T.discover_scenes() if T.label_of(s["head"]) == split and s["tag"] in a.elecs]
    held = {}
    for s in sc:
        held.setdefault(s["head"], []).append(s)
    print(f"[v14cw-bench] split={split} heads={len(held)} elecs={a.elecs} points={a.points}", flush=True)

    out = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        pred, nparam = build_predictor_v14(path, dev)
        perhead = {}; PP = []; TT = []
        for h, scs in sorted(held.items()):
            md, p, t = eval_head(pred, store, scs, dev, a.points, a.max_dec, a.clip_floor)
            perhead[h] = md; PP.append(p); TT.append(t)
        ta_mean = {m: float(np.nanmean([perhead[h][m] for h in perhead])) for m in METRICS}
        ta_std = {m: float(np.nanstd([perhead[h][m] for h in perhead])) for m in METRICS}
        pp = np.concatenate(PP); tt = np.concatenate(TT); ee = pp - tt          # POOLED (all voxels)
        pooled = dict(R2=r2(pp, tt), RMSE=float(np.sqrt((ee ** 2).mean())),
                      MAE=float(np.abs(ee).mean()), bias=float(np.median(ee)), n_voxels=int(pp.shape[0]))
        out[name] = {"family": pred.family, "params": nparam, "time_field_s": 0.0,
                     "per_head": perhead,
                     "agg": {"TEST-all": {"mean": ta_mean, "std": ta_std, "pooled": pooled,
                                          "heads": list(perhead)}}}
        print(f"=== {name} ({pred.family}, {nparam:,} p) R2 mean-per-head {ta_mean['R2']:.3f} | "
              f"R2 POOLED {pooled['R2']:.3f} | DICE {ta_mean['DICE']:.3f} gamma {ta_mean['gamma']:.3f}",
              flush=True)

    print("\n" + "=" * 130)
    print(f"{'model':12s} {'params':>11s} | {'R2meanH':>8s} {'R2pool':>7s} | "
          + " ".join(f"{m:>7s}" for m in METRICS))
    print("-" * 130)
    for name, r in out.items():
        ta = r["agg"]["TEST-all"]["mean"]; po = r["agg"]["TEST-all"]["pooled"]
        print(f"{name:12s} {r['params']:>11,} | {ta['R2']:8.3f} {po['R2']:7.3f} | "
              + " ".join(f"{ta[m]:7.3f}" for m in METRICS))
    outp = a.out or str(RC.RESULTS_DIR / "v10t2bench" / "benchmark.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nsaved -> {outp}", flush=True)


if __name__ == "__main__":
    main()
