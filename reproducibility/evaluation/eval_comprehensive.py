#!/usr/bin/env python3
"""
Comprehensive multi-metric evaluation for the FM-INR fluence surrogate.

Per held-out head (then stratified scb / bw as mean +/- std over heads):
  POINTWISE (volume-sampled, log10 Phi):
    R2, RMSE, MAE ; linear rel-err median / mean / p95 ; signed bias (mean dlog)
    ; shallow(<3dec) / deep(>=3dec) rel-err
  FIELD / STRUCTURAL (source-plane slice):
    SSIM, PSNR (on log10 Phi)
  DOSE-PHYSICS (source-plane slice, gold standard):
    gamma-index pass-rate (3%/2px, 10% dose threshold)
  CLINICAL (treated-volume overlap):
    DICE of the iso-fluence region Phi >= 0.1*max (within 1 decade of peak)

Usage:  python evaluation/eval_comprehensive.py --tag v8 --variants full nofm
"""
import os, sys, json, math, argparse
import numpy as np, torch
from skimage.metrics import structural_similarity as ssim_fn

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "INR_design"))
sys.path.insert(0, os.path.join(ROOT, "data_expansion"))
import inr_dataset_v3 as D, inr_dataset as D1, optical_config as OC, v2_manifest as M
from train_inr_v3 import xyz_to_norm, sample_raw
from train_inr_0620v6 import INRv6, PYRSRC_DIR
CK = M.INR_CKPT_V3; FLOOR_DEC = 8.0
PROD_V2, PROD_V3 = M.PYRAMID_V2, M.PYRAMID_V3   # capture (FMINR_V10 already redirected these)


def load_model(tag, name, dev):
    ck = torch.load(os.path.join(CK, f"inr_{tag}_{name}.pt"), map_location=dev)
    pyr = bool(ck.get("pyramid", 1)); src = ck.get("pyrsrc", "prod")
    if pyr and src != "prod":
        b = PYRSRC_DIR[src]; M.PYRAMID_V2, M.PYRAMID_V3 = b + "/v2", b + "/v3"
    else:
        M.PYRAMID_V2, M.PYRAMID_V3 = PROD_V2, PROD_V3
    m = INRv6(clean=bool(ck.get("clean", 1)), use_pyramid=pyr).to(dev); m.load_state_dict(ck["model"]); m.eval()
    return m, pyr


@torch.no_grad()
def predict(model, pyr, sd, xyz, dev):
    vs = sd["vol_shape"]; xn = xyz_to_norm(xyz, vs)
    opt = OC.normalize_points(D1.sample_volume(sd["prop"], xyz, vs))
    raw = sample_raw(None, xn, sd["pyramid"]).float() if pyr else None
    return model.forward_feats(raw, xn, opt, sd["light"], D.src_features(xyz, sd["srcpos"], sd["srcdir"])).squeeze(1)


def gamma2d(ref, test, mask, dd_frac=0.03, dr_px=2, thr_frac=0.10):
    """2D global gamma pass-rate (linear fluence). dd=% of ref max, dr in px."""
    rmax = ref[mask].max(); dd = dd_frac * rmax
    region = mask & (ref >= thr_frac * rmax)
    if region.sum() < 10:
        return np.nan
    best = np.full(ref.shape, np.inf)
    for di in range(-3, 4):
        for dj in range(-3, 4):
            Ts = np.roll(np.roll(test, di, 0), dj, 1)
            g = np.sqrt(((Ts - ref) / dd) ** 2 + (di * di + dj * dj) / (dr_px ** 2))
            best = np.minimum(best, g)
    return float((best[region] <= 1.0).mean())


def slice_metrics(model, pyr, sd, dev):
    """Reconstruct the source-plane sagittal slice; SSIM/PSNR/gamma/DICE vs MC."""
    vs = sd["vol_shape"]; sp = sd["srcpos"].cpu().numpy().astype(int)
    sx = int(np.clip(sp[0], 0, vs[0].item() - 1))
    prop = sd["prop"]; mask3 = (prop[0, 1] > 0)            # tissue (mu_s>0)
    msl = mask3[sx].cpu().numpy()                          # (Y,Z)
    if msl.sum() < 50:
        return {}
    ys, zs = np.where(msl)
    xyz = torch.tensor(np.stack([np.full_like(ys, sx), ys, zs], 1), dtype=torch.float32, device=dev)
    pr = predict(model, pyr, sd, xyz, dev).cpu().numpy()
    gt = D1.sample_volume(sd["logflu"], xyz, vs).squeeze(1).cpu().numpy()
    lm = math.log10(sd["fmax"]); floorv = lm - FLOOR_DEC
    P = np.full(msl.shape, floorv); T = np.full(msl.shape, floorv)
    P[ys, zs] = np.clip(pr, floorv, None); T[ys, zs] = np.clip(gt, floorv, None)
    dr = lm - floorv
    ss = ssim_fn(T, P, data_range=dr)
    mse = np.mean((P[msl] - T[msl]) ** 2); psnr = 10 * math.log10(dr ** 2 / max(mse, 1e-12))
    Pl = np.where(msl, 10.0 ** P, 0.0); Tl = np.where(msl, 10.0 ** T, 0.0)
    g = gamma2d(Tl, Pl, msl)
    thr = Tl[msl].max() * 0.1; mp = (Pl >= thr) & msl; mt = (Tl >= thr) & msl
    dice = 2 * (mp & mt).sum() / max(mp.sum() + mt.sum(), 1)
    return dict(SSIM=float(ss), PSNR=float(psnr), gamma=g, DICE=float(dice))


def r2(p, t):
    ss = ((t - t.mean()) ** 2).sum(); return float(1 - ((t - p) ** 2).sum() / ss)


def eval_head(model, pyr, scenes, dev, npts=40000, maxsc=12):
    store = D.SceneStoreV3(dev); P, T, DC = [], [], []
    sm = []
    sel = scenes[::max(1, len(scenes) // maxsc)][:maxsc]
    torch.manual_seed(0)
    for k, s in enumerate(sel):
        sd = store.get(s); vs = sd["vol_shape"]; idx = sd["valid_idx"]
        ii = idx[torch.randint(0, idx.shape[0], (npts,), device=idx.device)]
        pr = predict(model, pyr, sd, ii.float(), dev)
        gt = D1.sample_volume(sd["logflu"], ii.float(), vs).squeeze(1)
        dec = math.log10(sd["fmax"]) - gt
        P.append(pr.cpu().numpy()); T.append(gt.cpu().numpy()); DC.append(dec.cpu().numpy())
        if k < 3:                                         # slice metrics on a few scenes
            sm.append(slice_metrics(model, pyr, sd, dev))
    p, t, dc = np.concatenate(P), np.concatenate(T), np.concatenate(DC)
    d = p - t; rel = np.abs(10.0 ** d - 1.0); sh = dc < 3; dp = dc >= 3
    sm = [x for x in sm if x]
    agg = lambda key: float(np.nanmean([x[key] for x in sm])) if sm else float("nan")
    return dict(R2=r2(p, t), RMSE=float(np.sqrt((d ** 2).mean())), MAE=float(np.abs(d).mean()),
                relMed=float(np.median(rel)), relMean=float(rel.mean()),
                relP95=float(np.percentile(rel, 95)), bias=float(np.median(d)),
                shallow=float(np.median(rel[sh])) if sh.any() else np.nan,
                deep=float(np.median(rel[dp])) if dp.any() else np.nan,
                SSIM=agg("SSIM"), PSNR=agg("PSNR"), gamma=agg("gamma"), DICE=agg("DICE"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v8")
    ap.add_argument("--variants", nargs="*", default=["full", "nofm"]); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)               # override output path (avoid clobbering)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(a.gpu)
    metrics = ["R2", "RMSE", "MAE", "relMed", "relMean", "relP95", "bias", "shallow", "deep",
               "SSIM", "PSNR", "gamma", "DICE"]
    out = {}
    for name in a.variants:
        model, pyr = load_model(a.tag, name, dev)
        scenes = D.discover_scenes_v3(); labels, _ = D.split_labels(scenes)
        held = {}; head_reg = {}
        for s, L in zip(scenes, labels):
            if L in ("val_head", "test_head") and s["kind"] == "base":
                held.setdefault(s["head"], []).append(s); head_reg[s["head"]] = L
        perhead = {h: eval_head(model, pyr, sc, dev) for h, sc in sorted(held.items())}
        print(f"\n================  {a.tag}  {name}  ================")
        print("head   reg   " + " ".join(f"{m:>7s}" for m in metrics))
        for h, mm in perhead.items():
            r = "TEST" if head_reg[h] == "test_head" else "val "
            print(f"{h:6s} {r:4s} " + " ".join(f"{mm[m]:7.3f}" for m in metrics))
        # aggregate separately for VAL vs locked-TEST, split scb / bw (then pooled).
        agg = {}
        groups = [("val-scb",  "val_head",  "scb"), ("val-bw",  "val_head",  "bw"),
                  ("val-all",  "val_head",  ""),    ("TEST-scb", "test_head", "scb"),
                  ("TEST-bw",  "test_head", "bw"),  ("TEST-all", "test_head", "")]
        for lab, reg, pref in groups:
            hs = [h for h in perhead if head_reg[h] == reg and h.startswith(pref)]
            if not hs:
                continue
            mean = {m: float(np.nanmean([perhead[h][m] for h in hs])) for m in metrics}
            std = {m: float(np.nanstd([perhead[h][m] for h in hs])) for m in metrics}
            agg[lab] = {"heads": hs, "mean": mean, "std": std}
            print(f"[{lab:8s} mean] " + " ".join(f"{mean[m]:7.3f}" for m in metrics))
            print(f"[{lab:8s} std ] " + " ".join(f"{std[m]:7.3f}" for m in metrics))
        out[name] = {"per_head": perhead, "regime": head_reg, "agg": agg}
    os.makedirs(M.VIZ_V3, exist_ok=True)
    outp = a.out or os.path.join(M.VIZ_V3, f"eval_comprehensive_{a.tag}.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
