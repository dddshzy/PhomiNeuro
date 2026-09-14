#!/usr/bin/env python3
"""Two TARGETED metrics, because the standard panel cannot see what these two loss terms are for.

WHY THIS EXISTS.
  lambda_lin adds mean((10^d - 1)^2) on the shallow band, i.e. it penalises error in LINEAR space,
  where the few brightest voxels dominate. Every metric in the main panel (R2, RMSE, medAE, even the
  0-2-decade-band RMSE) is computed on log10 values over the whole reachable set, where the many dim
  voxels dominate. So the panel is structurally blind to peak accuracy, and reading "lambda_lin does
  nothing" off it is unsound.
  lambda_mono penalises d(log Phi)/dr > 0 -- fluence rising with distance from the source -- in the
  DEEP band (dec >= mono_dec_min). checkerboard_metric measures speck isolation and above-floor leak,
  which are consequences, not the violation itself, and it defaults to gates 2/4/6.

WHAT IS MEASURED (per gate, then gate-averaged).
  peak_logerr   |log10(pred) - log10(GT)| at the GT peak voxel of that gate. Direct peak fidelity.
  peak_ratio    10^(pred-GT) at the same voxel: >1 over-predicts the peak, <1 under-predicts.
  topK_relerr   median |10^(pred-GT) - 1| over the K brightest GT voxels -- the LINEAR relative
                error lambda_lin actually optimises.
  mono_viol     fraction of deep-band points whose prediction RISES when stepping one voxel FURTHER
                from the source centre, i.e. the discrete form of the monotonicity the term forbids.
                The centre is the entry point for src_ref=entry models, matching training.

  INR_SPLIT=v16_split EVAL_SPLIT=test python eval_v17_targeted.py --gpu 0 \
      --models hero=inr_checkpoints_v11/inr_v17_s0_base.pt --out out.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--elecs", nargs="+", default=["Cz"])
    ap.add_argument("--topk", type=int, default=100, help="brightest GT voxels for the linear-space error")
    ap.add_argument("--deep-dec", type=float, default=5.0, help="deep band for the monotonicity test "
                                                               "(matches --mono-dec-min)")
    ap.add_argument("--npts", type=int, default=60000, help="deep-band points sampled per scene")
    ap.add_argument("--step", type=float, default=1.0, help="radial step in voxels for the mono test")
    ap.add_argument("--floor", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu); dev = torch.device(f"cuda:{a.gpu}")
    split = os.environ.get("EVAL_SPLIT", "test")

    P = {}
    for spec in a.models:
        nm, ck = spec.split("=", 1)
        P[nm] = build_predictor_v14(ck, dev)[0]
        print(f"  {nm}: family={P[nm].family} src_ref={P[nm].cfg.get('src_ref','source')}", flush=True)

    # Match the main panel's scene set exactly: eval_bench_pergate.py and diag_pergate.py both drop
    # the OASIS heads, so including them here would put this column on a different footing from every
    # other column of the ablation table.
    scenes = [s for s in T.discover_scenes() if T.label_of(s["head"]) == split and s["tag"] in a.elecs
              and not s["head"].startswith(("oa", "oas"))]
    print(f"[targeted] split={split} scenes={len(scenes)} topk={a.topk} deep_dec={a.deep_dec}", flush=True)

    KEYS = ("peak_logerr", "peak_ratio", "topK_relerr", "mono_viol")
    acc = {nm: {k: {m: [] for m in KEYS} for k in range(T.N_STEP)} for nm in P}
    store = T.SceneStore(dev)

    for si, s in enumerate(scenes):
        sd = store.get(s)
        f7 = load(os.path.join(SIM7, f"fluence_{s['head']}_F810_{s['tag']}_r5_t10.mat"))
        idx = sd["valid_idx"]                                   # tissue voxels (N,3) float
        jj = idx.round().long().cpu().numpy()
        gt_lin = torch.from_numpy(np.ascontiguousarray(
            f7[jj[:, 0], jj[:, 1], jj[:, 2], :T.N_STEP])).to(dev).float()     # (N,10)

        # radial centre = what the model was trained to measure r from
        centre = sd["srcpos"].to(dev).float()

        # deep-band point subsample, seeded per scene so every model sees the same points
        g = torch.Generator(device=idx.device)
        g.manual_seed(zlib.crc32((s["head"] + s["tag"]).encode()) % (2 ** 31))
        perm = torch.randperm(idx.shape[0], device=idx.device, generator=g)

        # Deep-band point set, built ONCE as the union over gates. The naive version selected a
        # per-gate deep set and called predict_gates inside the gate loop, which returns all ten
        # gates every time and throws nine away -- eleven full forward passes per scene instead of
        # two, i.e. ~10x the cost of this whole stage. Taking the union here and masking per gate
        # afterwards keeps the per-gate definition of the deep band EXACTLY as before.
        lpk_all = [math.log10(float(f7[..., k].max())) if float(f7[..., k].max()) > 0 else None
                   for k in range(T.N_STEP)]
        deep_any = torch.zeros(idx.shape[0], dtype=torch.bool, device=gt_lin.device)
        dec_all = torch.full_like(gt_lin, -1.0)
        for k in range(T.N_STEP):
            if lpk_all[k] is None:
                continue
            dec_all[:, k] = lpk_all[k] - torch.log10(gt_lin[:, k].clamp_min(10.0 ** (lpk_all[k] - 30)))
            deep_any |= (dec_all[:, k] >= a.deep_dec)
        du = perm[deep_any[perm]][:a.npts]                       # capped union, same points per model
        if du.numel() >= 100:
            xu = idx[du].float()
            uu = xu - sd["srcpos"].to(dev).float().view(1, 3)
            rhat_u = uu / uu.norm(dim=1, keepdim=True).clamp_min(1e-6)
            xin_u = xu - a.step * rhat_u                         # one step CLOSER to the source
        else:
            xin_u = None

        # Only a tiny, known set of voxels is ever read: one GT-peak voxel per gate, the top-K
        # brightest per gate, and the capped deep-band union. Predicting the whole tissue volume
        # (4.4M voxels) to then index ~121k of them is a 36x waste, and at 380 scenes x 13 variants
        # that is the difference between ~30 minutes and ~13 hours. Gather the needed indices first,
        # predict once on that subset, and map back through `pos`.
        need = [du] if du.numel() >= 100 else []
        for k in range(T.N_STEP):
            if lpk_all[k] is None:
                continue
            gk = gt_lin[:, k]
            need.append(torch.argmax(gk).view(1))
            need.append(torch.topk(gk, min(a.topk, gk.numel())).indices)
        need_u = torch.unique(torch.cat(need))
        pos = torch.full((idx.shape[0],), -1, dtype=torch.long, device=need_u.device)
        pos[need_u] = torch.arange(need_u.numel(), device=need_u.device)
        xs = idx[need_u].float()

        for nm, pred in P.items():
            pred.reset()
            pg_s = predict_gates(pred, sd, xs)                                 # (M,10) log10
            pg = lambda ii, kk: pg_s[pos[ii], kk]                              # original-index view
            p_in_u = predict_gates(pred, sd, xin_u) if xin_u is not None else None
            for k in range(T.N_STEP):
                pk = float(f7[..., k].max())
                if pk <= 0:
                    continue
                lpk = math.log10(pk)
                gk = gt_lin[:, k]
                # ---- peak fidelity ----
                ip = int(torch.argmax(gk))
                d_peak = float(pg(ip, k) - math.log10(max(float(gk[ip]), 1e-30)))
                acc[nm][k]["peak_logerr"].append(abs(d_peak))
                # Clamped before exponentiating. d_peak is a log10 error, and an untrained encoder
                # produces genuinely enormous ones -- the randenc ablation reached a value whose
                # 10**d_peak raised OverflowError and killed the whole job partway through, losing
                # mono_viol, peak_logerr and topK_relerr as well. peak_ratio is a diagnostic that no
                # table reads; it must not be able to take the three that are read down with it.
                # +-30 decades is already far past any physically meaningful ratio.
                acc[nm][k]["peak_ratio"].append(10.0 ** min(max(d_peak, -30.0), 30.0))
                # ---- linear-space relative error on the brightest K ----
                K = min(a.topk, gk.numel())
                top = torch.topk(gk, K).indices
                dtop = pg(top, k) - torch.log10(gk[top].clamp_min(1e-30))
                acc[nm][k]["topK_relerr"].append(float((10.0 ** dtop - 1.0).abs().median()))
                # ---- radial monotonicity violation in the deep band ----
                # Mask this gate's deep band out of the pre-computed union, so the band is still
                # defined per gate while the forward pass was shared.
                if p_in_u is not None:
                    sel = dec_all[du, k] >= a.deep_dec
                    p_out = pg(du, k)[sel]
                    p_in = p_in_u[:, k][sel]
                if p_in_u is not None and int(sel.sum()) >= 100:
                    # violation: prediction RISES as we move away from the source
                    acc[nm][k]["mono_viol"].append(float((p_out > p_in).float().mean()))
        if (si + 1) % 5 == 0:
            print(f"  {si+1}/{len(scenes)}", flush=True)

    out = {}
    for nm in P:
        out[nm] = {"per_gate": {}}
        for k in range(T.N_STEP):
            out[nm]["per_gate"][k] = {m: float(np.median(v)) if m == "peak_ratio" else float(np.mean(v))
                                      for m, v in acc[nm][k].items() if v}
    for nm in out:
        pg = out[nm]["per_gate"]
        gm = lambda m: float(np.mean([pg[k][m] for k in range(T.N_STEP) if m in pg[k]]))
        print(f"\n### {nm}")
        print(f"  {'gate':>5}{'peak_logerr':>13}{'peak_ratio':>12}{'topK_relerr':>13}{'mono_viol':>11}")
        for k in range(T.N_STEP):
            d = pg[k]
            print(f"  {k:>5}{d.get('peak_logerr',float('nan')):>13.4f}{d.get('peak_ratio',float('nan')):>12.4f}"
                  f"{d.get('topK_relerr',float('nan')):>13.4f}{d.get('mono_viol',float('nan')):>11.4f}")
        print(f"  {'MEAN':>5}{gm('peak_logerr'):>13.4f}{gm('peak_ratio'):>12.4f}"
              f"{gm('topK_relerr'):>13.4f}{gm('mono_viol'):>11.4f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
