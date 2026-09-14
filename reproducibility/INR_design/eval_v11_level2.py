#!/usr/bin/env python3
"""LEVEL 2 — does the surrogate BEAT its own 5e7 training data in the dark?

Level 1 asks whether the surrogate can replace a 5e7 Monte-Carlo run inside the region that run
actually resolves. Level 2 asks the stronger question: BELOW the 5e7 detection floor -- where the
training target was censored and the model was free to extrapolate -- are the surrogate's values
closer to the truth than the 5e7 data itself?

Reference truth = 5e9 photons (100x the training budget). Verified prerequisite: the three photon
budgets share one absolute scale (peak ratio 5e9/5e7 = 0.9987-0.9997 across the four heads), so
no renormalisation is applied or needed.

THE BAND. Voxels with DEC9 = log10(peak_5e9) - log10(Phi_5e9) in [8, 10] -- i.e. the two decades
BELOW the 5e7 floor, which is exactly the region Level 1 refuses to score.

THE FAIR COMPARISON. Comparing the surrogate against a 5e7 field CLIPPED at its floor would be
rigged: a constant floor value cannot compete with a smooth extrapolation. So the band is split by
what the 5e7 run actually has:

  (A) 5e7 saw >= 1 photon  -> both predictors have a value. This is the honest head-to-head.
      RMSE(surrogate, 5e9) vs RMSE(5e7_raw, 5e9), plus a PAIRED win-rate (per voxel, whose error
      is smaller). Winning here means the surrogate denoises MC shot noise.
  (B) 5e7 saw ZERO photons -> the 5e7 run carries no information at all. Only the surrogate has a
      value; report its error and the size of this blind region. Winning here is trivial in the
      sense that there is nothing to beat, but it is the practically important half: it is the
      volume a 5e7 solve simply cannot inform.

  5e8 is carried through as the MC-convergence yardstick: it calibrates how much a 10x photon
  budget buys, so "the surrogate is as good as 5e8" is a statement with a scale attached.

Also reported: the 5e7 field as the TRAINING TARGET saw it (clipped at its floor), which is the
baseline the surrogate was actually fit against.

Usage: python eval_v11_level2.py <ckpt> [--heads bw14 scb15 sh001 sh027] [--elec C3]
"""
import os, sys, json, argparse
import numpy as np, torch, scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data_expansion")):
    sys.path.insert(0, p)
import optical_config as OC
import inr_dataset as D1
from train_inr_v3 import xyz_to_norm, sample_raw
import train_inr_v11 as T
import repro_config as RC
from make_pathfeat import path_features

SIM7 = T.SIMDIR
HIPHOT = RC.HIPHOT_DIR
FLOOR_DEC = T.FLU_FLOOR_DECADES        # 8


def load(path):
    return sio.loadmat(path)["fluence"].astype(np.float64)


@torch.no_grad()
def predict(model, cfg, sd, xyz, dev, chunk=200000):
    vs = sd["vol_shape"]; tenc = T.T_ENC.to(dev)
    out = torch.empty(xyz.shape[0], T.N_STEP, device=dev)
    for i in range(0, xyz.shape[0], chunk):
        x = xyz[i:i + chunk]
        xn = xyz_to_norm(x, vs)
        raw = sample_raw(None, xn, sd["pyramid"]).float() if cfg["use_pyramid"] else None
        opt = OC.normalize_points(D1.sample_volume(sd["prop"], x, vs))
        sf = T.src_features(x, sd["srcpos"], sd["srcdir"])
        # THIRD copy of this call, and the third to have forgotten cfg["path_tri"]. A model trained
        # with trilinear path sampling was fed nearest-neighbour gathers here too, which is why the
        # checkerboard columns disagreed with a probe built on predict_gates: different feature path,
        # different predicted peak, and therefore a different self floor.
        pf = (path_features(x, sd["srcpos"], sd["prop"], vs, cfg.get("path_nseg", 0),
                            cfg.get("path_tri", False))
              if cfg.get("path_dim", 0) else None)
        for k in range(T.N_STEP):
            xt = torch.cat([xn, tenc[k].expand(xn.shape[0], 1)], 1)
            out[i:i + chunk, k] = model.forward_feats(raw, xt, opt, sd["light"], sf, pf).squeeze(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--heads", nargs="+", default=["bw14", "scb15", "sh001", "sh027"])
    ap.add_argument("--elec", default="C3")
    ap.add_argument("--lo", type=float, default=8.0)
    ap.add_argument("--hi", type=float, default=10.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda:0")

    ck = torch.load(a.ckpt, map_location="cpu"); cfg = ck["cfg"]
    m = T.build_model(cfg).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    store = T.SceneStore(dev)
    print(f"ckpt {os.path.basename(a.ckpt)} (ep {ck.get('epoch')}, path_dim={cfg.get('path_dim',0)}, "
          f"pinn={cfg.get('lambda_pinn')})")
    print(f"band: DEC vs the 5e9 peak in [{a.lo}, {a.hi}] -- the two decades BELOW the 5e7 floor\n")

    agg = {k: [] for k in ("nA", "nB", "blind%",
                           "A_sur", "A_mc7", "A_mc8", "A_win%", "A_bias", "A_sur_debiased",
                           "B_sur", "B_mc8", "tgt_sur", "tgt_mc7")}
    for h in a.heads:
        base = f"fluence_{h}_F810_{a.elec}_r5_t10.mat"
        f7 = load(os.path.join(SIM7, base))
        f8 = load(os.path.join(HIPHOT, "p5e8", base))
        f9 = load(os.path.join(HIPHOT, "p5e9", base))
        sc = [s for s in T.discover_scenes() if s["head"] == h and s["tag"] == a.elec][0]
        sd = store.get(sc)
        vi = sd["valid_idx"].float()
        pr = predict(m, cfg, sd, vi, dev).cpu().numpy()               # (V,10) log10, UNclipped
        idx = vi.round().long().cpu().numpy()
        g = lambda F: F[idx[:, 0], idx[:, 1], idx[:, 2], :]           # tissue voxels only
        a7, a8, a9 = g(f7), g(f8), g(f9)

        peak9 = float(f9.max()); peak7 = float(f7.max())
        ok9 = a9 > 0
        dec9 = np.where(ok9, np.log10(peak9) - np.log10(np.clip(a9, 1e-300, None)), np.inf)
        band = ok9 & (dec9 >= a.lo) & (dec9 < a.hi)
        if band.sum() < 500:
            print(f"  {h}: only {int(band.sum())} voxels in band -- skipped"); continue
        l9 = np.log10(np.clip(a9, 1e-300, None))

        # (A) the 5e7 run saw at least one photon here -> honest head-to-head
        A = band & (a7 > 0) & (a8 > 0)
        # (B) the 5e7 run is BLIND here (zero photons) -> nothing to beat, but this is the volume
        #     a 5e7 solve cannot inform at all
        B = band & (a7 == 0) & (a8 > 0)

        rmse = lambda p, t, s: float(np.sqrt(np.mean((p[s] - t[s]) ** 2))) if s.sum() else np.nan
        l7 = np.where(a7 > 0, np.log10(np.clip(a7, 1e-300, None)), np.nan)
        l8 = np.where(a8 > 0, np.log10(np.clip(a8, 1e-300, None)), np.nan)
        # what the TRAINING TARGET looked like: 5e7 clipped at its own floor
        t7 = np.log10(np.clip(a7, peak7 * 10 ** (-FLOOR_DEC), None))

        e_sur_A = np.abs(pr[A] - l9[A]); e_mc7_A = np.abs(l7[A] - l9[A])
        # Is the dark-zone error a LEVEL error or a SHAPE error? Below the floor the hinge only
        # says "not above the floor" -- it never says what the value should BE -- so the model's
        # sub-floor field is unsupervised extrapolation and may be right in shape but off in
        # level. Removing a single global offset separates the two. This is a DIAGNOSTIC, not a
        # correction: the de-biased number is not a result the model can claim.
        bias = float(np.mean(pr[A] - l9[A]))
        sur_db = float(np.sqrt(np.mean((pr[A] - bias - l9[A]) ** 2))) if A.sum() else np.nan
        r = dict(
            nA=int(A.sum()), nB=int(B.sum()),
            blindpct=100.0 * float(B.sum()) / max(int(band.sum()), 1),
            A_sur=rmse(pr, l9, A), A_mc7=rmse(l7, l9, A), A_mc8=rmse(l8, l9, A),
            A_win=100.0 * float((e_sur_A < e_mc7_A).mean()) if A.sum() else np.nan,
            B_sur=rmse(pr, l9, B), B_mc8=rmse(l8, l9, B),
            tgt_sur=rmse(pr, l9, band), tgt_mc7=rmse(t7, l9, band),
            A_bias=bias, A_sur_debiased=sur_db,
        )
        print(f"  {h}: band {int(band.sum()):>7,} vox | (A) 5e7 has photons {r['nA']:>7,} "
              f"| (B) 5e7 BLIND {r['nB']:>7,} ({r['blindpct']:.0f}%)")
        for k, v in (("nA", r["nA"]), ("nB", r["nB"]), ("blind%", r["blindpct"]),
                     ("A_sur", r["A_sur"]), ("A_mc7", r["A_mc7"]), ("A_mc8", r["A_mc8"]),
                     ("A_win%", r["A_win"]), ("B_sur", r["B_sur"]), ("B_mc8", r["B_mc8"]),
                     ("tgt_sur", r["tgt_sur"]), ("tgt_mc7", r["tgt_mc7"]),
                     ("A_bias", r["A_bias"]), ("A_sur_debiased", r["A_sur_debiased"])):
            agg[k].append(v)

    M = {k: float(np.nanmean(v)) for k, v in agg.items() if v}
    print(f"\n{'':38s}{'RMSE vs 5e9 (decades)':>24}")
    print("-" * 64)
    print("(A) where the 5e7 run HAS photons  -- honest head-to-head")
    print(f"    surrogate                          {M['A_sur']:>10.3f}")
    print(f"    5e7 raw  (its own training data)   {M['A_mc7']:>10.3f}")
    print(f"    5e8      (10x photons, yardstick)  {M['A_mc8']:>10.3f}")
    print(f"    -> surrogate wins on {M['A_win%']:.0f}% of voxels (paired, per-voxel)")
    print(f"    [diagnostic] surrogate bias {M['A_bias']:+.3f} dec; RMSE after removing it "
          f"{M['A_sur_debiased']:.3f}  <- a LEVEL error, not a SHAPE error, if this drops below "
          f"the 5e7 row")
    print()
    print(f"(B) where the 5e7 run is BLIND (0 photons) -- {M['blind%']:.0f}% of the band")
    print(f"    surrogate                          {M['B_sur']:>10.3f}")
    print(f"    5e8                                {M['B_mc8']:>10.3f}")
    print(f"    5e7                                {'no value':>10s}")
    print()
    print("(C) vs the TRAINING TARGET (5e7 clipped at its floor) over the whole band")
    print(f"    surrogate                          {M['tgt_sur']:>10.3f}")
    print(f"    5e7-as-trained (constant floor)    {M['tgt_mc7']:>10.3f}")
    print("\nA surrogate RMSE below the 5e7 row in (A) means it genuinely denoised its own noisy\n"
          "training data; the 5e8 row says how much a 10x photon budget would have bought instead.")

    out = a.json or a.ckpt.replace(".pt", "_level2.json")
    json.dump({"ckpt": os.path.basename(a.ckpt), "band": [a.lo, a.hi],
               "heads": a.heads, "mean": {k: round(v, 4) for k, v in M.items()}},
              open(out, "w"), indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
