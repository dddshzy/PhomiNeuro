#!/usr/bin/env python3
"""How faithful is the surrogate's J(A,B) LANDSCAPE, measured against MCX on the same grid?

WHY. On the vertex target the surrogate's autograd refinement moved AWAY from the MC-true optimum:
the 2-degree grid point reached 1.25x the Cz dose while all three Adam refinements reached 0.99-1.12x,
and the surrogate under-predicted the grid point by 24%. A gradient is only as good as the function it
descends, so the question is not "is the gradient correct" (it is -- V17 fixed that) but "is the
landscape correct, and at what angular scale". That is what this measures.

WHAT IT DOES. One MCX run per grid node, one surrogate evaluation per grid node per seed, then:
  global   Pearson / Spearman between surrogate J and log10(MCX Phi) over the whole box, plus the
           RMSE and bias in decades. A wide box spans orders of magnitude, so a high correlation here
           is easy and means little on its own -- it is reported to bound the gross behaviour.
  local    the same statistics restricted to a window around the optimum, which is the regime a
           refinement step actually operates in and where the vertex-target failure occurred.
  ranking  does the surrogate's argmax coincide with MCX's? How far apart are they in degrees, and
           what fraction of the achievable dose does the surrogate's pick actually deliver? This is
           the decision-relevant number: a landscape can correlate well and still pick the wrong peak.

  INR_SPLIT=v16_split python landscape_fidelity_v17.py --head scb15 --offset -20 -10 \
      --simgpu 1 --nphoton 5e7
"""
import os, sys, json, math, argparse, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "data_expansion"),
          os.path.join(os.path.dirname(HERE), "pmcx_sim")):
    sys.path.insert(0, p)
import torch
import train_inr_v11 as T
import illum_opt_v16 as IO
import run_v11_mni as RV
import scipy.io as sio
import repro_config as RC
from calib_gt_angres import srcdir_from_AB, phi_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="scb15")
    ap.add_argument("--depth", type=float, default=30.0)
    ap.add_argument("--offset", nargs=2, type=int, default=[0, 0], help="off-axis column offset")
    ap.add_argument("--A", nargs=3, type=float, default=[30, 85, 10], help="start stop step")
    ap.add_argument("--B", nargs=3, type=float, default=[-180, 179, 30])
    ap.add_argument("--local-half", type=float, default=10.0, help="local window around the MCX argmax")
    ap.add_argument("--local-step", type=float, default=5.0)
    ap.add_argument("--nphoton", type=float, default=5e7)
    ap.add_argument("--nphoton-local", type=float, default=2e8)
    ap.add_argument("--simgpu", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"], help="V17 K=3000 seeds")
    ap.add_argument("--tmp", default=os.path.join(RC.RESULTS_DIR, "mcx_landscape"))
    ap.add_argument("--out", default=os.path.join(RC.RESULTS_DIR, "landscape.json"))
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    dev = torch.device(f"cuda:{a.gpu}")
    T.SRC_REF = "entry"

    sc = [s for s in T.discover_scenes() if s["head"] == a.head][0]
    vol = sio.loadmat(sc["prop"])[T.PROP_KEY].astype(np.float32)
    P1, tname, depth_mm = IO.superior_target(vol, a.depth, offset=tuple(a.offset))
    cfg = RV.get_cfg(a.head)
    n_gate = T.N_STEP
    print(f"[land] {a.head} offset={a.offset} target={P1.tolist()} ({tname}, {depth_mm:.0f} mm)", flush=True)

    As = np.arange(a.A[0], a.A[1] + 1e-9, a.A[2])
    Bs = np.arange(a.B[0], a.B[1] + 1e-9, a.B[2])
    nodes = [(float(A), float(B)) for A in As for B in Bs]
    print(f"[land] coarse grid {len(As)}x{len(Bs)} = {len(nodes)} nodes @ {a.nphoton:.0e} photons", flush=True)

    def mcx(A, B, nph, tag):
        r = RV.run_one(a.head, tag, srcdir_from_AB(A, B), nph, a.tmp, cfg,
                       n_step=n_gate, gpu_id=a.simgpu, save=False)
        return phi_at(r["fluence"], P1, n_gate)

    gt = []
    for i, (A, B) in enumerate(nodes):
        gt.append(mcx(A, B, a.nphoton, f"ld{i}"))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(nodes)}", flush=True)
    gt = np.array(gt)

    # ---- surrogate on the SAME nodes, one column per seed ----
    from illum_opt_v17 import SurrogateV17, t as tt
    sur = {}
    for s in a.seeds:
        S = SurrogateV17(f"{HERE}/inr_checkpoints_v11/inr_v17k3_s{s}_base.pt", a.head, dev,
                         target_depth=a.depth, dA=1.0, dB=1.0)
        S.base.retarget  # target is set by depth; override the voxel to match the off-axis choice
        S.base.P1 = P1
        S.base.p1 = torch.tensor(P1, dtype=torch.float32, device=dev).view(1, 3)
        from train_inr_v3 import xyz_to_norm, sample_raw
        import inr_dataset as D1, optical_config as OC
        S.base.xn = xyz_to_norm(S.base.p1, S.base.vs)
        S.base.raw = sample_raw(None, S.base.xn, S.base.pyr).float()
        S.base.opt = OC.normalize_points(D1.sample_volume(S.base.prop, S.base.p1, S.base.vs))
        with torch.no_grad():
            sur[s] = np.array([float(S.J(tt(A, dev), tt(B, dev), "int")) for A, B in nodes])
        del S; torch.cuda.empty_cache()
        print(f"    surrogate seed {s} done", flush=True)

    lg = np.log10(np.maximum(gt, 1e-30))
    res = {"head": a.head, "offset": a.offset, "target": P1.tolist(), "tissue": tname,
           "depth_mm": depth_mm, "nodes": nodes, "mcx_phi": gt.tolist(),
           "surrogate_J": {s: v.tolist() for s, v in sur.items()}, "nphoton": a.nphoton}

    print(f"\n[global] over the whole box ({len(nodes)} nodes, log10 Phi spans "
          f"{lg.max()-lg.min():.2f} decades)")
    print(f"  {'seed':<8}{'Pearson':>10}{'Spearman':>10}{'RMSE(dec)':>12}{'bias(dec)':>12}")
    from scipy.stats import spearmanr
    for s, v in sur.items():
        r = float(np.corrcoef(v, lg)[0, 1]); rs = float(spearmanr(v, lg).statistic)
        d = v - lg
        print(f"  s{s:<7}{r:>10.4f}{rs:>10.4f}{float(np.sqrt((d**2).mean())):>12.4f}{float(d.mean()):>12.4f}")
        res.setdefault("global", {})[s] = {"pearson": r, "spearman": rs,
                                           "rmse": float(np.sqrt((d ** 2).mean())), "bias": float(d.mean())}

    # ---- ranking: does the surrogate pick the right peak? ----
    i_gt = int(np.argmax(gt))
    print(f"\n[ranking] MCX argmax = node {i_gt} at A={nodes[i_gt][0]:.0f} B={nodes[i_gt][1]:.0f} "
          f"(Phi={gt[i_gt]:.3e})")
    print(f"  {'seed':<8}{'surrogate pick':>20}{'angular dist':>14}{'delivered/best':>16}{'MCX rank':>10}")
    order = np.argsort(-gt)
    for s, v in sur.items():
        i_s = int(np.argmax(v))
        dA = nodes[i_s][0] - nodes[i_gt][0]
        dB = (nodes[i_s][1] - nodes[i_gt][1] + 180) % 360 - 180
        dist = math.hypot(dA, dB)
        rank = int(np.where(order == i_s)[0][0]) + 1
        print(f"  s{s:<7}{f'A={nodes[i_s][0]:.0f} B={nodes[i_s][1]:.0f}':>20}{dist:>13.1f}°"
              f"{gt[i_s]/gt[i_gt]:>15.2f}x{rank:>10}/{len(nodes)}")
        res.setdefault("ranking", {})[s] = {"pick": nodes[i_s], "dist_deg": dist,
                                            "delivered_frac": float(gt[i_s] / gt[i_gt]), "mcx_rank": rank}

    # ---- local window around the MCX argmax, finer and with more photons ----
    A0, B0 = nodes[i_gt]
    h, st = a.local_half, a.local_step
    lnodes = [(A0 + dA, B0 + dB) for dA in np.arange(-h, h + 1e-9, st) for dB in np.arange(-h, h + 1e-9, st)]
    print(f"\n[local] {len(lnodes)} nodes within +-{h}° of the MCX argmax @ {a.nphoton_local:.0e} photons",
          flush=True)
    lgt = np.array([mcx(A, B, a.nphoton_local, f"lloc{i}") for i, (A, B) in enumerate(lnodes)])
    llg = np.log10(np.maximum(lgt, 1e-30))
    lsur = {}
    for s in a.seeds:
        S = SurrogateV17(f"{HERE}/inr_checkpoints_v11/inr_v17k3_s{s}_base.pt", a.head, dev,
                         target_depth=a.depth, dA=1.0, dB=1.0)
        S.base.P1 = P1
        S.base.p1 = torch.tensor(P1, dtype=torch.float32, device=dev).view(1, 3)
        from train_inr_v3 import xyz_to_norm, sample_raw
        import inr_dataset as D1, optical_config as OC
        S.base.xn = xyz_to_norm(S.base.p1, S.base.vs)
        S.base.raw = sample_raw(None, S.base.xn, S.base.pyr).float()
        S.base.opt = OC.normalize_points(D1.sample_volume(S.base.prop, S.base.p1, S.base.vs))
        with torch.no_grad():
            lsur[s] = np.array([float(S.J(tt(A, dev), tt(B, dev), "int")) for A, B in lnodes])
        del S; torch.cuda.empty_cache()
    print(f"  local log10 Phi spans only {llg.max()-llg.min():.3f} decades")
    print(f"  {'seed':<8}{'Pearson':>10}{'Spearman':>10}{'RMSE(dec)':>12}{'picks best?':>13}")
    j_gt = int(np.argmax(lgt))
    for s, v in lsur.items():
        r = float(np.corrcoef(v, llg)[0, 1]); rs = float(spearmanr(v, llg).statistic)
        d = v - llg; j_s = int(np.argmax(v))
        print(f"  s{s:<7}{r:>10.4f}{rs:>10.4f}{float(np.sqrt((d**2).mean())):>12.4f}"
              f"{lgt[j_s]/lgt[j_gt]:>12.2f}x")
        res.setdefault("local", {})[s] = {"pearson": r, "spearman": rs,
                                          "rmse": float(np.sqrt((d ** 2).mean())),
                                          "delivered_frac": float(lgt[j_s] / lgt[j_gt])}
    res["local_nodes"] = lnodes; res["local_mcx"] = lgt.tolist()
    res["local_sur"] = {s: v.tolist() for s, v in lsur.items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print(f"\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
