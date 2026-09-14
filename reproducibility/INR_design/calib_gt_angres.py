#!/usr/bin/env python3
"""How finely can the Monte-Carlo GROUND TRUTH actually resolve the incidence angle (A,B)?

WHY THIS MUST BE MEASURED, NOT ASSUMED. V10 optimised on a 2-degree grid; V17's objective is smooth
enough to step at 0.1 degrees. Neither number came from the data. Two facts about the MC pipeline
(pmcx_sim/run_v11_mni.py) decide the answer:

  * `srcdir` is a continuous float, so the beam DIRECTION varies smoothly with the angle;
  * `srcpos` comes from find_source_position, which returns an INTEGER voxel and is only then cast
    to float -- the source POSITION is quantised to the 1 mm grid. At an entry radius of ~100 mm a
    one-voxel move needs about 0.57 deg, so below that step the source does not move at all;
  * `seed` is a FIXED constant (29012392), so re-running the same angle reproduces the output bit for
    bit. The MC estimator's variance therefore CANNOT be measured by repetition -- only by changing
    the seed, which is what this script does.

On top of that sits photon noise. At a target 44 mm deep the fluence is several decades down, so the
relative standard error at 5e7 photons may well exceed the physical change produced by a 0.1 deg
step. If so, an optimum quoted to 0.1 deg is not verifiable, however smooth the surrogate is.

WHAT IT REPORTS
  sigma_MC   spread of Phi(target) across N different MC seeds at ONE fixed angle -- the noise floor.
  d(delta)   |Phi(A+delta) - Phi(A)| for delta = 0.1 .. 2 deg, along A and along B separately.
  delta_GT   the smallest delta whose change clears 2*sigma_MC, i.e. the finest angular step the
             ground truth can actually distinguish. This is the step the comparison grid and the
             MCX verification should use; the surrogate's own optimisation stays continuous.

  INR_SPLIT=v16_split python calib_gt_angres.py --head scb15 --A0 72.4 --B0 179.9 --simgpu 1
"""
import os, sys, json, math, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "data_expansion"),
          os.path.join(os.path.dirname(HERE), "pmcx_sim")):
    sys.path.insert(0, p)
import train_inr_v11 as T
import illum_opt_v16 as IO
import run_v11_mni as RV
import scipy.io as sio
import repro_config as RC


def srcdir_from_AB(A_deg, B_deg):
    """numpy twin of illum_opt_v16.srcdir_torch's srcdir (= -n, pointing INTO the head)."""
    A, B = math.radians(A_deg), math.radians(B_deg)
    horiz = math.cos(B) * np.array([0., 1., 0.]) + math.sin(B) * np.array([1., 0., 0.])
    n = math.cos(A) * horiz + math.sin(A) * np.array([0., 0., 1.])
    n = n / np.linalg.norm(n)
    return -n


def phi_at(fluence, P, n_gate):
    """Time-INTEGRATED fluence at the target voxel, the quantity the 'int' objective maximises."""
    v = np.asarray(fluence[P[0], P[1], P[2], :n_gate], dtype=np.float64)
    return float(v.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="scb15")
    ap.add_argument("--depth", type=float, default=30.0)
    ap.add_argument("--A0", type=float, required=True, help="reference angle, from the coarse optimum")
    ap.add_argument("--B0", type=float, required=True)
    ap.add_argument("--simgpu", type=int, default=1, help="pmcx gpuid, 1-INDEXED")
    ap.add_argument("--nphoton", type=float, default=5e7)
    ap.add_argument("--nseed", type=int, default=5, help="MC seeds for the noise floor")
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--tmp", default=os.path.join(RC.RESULTS_DIR, "mcx_angle_calibration"))
    ap.add_argument("--out", default=os.path.join(RC.RESULTS_DIR, "gt_angres.json"))
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)

    sc = [s for s in T.discover_scenes() if s["head"] == a.head][0]
    vol = sio.loadmat(sc["prop"])[T.PROP_KEY].astype(np.float32)
    P1, tname, depth_mm = IO.superior_target(vol, a.depth)
    print(f"[calib] head={a.head} target={P1.tolist()} ({tname}, {depth_mm:.0f} mm below vertex)", flush=True)
    print(f"[calib] reference angle A0={a.A0} B0={a.B0}, {a.nphoton:.0e} photons", flush=True)

    cfg = RV.get_cfg(a.head) if hasattr(RV, "get_cfg") else None
    if cfg is None:                                     # build via the module's own loader
        cfg = RV.build_cfg(a.head) if hasattr(RV, "build_cfg") else None
    if cfg is None:
        raise SystemExit("cannot obtain the mcx cfg from run_v11_mni; check its loader name")
    n_gate = T.N_STEP
    SEED0 = RV.SEED

    def run(A, B, seed, tag):
        RV.SEED = int(seed)                             # module global read when the cfg dict is built
        r = RV.run_one(a.head, tag, srcdir_from_AB(A, B), a.nphoton, a.tmp, cfg,
                       n_step=n_gate, gpu_id=a.simgpu, save=False)
        if r is None or "fluence" not in r:             # save=False must still return the array
            raise SystemExit("run_one returned no fluence with save=False -- inspect its return dict")
        return phi_at(r["fluence"], P1, n_gate), r.get("sim_seconds", float("nan"))

    # ---------- 1. noise floor: SAME angle, DIFFERENT seeds ----------
    print(f"\n[1] noise floor -- {a.nseed} MC seeds at the SAME angle", flush=True)
    base = []
    for i in range(a.nseed):
        phi, secs = run(a.A0, a.B0, SEED0 + 7919 * i, f"cal_seed{i}")
        base.append(phi)
        print(f"    seed {i}: Phi = {phi:.6e}  ({secs:.1f}s)", flush=True)
    base = np.array(base)
    mu, sig = float(base.mean()), float(base.std(ddof=1))
    print(f"    mu = {mu:.6e}   sigma_MC = {sig:.3e}   rel = {sig/mu*100:.2f}%", flush=True)

    # ---------- 2b. DENSE SWEEP + FIT: separate signal from noise ----------
    # Comparing single points ("is |dPhi| > 2 sigma?") cannot work here: the noise on a DIFFERENCE of
    # two independent MC runs is sqrt(2)*sigma, and testing five deltas independently invites false
    # positives. The first pass showed exactly that failure -- SNR came out 1.5 / 3.4 / 1.5 / 4.0 / 0.6
    # along A, NON-MONOTONE, which is the signature of noise, not of an angular trend. A genuine
    # signal must grow with delta.
    # So sweep densely and FIT: the fit averages the noise down, its residual RMS re-measures the
    # noise, and the fitted curve is the physical angular dependence. The resolvable step is then read
    # off the FITTED curve against the difference-noise sqrt(2)*sigma.
    def sweep_fit(axis, half=2.0, npt=21, deg=2):
        xs = np.linspace(-half, half, npt)
        ys = []
        for x in xs:
            A, B = (a.A0 + x, a.B0) if axis == "A" else (a.A0, a.B0 + x)
            phi, _ = run(A, B, SEED0, f"sw_{axis}{x:+.2f}")
            ys.append(phi)
        ys = np.array(ys)
        c = np.polyfit(xs, ys, deg)
        fit = np.polyval(c, xs)
        resid = float(np.sqrt(np.mean((ys - fit) ** 2)))
        f0 = float(np.polyval(c, 0.0))
        return xs, ys, c, resid, f0

    print(f"\n[2b] dense sweep + polynomial fit (signal vs noise)", flush=True)
    RV.SEED = SEED0
    phi0_pre, _ = run(a.A0, a.B0, SEED0, "cal_ref")
    res = {"sigma_MC": sig, "sigma_rel": sig / mu, "phi0": phi0_pre, "mu_seeds": mu,
           "target": P1.tolist(), "tissue": tname, "depth_mm": depth_mm,
           "A0": a.A0, "B0": a.B0, "nphoton": a.nphoton, "nseed": a.nseed, "deltas": {}}
    res_fit = {}
    for axis in ("A", "B"):
        xs, ys, c, resid, f0 = sweep_fit(axis)
        # difference-noise: two independent MC runs -> sqrt(2)*sigma
        dn = math.sqrt(2.0) * sig
        # smallest |delta| at which the FITTED change clears 2*difference-noise
        dres = None
        for x in np.linspace(0.05, 2.0, 40):
            if abs(np.polyval(c, x) - f0) > 2 * dn and abs(np.polyval(c, -x) - f0) > 2 * dn:
                dres = round(float(x), 2); break
        res_fit[axis] = {"coef": c.tolist(), "resid_rms": resid, "phi_fit0": f0,
                         "delta_resolvable": dres, "diff_noise": dn,
                         "sweep_x": xs.tolist(), "sweep_phi": ys.tolist()}
        print(f"    {axis}: 拟合残差RMS = {resid:.3e} (对比 seed 噪声 {sig:.3e})", flush=True)
        print(f"       拟合 Phi(0) = {f0:.4e},  |dPhi| @1deg = {abs(np.polyval(c,1.0)-f0):.3e}"
              f",  @2deg = {abs(np.polyval(c,2.0)-f0):.3e}", flush=True)
        print(f"       -> 可分辨步长(拟合曲线 vs 2*sqrt2*sigma) = "
              f"{dres if dres else '> 2.0'} deg", flush=True)
    res["sweep_fit"] = res_fit

    # ---------- 2. angular sensitivity at the DEFAULT seed ----------
    phi0 = res["phi0"]
    for axis in ("A", "B"):
        print(f"\n[2] angular sensitivity along {axis}", flush=True)
        print(f"    {'delta(deg)':>11}{'Phi':>14}{'|dPhi|':>12}{'|dPhi|/sigma':>14}{'resolvable':>12}")
        res["deltas"][axis] = []
        for d in a.deltas:
            A, B = (a.A0 + d, a.B0) if axis == "A" else (a.A0, a.B0 + d)
            phi, _ = run(A, B, SEED0, f"cal_{axis}{d}")
            dphi = abs(phi - phi0)
            ok = dphi > 2 * sig
            res["deltas"][axis].append({"delta": d, "phi": phi, "dphi": dphi,
                                        "snr": dphi / sig if sig > 0 else float("inf"),
                                        "resolvable": bool(ok)})
            print(f"    {d:>11.2f}{phi:>14.4e}{dphi:>12.3e}{dphi/sig:>14.2f}{'YES' if ok else 'no':>12}",
                  flush=True)

    # ---------- 3. the verdict ----------
    dgt = {}
    for axis in ("A", "B"):
        good = [r["delta"] for r in res["deltas"][axis] if r["resolvable"]]
        dgt[axis] = min(good) if good else None
    res["delta_GT"] = dgt
    print(f"\n[3] delta_GT  (smallest step whose change exceeds 2*sigma_MC)")
    print(f"    along A: {dgt['A'] if dgt['A'] else '> ' + str(max(a.deltas))} deg")
    print(f"    along B: {dgt['B'] if dgt['B'] else '> ' + str(max(a.deltas))} deg")
    print(f"    -> use this as the comparison-grid and MCX-verification step; the surrogate's own")
    print(f"       optimisation stays continuous (autograd), but an optimum quoted finer than this")
    print(f"       is not verifiable by the ground truth.")
    RV.SEED = SEED0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print(f"\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
