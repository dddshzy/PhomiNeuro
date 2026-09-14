#!/usr/bin/env python3
"""V16 TIME-RESOLVED illumination inverse design with the differentiable surrogate.

The V10 ancestor (illum_compare_v10.py) optimised ONE scalar: the steady-state fluence at a target
voxel. V16 predicts a 10-gate tPSF there instead, so "the objective" is now a choice, and the choice
is the scientific question. Three are implemented, all over the SAME two parameters (A,B):

    int    J = log10 sum_k 10^f_k          time-integrated dose  == the CW/V10 analogue
    peak   J = smoothmax_k f_k             the peak of the tPSF at the target
    gate<k> J = f_k                        one gate (k=0 earliest .. 9 latest)

There is only ONE source, hence ONE (A,B). A "sequentially optimise gate0, then gate1, ..." schedule
therefore does NOT accumulate: each stage overwrites the previous stage's (A,B) and the answer is
whatever the LAST gate wanted. What that schedule is legitimately good for is (a) a warm-start
continuation for the peak objective and (b) tracing HOW the optimum moves with photon arrival time --
which is what `--mode diag` measures directly, and is the thing a CW surrogate cannot answer.

Physics to expect: early gates select short optical paths (few scattering events, and CSF acts as a
low-absorption pipe), late gates are diffuse and nearly isotropic, so the late-gate landscape should
be flatter and the integral should sit near the mid/late optimum. Whether the early-gate optimum is
displaced ENOUGH to matter is exactly what we measure rather than assume.

Angle convention is inherited unchanged from the dataset (train_inr_v11._illum):
    n = -srcdir = [cos A sin B, cos A cos B, sin A],  A = elevation, B = azimuth
so illum_opt.srcdir_torch is reused verbatim. Training angle coverage (19 EEG 10-20 electrodes + 45
augmentation directions) is A in [-8, 85] deg, B over the full circle -- the box is set from that, so
the optimiser never leaves the region the surrogate was actually fitted on.

  INR_SPLIT=v16_split python illum_opt_v16.py --mode diag --heads sh001 --gpu 0
"""
import os, sys, json, math, time, argparse
import numpy as np, torch, scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pmcx_sim"))
import train_inr_v11 as T
from train_inr_v3 import xyz_to_norm, sample_raw
from make_pathfeat import path_features
from illum_opt import srcdir_torch, classify, ORD
import illumination as ILL

# training angle coverage -> the optimisation box (A clamped, B periodic)
A_LO, A_HI = -8.0, 85.0
OUT_DEFAULT = os.path.join(os.path.dirname(HERE), "viz", "out", "v16", "viz1-optimize")


# --------------------------------------------------------------------------- target
def superior_target(vol, depth_mm, offset=(0, 0)):
    """Central superior column: ~depth mm below the vertex scalp at the brain centroid (x,y), snapped
    to the nearest GM/WM voxel. Identical rule to V10's so the two campaigns target the same anatomy."""
    mask = vol[..., 3] > 1.05
    mua, mus = vol[..., 0], vol[..., 1]
    lab = np.full(mua.shape, -1)
    lab[mask] = classify(mua[mask], mus[mask])
    brain = (lab == ORD.index("WM")) | (lab == ORD.index("GM"))
    cb = np.argwhere(brain).mean(0).round().astype(int)
    cb[0] += int(offset[0]); cb[1] += int(offset[1])     # off-axis column: the 10-20 grid is sparse,
    ztop = int(np.where(mask[cb[0], cb[1], :])[0].max())
    tz0 = ztop - int(depth_mm)
    zb = np.where(brain[cb[0], cb[1], :])[0]
    tz = int(zb[np.argmin(np.abs(zb - tz0))]) if zb.size else tz0
    tl = lab[cb[0], cb[1], tz]
    return np.array([cb[0], cb[1], tz]), (ORD[tl] if tl >= 0 else "bg"), float(ztop - tz)


# --------------------------------------------------------------------------- surrogate
class Surrogate:
    """f(A,B) -> (10,) log10 Phi at ONE target voxel, differentiable in (A,B).

    Everything that depends on the target only (FM pyramid features, local optics, normalised coords)
    is computed once; everything that depends on the source (light vector, src_features, path
    integrals) is rebuilt per call, because moving the source is the whole point.

    Gradient note: find_source_position marches over a voxel grid and path_features looks tissue up
    with a nearest-voxel index, so neither is differentiable in (A,B). We re-attach the differentiable
    direction the way V10 did -- srcpos = origin + |sp-origin| * n(A,B) -- which carries the gradient
    through the geometry (r, cos theta, path length) but not through which voxels the ray crosses.
    Adam is therefore validated against the gradient-free dense grid before any result is believed.
    """

    def __init__(self, ckpt, head, dev, target_depth=30.0, offset=(0, 0)):
        self.dev = dev; self.offset = offset
        ck = torch.load(ckpt, map_location=dev)
        self.cfg = ck.get("cfg", {}) or {}
        self.model = T.build_model(self.cfg).to(dev)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.path_dim = int(self.cfg.get("path_dim", 0))
        self.nseg = max(0, self.path_dim - 6)

        scenes = [s for s in T.discover_scenes() if s["head"] == head]
        if not scenes:
            raise SystemExit(f"no scenes for head {head}")
        self.scenes = {s["tag"]: s for s in scenes}
        sc = scenes[0]
        self.vol = sio.loadmat(sc["prop"])[T.PROP_KEY].astype(np.float32)      # (X,Y,Z,4)
        prop = np.transpose(self.vol, (3, 0, 1, 2))
        self.prop = torch.from_numpy(prop).unsqueeze(0).to(dev)
        X, Y, Z = prop.shape[-3:]
        self.vs = torch.tensor([X, Y, Z], device=dev)
        self.occ = (prop[1] > 0).astype(np.uint8)                              # tissue mask for ray-cast
        self.pyr = [p.to(dev) for p in torch.load(sc["pyramid"], map_location="cpu")["pyramid"]]
        self.origin = np.array(json.load(open(sc["meta"]))["origin_AC"], dtype=float)
        self.origin_t = torch.tensor(self.origin, dtype=torch.float32, device=dev)

        self.t_enc = T.T_ENC.to(dev)                                           # (10,) gate -> t
        self.n_eval = 0
        self.retarget(target_depth)

    def retarget(self, depth_mm):
        """Move the target voxel (same head/geometry). Only the target-dependent tensors are rebuilt,
        so a depth sweep costs one pyramid sample per depth, not a reload."""
        self.P1, self.tname, self.depth_mm = superior_target(self.vol, depth_mm, self.offset)
        self.p1 = torch.tensor(self.P1, dtype=torch.float32, device=self.dev).view(1, 3)
        self.xn = xyz_to_norm(self.p1, self.vs)
        self.raw = sample_raw(None, self.xn, self.pyr).float()
        import inr_dataset as D1, optical_config as OC
        self.opt = OC.normalize_points(D1.sample_volume(self.prop, self.p1, self.vs))
        return self

    # ---- source placement (non-differentiable part, cached per direction) ----
    def _srcpos_radius(self, n_np):
        sp = ILL.find_source_position(self.occ, -n_np, self.origin)
        return float(np.linalg.norm(np.asarray(sp, float) - self.origin)), np.asarray(sp, float)

    def gates(self, A_deg, B_deg):
        """(10,) log10 Phi at the target, differentiable in A_deg/B_deg (torch scalars)."""
        sdir, nout = srcdir_torch(A_deg, B_deg, self.dev)
        rm, _ = self._srcpos_radius(nout.detach().cpu().numpy())
        srcpos = self.origin_t + rm * nout                                     # differentiable in (A,B)
        Ar = A_deg * math.pi / 180.0; Br = B_deg * math.pi / 180.0
        light = torch.cat([srcpos / (self.vs.float() - 1), sdir,
                           torch.stack([torch.sin(Ar), torch.cos(Ar),
                                        torch.sin(Br), torch.cos(Br)])]).view(1, -1)
        u = self.p1 - srcpos.view(1, 3)
        rr = u.norm(dim=1, keepdim=True).clamp_min(1e-3)
        sf = torch.cat([rr / T.R_SCALE, (u * sdir.view(1, 3)).sum(1, keepdim=True) / rr], dim=1)
        path = (path_features(self.p1, srcpos, self.prop, self.vs, self.nseg)
                if self.path_dim else None)
        K = self.t_enc.shape[0]
        xyzt = torch.cat([self.xn.expand(K, -1), self.t_enc.view(K, 1)], dim=1)
        out = self.model.forward_feats(
            self.raw.expand(K, -1), xyzt, self.opt.expand(K, -1), light.expand(K, -1),
            sf.expand(K, -1), None if path is None else path.expand(K, -1))
        self.n_eval += 1
        return out.view(-1)

    # ---- objectives (all take the (10,) gate vector) ----
    @staticmethod
    def objective(f, kind, tau=0.15):
        if kind == "int":                       # log10 sum_k 10^f_k  (time-integrated dose)
            return torch.logsumexp(f * math.log(10.0), dim=0) / math.log(10.0)
        if kind == "peak":                      # smooth max over gates (tau in decades)
            return tau * torch.logsumexp(f / tau, dim=0)
        if kind.startswith("gate"):
            return f[int(kind[4:])]
        raise ValueError(kind)


# --------------------------------------------------------------------------- diagnostic
def diag(S, dA=3.0, dB=5.0):
    """Dense (A,B) landscape for EVERY gate. Answers, without any optimiser in the loop:
    does the best illumination angle depend on photon arrival time?"""
    Ag = np.arange(A_LO, A_HI + 1e-6, dA)
    Bg = np.arange(-180.0, 180.0, dB)
    land = np.zeros((T.N_STEP, len(Ag), len(Bg)), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, Av in enumerate(Ag):
            At = torch.tensor(float(Av), device=S.dev)
            for j, Bv in enumerate(Bg):
                land[:, i, j] = S.gates(At, torch.tensor(float(Bv), device=S.dev)).cpu().numpy()
    secs = time.time() - t0
    integ = np.log10(np.power(10.0, land).sum(0))                 # time-integrated landscape
    peak = land.max(0)                                            # tPSF-peak landscape
    return dict(Ag=Ag.tolist(), Bg=Bg.tolist(), land=land, integ=integ, peak=peak,
                grid_s=secs, eval_ms=secs / (len(Ag) * len(Bg)) * 1e3)


def argmax_ab(M, Ag, Bg):
    i, j = np.unravel_index(int(np.nanargmax(M)), M.shape)
    return float(Ag[i]), float(Bg[j]), float(M[i, j])


# --------------------------------------------------------------------------- optimiser
def grad_fd(S, kind, A, B, h=1.0):
    """Central finite-difference gradient of the objective. 4 extra forwards (~12 ms total)."""
    def J(a, b):
        with torch.no_grad():
            return float(Surrogate.objective(
                S.gates(torch.tensor(float(a), device=S.dev), torch.tensor(float(b), device=S.dev)), kind))
    return (J(A + h, B) - J(A - h, B)) / (2 * h), (J(A, B + h) - J(A, B - h)) / (2 * h)


def adam_fd(S, kind, A0, B0, steps=200, lr=1.5, h=1.0, log=False):
    """TEMPORARY production optimiser: Adam driven by FINITE-DIFFERENCE gradients, in raw degrees.

    Why not autograd -- ARCHIVED BUG (2026-07-25, measured on sh001 @ the grid optimum A=73,B=-170):
        autograd  dJ/dA = -0.0587      finite-diff dJ/dA = +0.0577   <- SAME MAGNITUDE, OPPOSITE SIGN
        autograd  dJ/dB = -0.0077      finite-diff dJ/dB = -0.0107   <- sign agrees
    so autograd-Adam walks DOWNHILL out of the grid optimum (J_int -3.277 -> -3.900, i.e. -0.62
    decades). Two known causes, both in the source->target path term and both to be fixed properly
    later: (1) `rm` (the scalp radius returned by find_source_position) is detached, so autograd moves
    the source on a SPHERE while the real source slides along the scalp contour; (2) path_features
    indexes tissue with `pts.round().long()`, so the change in WHICH tissue the ray crosses -- the
    term that dominates near the optimum -- has exactly zero gradient, leaving only the length-scaling
    term d(tau)/dr, which points the other way. FD costs 4 forwards (~12 ms) and is correct, so it is
    the interim answer; the surrogate's speed is what makes that affordable at all.
    """
    A, B = float(A0), float(B0)
    mA = vA = mB = vB = 0.0; b1, b2, eps = 0.9, 0.999, 1e-8
    traj = []
    for s in range(1, steps + 1):
        gA, gB = grad_fd(S, kind, A, B, h)
        mA = b1 * mA + (1 - b1) * gA; vA = b2 * vA + (1 - b2) * gA * gA
        mB = b1 * mB + (1 - b1) * gB; vB = b2 * vB + (1 - b2) * gB * gB
        A += lr * (mA / (1 - b1 ** s)) / (math.sqrt(vA / (1 - b2 ** s)) + eps)
        B += lr * (mB / (1 - b1 ** s)) / (math.sqrt(vB / (1 - b2 ** s)) + eps)
        A = float(np.clip(A, A_LO, A_HI)); B = float((B + 180) % 360 - 180)
        if log:
            with torch.no_grad():
                traj.append([s - 1, float(Surrogate.objective(
                    S.gates(torch.tensor(A, device=S.dev), torch.tensor(B, device=S.dev)), kind)), A, B])
    with torch.no_grad():
        val = float(Surrogate.objective(
            S.gates(torch.tensor(A, device=S.dev), torch.tensor(B, device=S.dev)), kind))
    return val, (A, B), traj


def adam_opt(S, kind, A0, B0, steps=200, lr=0.05, log=False):
    """Box-constrained AUTOGRAD Adam -- kept only as the control for the archived gradient bug above
    (see adam_fd). Do not use it to produce a reported optimum.
    A is squashed into [A_LO,A_HI]; B is left FREE and wrapped, because azimuth is periodic --
    V10's tanh(B/90) would have created two artificial walls at +-90 deg."""
    uA = torch.tensor(math.atanh(np.clip((A0 - A_LO) / (A_HI - A_LO) * 2 - 1, -0.999, 0.999)),
                      device=S.dev, requires_grad=True)
    uB = torch.tensor(float(B0), device=S.dev, requires_grad=True)
    o = torch.optim.Adam([uA, uB], lr=lr); traj = []
    for s in range(steps):
        A = A_LO + (A_HI - A_LO) * torch.sigmoid(uA)
        loss = -Surrogate.objective(S.gates(A, uB), kind)
        o.zero_grad(); loss.backward(); o.step()
        if log:
            with torch.no_grad():
                traj.append([s, float(-loss), float(A), float((uB + 180) % 360 - 180)])
    with torch.no_grad():
        A = A_LO + (A_HI - A_LO) * torch.sigmoid(uA); B = (uB + 180) % 360 - 180
        val = float(Surrogate.objective(S.gates(A, B), kind))
    return val, (float(A), float(B)), traj


def multistart(S, kind, steps=200, lr=0.05):
    best = (-1e9, None, None)
    for A0 in (10.0, 40.0, 70.0):
        for B0 in (-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0):
            v, ab, tj = adam_opt(S, kind, A0, B0, steps=steps, lr=lr, log=True)
            if v > best[0]:
                best = (v, ab, tj)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["diag", "opt", "depth"], default="diag")
    ap.add_argument("--depths", nargs="+", type=float,
                    default=[8.0, 14.0, 20.0, 26.0, 32.0, 40.0, 50.0],
                    help="--mode depth: target depths (mm below vertex) to sweep")
    ap.add_argument("--heads", nargs="+", default=["sh001"])
    ap.add_argument("--ckpt", default="inr_checkpoints_v11/inr_v16hero2_s1_base.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--depth", type=float, default=30.0)
    ap.add_argument("--offset", nargs=2, type=int, default=[0, 0],
                    help="(dx,dy) voxel offset of the target column from the brain centroid. The 19 "
                         "10-20 electrodes are ~30-40 deg apart, so a target under the VERTEX is "
                         "already served near-optimally by Cz; an off-axis target is where a "
                         "continuous (A,B) search can actually buy something.")
    ap.add_argument("--dA", type=float, default=3.0)
    ap.add_argument("--dB", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--objectives", nargs="+",
                    default=["int", "peak", "gate0", "gate2", "gate9"],
                    help="int=time-integrated dose (the CW/V10 analogue), peak=tPSF peak, "
                         "gate<k>=one gate. Gates 0-2 are the early window where SHALLOW voxels put "
                         "their energy, so they are the depth-selectivity handle.")
    ap.add_argument("--out", default=OUT_DEFAULT)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    for head in a.heads:
        print(f"\n===== {head} =====", flush=True)
        S = Surrogate(a.ckpt, head, dev, target_depth=a.depth, offset=tuple(a.offset))
        print(f"[{head}] target={S.P1.tolist()} ({S.tname}, {S.depth_mm:.0f} mm below vertex) "
              f"path_dim={S.path_dim}", flush=True)
        rec = dict(head=head, offset=list(a.offset), target=S.P1.tolist(), target_tissue=S.tname, depth_mm=S.depth_mm,
                   ckpt=a.ckpt, A_box=[A_LO, A_HI])

        # ---- source-placement reproduction check: re-derive a known electrode's srcpos ----
        chk = []
        for tag in ("Cz", "C3", "F4"):
            if tag not in S.scenes:
                continue
            m = json.load(open(S.scenes[tag]["meta"]))
            d = np.array(m["srcdir"], float); n = -d
            A = math.degrees(math.asin(np.clip(n[2], -1, 1))); B = math.degrees(math.atan2(n[0], n[1]))
            _, sp = S._srcpos_radius(n)
            chk.append(dict(tag=tag, A=A, B=B, srcpos_meta=list(map(float, m["srcpos"])),
                            srcpos_reproduced=list(map(float, sp)),
                            err_vox=float(np.linalg.norm(sp - np.array(m["srcpos"], float)))))
            print(f"  [check] {tag}: A={A:6.1f} B={B:7.1f}  srcpos meta={m['srcpos']} "
                  f"repro={sp.astype(int).tolist()}  err={chk[-1]['err_vox']:.2f} vox", flush=True)
        rec["srcpos_check"] = chk

        if a.mode == "depth":
            # Does the tPSF shape -- and with it the best illumination -- depend on target DEPTH?
            # Two readings per depth: (i) re-optimised for that depth, (ii) held at the DEEPEST
            # target's optimum, which isolates the shape change from the angle change.
            rows = []
            S.retarget(max(a.depths))
            D0 = diag(S, a.dA, a.dB)
            Aref, Bref, _ = argmax_ab(D0["integ"], np.array(D0["Ag"]), np.array(D0["Bg"]))
            print(f"  reference angle (deepest target, integrated) = A {Aref:.1f}, B {Bref:.1f}")
            print(f"  {'depth':>7}{'tissue':>7}{'A':>7}{'B':>8}{'peak_g':>8}{'early3':>8}"
                  f"{'peak_g@ref':>11}{'early3@ref':>11}")
            for d in sorted(a.depths):
                S.retarget(d)
                Dd = diag(S, a.dA, a.dB)
                Ad, Bd, _ = argmax_ab(Dd["integ"], np.array(Dd["Ag"]), np.array(Dd["Bg"]))
                vd, abd, _ = adam_fd(S, "int", Ad, Bd, steps=60)
                with torch.no_grad():
                    f_opt = S.gates(torch.tensor(abd[0], device=dev),
                                    torch.tensor(abd[1], device=dev)).cpu().numpy()
                    f_ref = S.gates(torch.tensor(Aref, device=dev),
                                    torch.tensor(Bref, device=dev)).cpu().numpy()
                def frac(f):
                    w = 10.0 ** f; w = w / w.sum(); return w, float(w[:3].sum())
                w_o, e3_o = frac(f_opt); w_r, e3_r = frac(f_ref)
                rows.append(dict(depth_req=d, depth_mm=S.depth_mm, tissue=S.tname,
                                 target=S.P1.tolist(), A=abd[0], B=abd[1], J=vd,
                                 tpsf=f_opt.tolist(), peak_gate=int(np.argmax(f_opt)), early3=e3_o,
                                 tpsf_at_ref=f_ref.tolist(), peak_gate_at_ref=int(np.argmax(f_ref)),
                                 early3_at_ref=e3_r))
                print(f"  {S.depth_mm:>7.0f}{S.tname:>7}{abd[0]:>7.1f}{abd[1]:>8.1f}"
                      f"{int(np.argmax(f_opt)):>8d}{e3_o*100:>7.1f}%"
                      f"{int(np.argmax(f_ref)):>11d}{e3_r*100:>10.1f}%", flush=True)
            rec["depth_sweep"] = rows; rec["ref_angle"] = [Aref, Bref]
        elif a.mode == "diag":
            D = diag(S, a.dA, a.dB)
            Ag, Bg = np.array(D["Ag"]), np.array(D["Bg"])
            print(f"  grid {len(Ag)}x{len(Bg)} in {D['grid_s']:.1f}s ({D['eval_ms']:.2f} ms/eval, "
                  f"10 gates each)", flush=True)
            print(f"  {'objective':>10}{'A':>8}{'B':>9}{'value':>10}")
            pg = []
            for k in range(T.N_STEP):
                A, B, v = argmax_ab(D["land"][k], Ag, Bg)
                pg.append(dict(gate=k, A=A, B=B, val=v))
                print(f"  {'gate'+str(k):>10}{A:>8.1f}{B:>9.1f}{v:>10.3f}")
            Ai, Bi, vi = argmax_ab(D["integ"], Ag, Bg); print(f"  {'integrated':>10}{Ai:>8.1f}{Bi:>9.1f}{vi:>10.3f}")
            Ap, Bp, vp = argmax_ab(D["peak"], Ag, Bg);  print(f"  {'tPSF-peak':>10}{Ap:>8.1f}{Bp:>9.1f}{vp:>10.3f}")
            # how far does the optimum travel across gates? (angular distance on the sphere)
            def ang(a1, b1, a2, b2):
                v1 = np.array([math.cos(math.radians(a1)) * math.sin(math.radians(b1)),
                               math.cos(math.radians(a1)) * math.cos(math.radians(b1)),
                               math.sin(math.radians(a1))])
                v2 = np.array([math.cos(math.radians(a2)) * math.sin(math.radians(b2)),
                               math.cos(math.radians(a2)) * math.cos(math.radians(b2)),
                               math.sin(math.radians(a2))])
                return math.degrees(math.acos(float(np.clip(v1 @ v2, -1, 1))))
            d_early_late = ang(pg[0]["A"], pg[0]["B"], pg[-1]["A"], pg[-1]["B"])
            d_g0_int = ang(pg[0]["A"], pg[0]["B"], Ai, Bi)
            print(f"  optimum travel: gate0->gate9 = {d_early_late:.1f} deg | gate0->integrated = {d_g0_int:.1f} deg")
            # landscape correlation vs the integrated one (are the gates redundant?)
            fi = D["integ"].ravel()
            cors = [float(np.corrcoef(D["land"][k].ravel(), fi)[0, 1]) for k in range(T.N_STEP)]
            print("  corr(gate_k landscape, integrated): " + " ".join(f"{c:.3f}" for c in cors))
            rec.update(per_gate_opt=pg, integ_opt=[Ai, Bi, vi], peak_opt=[Ap, Bp, vp],
                       travel_g0_g9_deg=d_early_late, travel_g0_int_deg=d_g0_int,
                       corr_gate_vs_integ=cors, grid_s=D["grid_s"], eval_ms=D["eval_ms"],
                       Ag=D["Ag"], Bg=D["Bg"])
            np.savez_compressed(os.path.join(a.out, f"diag_{head}.npz"),
                                land=D["land"], integ=D["integ"], peak=D["peak"],
                                Ag=Ag, Bg=Bg, target=S.P1)
        else:
            # V10-equivalent flow, with the grid standing in for the (currently unusable) autograd
            # global search: dense landscape -> FD-Adam polish from its argmax -> trajectory + tPSF.
            D = diag(S, a.dA, a.dB)
            Ag, Bg = np.array(D["Ag"]), np.array(D["Bg"])
            rec.update(Ag=D["Ag"], Bg=D["Bg"], grid_s=D["grid_s"], eval_ms=D["eval_ms"],
                       n_grid=len(Ag) * len(Bg))
            print(f"  grid {len(Ag)}x{len(Bg)} in {D['grid_s']:.1f}s ({D['eval_ms']:.2f} ms/eval)",
                  flush=True)
            LAND = {"int": D["integ"], "peak": D["peak"],
                    **{f"gate{k}": D["land"][k] for k in range(T.N_STEP)}}
            print(f"  {'objective':>10}{'A_grid':>8}{'B_grid':>9}{'->':>4}{'A_opt':>8}{'B_opt':>9}"
                  f"{'J':>9}{'dJ_refine':>11}{'peak_gate':>10}")
            for kind in a.objectives:
                A0, B0, v0 = argmax_ab(LAND[kind], Ag, Bg)
                t0 = time.time()
                v, ab, tj = adam_fd(S, kind, A0, B0, steps=a.steps, log=True)
                if v < v0:                       # never report worse than the grid it started from
                    v, ab, tj = v0, (A0, B0), tj
                # naive-start convergence curve (V10 plotted one from a "frontal" guess)
                _, _, tj_naive = adam_fd(S, kind, 45.0, 0.0, steps=a.steps, log=True)
                with torch.no_grad():
                    f = S.gates(torch.tensor(ab[0], device=dev), torch.tensor(ab[1], device=dev))
                tpsf = f.cpu().numpy().tolist(); pk = int(np.argmax(tpsf))
                rec[f"opt_{kind}"] = dict(A=ab[0], B=ab[1], val=v, grid_A=A0, grid_B=B0, grid_val=v0,
                                          refine_gain=v - v0, secs=time.time() - t0,
                                          traj=tj, traj_naive=tj_naive, tpsf=tpsf, peak_gate=pk)
                print(f"  {kind:>10}{A0:>8.1f}{B0:>9.1f}{'->':>4}{ab[0]:>8.1f}{ab[1]:>9.1f}"
                      f"{v:>9.3f}{v - v0:>11.3f}{pk:>10d}", flush=True)
            # where does the tPSF energy sit? (gates 0-2 = the shallow-dominated early window)
            f_int = np.array(rec["opt_int"]["tpsf"]); w = 10.0 ** f_int; w = w / w.sum()
            rec["gate_energy_frac"] = w.tolist()
            rec["early3_frac"] = float(w[:3].sum())
            print(f"  gate energy fraction: " + " ".join(f"{x:.3f}" for x in w) +
                  f"   | gates0-2 = {w[:3].sum()*100:.1f}%", flush=True)
            np.savez_compressed(os.path.join(a.out, f"land_{head}.npz"),
                                land=D["land"], integ=D["integ"], peak=D["peak"],
                                Ag=Ag, Bg=Bg, target=S.P1)

        fn = os.path.join(a.out, f"{a.mode}_{head}.json")
        json.dump(rec, open(fn, "w"), indent=2, default=float)
        print(f"  -> {fn}", flush=True)


if __name__ == "__main__":
    main()
