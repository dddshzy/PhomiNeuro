#!/usr/bin/env python3
"""
Illumination inverse-design via the differentiable v5 surrogate.

Maximize log10 Phi at a target voxel P1 over the two source angles (A, B) using
gradient ascent (Adam, multi-start) + a brute-force grid for verification.

P1 = anterior-most WM voxel on the central A-P line through the brain (GM|WM)
centroid (= first WM a forehead-incident photon meets), per the task spec.

Differentiability: srcdir(A,B) is analytic (torch); srcpos is made differentiable
by srcpos = origin + r_march * (-srcdir), with r_march from find_source_position
recomputed and DETACHED each step (stop-grad on the marched radius).

Usage: python illum_opt.py --head scb01 --gpu 3
Writes viz/out/v3/illumopt_<head>.json (best A,B + grid argmax + P1 coords).
"""
import os, sys, json, math, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pmcx_sim"))
import inr_dataset_v3 as D
import inr_dataset as D1
import optical_config as OC
import v2_manifest as M
from train_inr_v3 import INRv3, xyz_to_norm, sample_raw
import sim_core_v2 as SC

R_SCALE = 128.0
# 5-tissue anchors (model values) for classifying WM/GM
LAY = {"scalp": (0.0306, 11.69), "skull": (0.0110, 17.45), "CSF": (0.0026, 0.09),
       "GM": (0.0280, 7.30), "WM": (0.0920, 38.0)}
ORD = ["scalp", "skull", "CSF", "GM", "WM"]
ANCH = np.array([[np.log10(LAY[l][0]), np.log10(max(LAY[l][1], 1e-3))] for l in ORD])


def classify(mua, mus):
    a = np.log10(np.clip(mua, 1e-5, None)); s = np.log10(np.clip(mus, 1e-3, None))
    return np.argmin((np.stack([a, s], -1)[..., None, :] - ANCH[None]) ** 2 @ np.ones(2), -1)


def find_P1(vol):
    """anterior-most WM voxel on central A-P line through (GM|WM) centroid."""
    mua, mus = vol[..., 0], vol[..., 1]
    mask = vol[..., 3] > 1.05
    lab = np.full(mua.shape, -1)
    lab[mask] = classify(mua[mask], mus[mask])
    WM, GM = ORD.index("WM"), ORD.index("GM")
    brain = (lab == WM) | (lab == GM)
    cb = np.argwhere(brain).mean(0).round().astype(int)
    col = lab[cb[0], :, cb[2]]                      # along Y at (cx, :, cz)
    wmy = np.where(col == WM)[0]
    if len(wmy) == 0:                               # fallback: search small x,z window
        for dx in range(-3, 4):
            for dz in range(-3, 4):
                c = lab[cb[0]+dx, :, cb[2]+dz]; w = np.where(c == WM)[0]
                if len(w):
                    return np.array([cb[0]+dx, int(w.max()), cb[2]+dz]), cb
    return np.array([cb[0], int(wmy.max()), cb[2]]), cb


def find_target_at_depth(vol, depth_mm):
    """Central-line target at depth_mm inward from the ANTERIOR scalp surface
    (geometry-defined, angle-independent, reachable & within surrogate range).
    Returns (target_voxel, brain_centroid, tissue_label_name)."""
    mua, mus = vol[..., 0], vol[..., 1]
    mask = vol[..., 3] > 1.05
    lab = np.full(mua.shape, -1); lab[mask] = classify(mua[mask], mus[mask])
    WM, GM = ORD.index("WM"), ORD.index("GM")
    brain = (lab == WM) | (lab == GM)
    cb = np.argwhere(brain).mean(0).round().astype(int)
    colm = mask[cb[0], :, cb[2]]
    y_scalp = int(np.where(colm)[0].max())          # anterior-most tissue on the line
    ty = int(y_scalp - depth_mm)
    tlab = lab[cb[0], ty, cb[2]]
    name = ORD[tlab] if tlab >= 0 else "background"
    return np.array([cb[0], ty, cb[2]]), cb, name


def srcdir_torch(A_deg, B_deg, dev):
    A = A_deg * math.pi / 180; B = B_deg * math.pi / 180
    f = torch.tensor([0., 1., 0.], device=dev); u = torch.tensor([0., 0., 1.], device=dev)
    r = torch.tensor([1., 0., 0.], device=dev)
    horiz = torch.cos(B) * f + torch.sin(B) * r
    n = torch.cos(A) * horiz + torch.sin(A) * u
    n = n / n.norm()
    return -n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="scb01"); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=float, default=0.0,
                    help="if >0, target = central-line voxel this many mm inward from anterior scalp; "
                         "else original anterior-WM-boundary P1")
    ap.add_argument("--target", default="", help="explicit target voxel 'x,y,z' (overrides P1 selection)")
    ap.add_argument("--name", default="", help="output json name suffix (per target)")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    # model
    ck = torch.load(os.path.join(M.INR_CKPT_V3, "inr_v5_b0n1.pt"), map_location=dev)
    model = INRv3().to(dev); model.load_state_dict(ck["model"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # geometry: std vol, pyramid, cfg (for source marching)
    vol = __import__("scipy.io", fromlist=["loadmat"]).loadmat(M.std_mat(a.head))[M.PROP_KEY].astype(np.float32)
    cfg_vol, cfg_prop, mask, origin = SC.get_cfg(a.head)
    vol_size = cfg_vol.shape
    pyr = [p.to(dev) for p in torch.load(os.path.join(M.PYRAMID_V2,
           f"{a.head}_copmri_withHermiteF{M.WL}_pyramid.pt"), map_location="cpu")["pyramid"]]
    vs = torch.tensor(list(vol_size), device=dev)

    if a.target:
        P1 = np.array([int(v) for v in a.target.split(",")]); cb = P1
        print(f"[illumopt] head={a.head} explicit target={P1.tolist()}", flush=True)
    elif a.depth > 0:
        P1, cb, tname = find_target_at_depth(vol, a.depth)
        print(f"[illumopt] head={a.head} centroid={cb.tolist()} target@{a.depth:.0f}mm={P1.tolist()} tissue={tname}", flush=True)
    else:
        P1, cb = find_P1(vol)
        print(f"[illumopt] head={a.head} centroid={cb.tolist()} P1(anterior-WM)={P1.tolist()}", flush=True)

    # fixed per-point inputs at P1
    p1 = torch.tensor(P1, dtype=torch.float32, device=dev).view(1, 3)
    xn = xyz_to_norm(p1, vs)
    raw = sample_raw(None, xn, pyr).float()
    prop_t = torch.from_numpy(np.transpose(vol, (3, 0, 1, 2))).unsqueeze(0).to(dev)
    opt = OC.normalize_points(D1.sample_volume(prop_t, p1, vs))
    origin_t = torch.tensor(origin, dtype=torch.float32, device=dev)

    def logphi_P1(A_deg, B_deg):
        sdir, nout = srcdir_torch(A_deg, B_deg, dev)
        sd_np = sdir.detach().cpu().numpy()
        srcpos_np = SC.S.find_source_position(cfg_vol, vol_size, sd_np, origin)
        r_march = float(np.linalg.norm(srcpos_np - origin))
        srcpos = origin_t + r_march * nout                     # differentiable via nout(A,B)
        srcpos_norm = srcpos / (vs.float() - 1)
        Ar = A_deg * math.pi / 180; Br = B_deg * math.pi / 180
        light = torch.cat([srcpos_norm, sdir,
                           torch.stack([torch.sin(Ar), torch.cos(Ar), torch.sin(Br), torch.cos(Br)])]).view(1, -1)
        u = p1 - srcpos.view(1, 3); rr = u.norm(dim=1, keepdim=True).clamp_min(1e-3)
        sf = torch.cat([rr / R_SCALE, (u * sdir.view(1, 3)).sum(1, keepdim=True) / rr], dim=1)
        return model.forward_feats(raw, xn, opt, light, sf).squeeze()

    # ---- brute-force grid (verification) ----
    with torch.no_grad():
        Ag = np.arange(45, 90.01, 2.0); Bg = np.arange(-90, 90.01, 2.0)
        best_g = (-1e9, None)
        for Av in Ag:
            for Bv in Bg:
                v = float(logphi_P1(torch.tensor(float(Av), device=dev), torch.tensor(float(Bv), device=dev)))
                if v > best_g[0]:
                    best_g = (v, (float(Av), float(Bv)))
    print(f"[grid] best logPhi(P1)={best_g[0]:.4f} at A,B={best_g[1]}", flush=True)

    # ---- multi-start gradient ascent ----
    def opt_from(A0, B0):
        uA = torch.tensor(math.atanh(np.clip((A0-45)/45*2-1, -0.999, 0.999)), device=dev, requires_grad=True)
        uB = torch.tensor(math.atanh(np.clip(B0/90, -0.999, 0.999)), device=dev, requires_grad=True)
        opt_ = torch.optim.Adam([uA, uB], lr=a.lr)
        for _ in range(a.steps):
            A = 45 + 45 * torch.sigmoid(uA); B = 90 * torch.tanh(uB)   # A in (45,90), B in (-90,90)
            loss = -logphi_P1(A, B)
            opt_.zero_grad(); loss.backward(); opt_.step()
        with torch.no_grad():
            A = 45 + 45 * torch.sigmoid(uA); B = 90 * torch.tanh(uB)
            return float(logphi_P1(A, B)), (float(A), float(B))

    starts = [(A0, B0) for A0 in (50, 67, 83) for B0 in (-60, -30, 0, 30, 60)]
    best_o = (-1e9, None)
    for A0, B0 in starts:
        v, ab = opt_from(A0, B0)
        if v > best_o[0]:
            best_o = (v, ab)
    print(f"[grad] best logPhi(P1)={best_o[0]:.4f} at A,B=({best_o[1][0]:.1f},{best_o[1][1]:.1f})", flush=True)

    out = dict(head=a.head, P1=P1.tolist(), brain_centroid=cb.tolist(),
               grid_best_logphi=best_g[0], grid_best_AB=best_g[1],
               grad_best_logphi=best_o[0], grad_best_AB=list(best_o[1]))
    os.makedirs(M.VIZ_V3, exist_ok=True)
    fn = os.path.join(M.VIZ_V3, f"illumopt_{a.head}{('_'+a.name) if a.name else ''}.json")
    json.dump(out, open(fn, "w"), indent=2)
    print("[illumopt] wrote", fn)


if __name__ == "__main__":
    main()
