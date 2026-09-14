#!/usr/bin/env python3
"""
Build the SOURCE->POINT PATH features and align them with the existing point buffers.

Why. The INR sees FM-pyramid features sampled AT the point (local anatomy), the local optical
properties, and src_features = (r, cos theta) -- pure geometry. It sees NOTHING about the
tissue the light had to cross to reach the point. In the diffusion limit

    log10 Phi(r) ~= -(1/ln10) * INT_0^r mu_eff(s) ds  -  log10(4 pi D r)  +  const

so the dominant term of the quantity we regress is a PATH INTEGRAL the model cannot observe.
Shallow points survive this because their short path is nearly always scalp+skull, making r a
good proxy; deep points do not, because their path crosses subject-specific amounts of skull
and CSF (a low-absorption light pipe). A linear probe on TEST heads confirms it: adding these
features more than doubles the explained variance of deep log10 Phi (R^2 0.12 -> 0.28 at gate
3, 0.33 -> 0.54 at gate 6) over geometry + local optics.

Alignment. The trainer's precompute writes K=2000 points contiguously per scene, scenes sorted
by geom_key, sharded by head as heads[shard::8], and consolidate_buffers.py concatenated the 8
shards in order. That makes the global row index -> scene mapping exactly reproducible, and the
point coordinates themselves are recoverable from the stored XN. So we never re-sample; we just
recompute a new column block for rows that already exist.

Writes, next to the existing buffers:
    PATH.npy    (total, PATHDIM)      fp16, row-aligned with RAW.npy
    P_PATH.npy  (npin, 7, PATHDIM)    fp16, row-aligned with P_RAW.npy

Usage: python make_pathfeat.py <bufmm_dir> [--gpu 0]
"""
import os, sys, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
import train_inr_v11 as T

NS = 96          # ray-march samples per source->point segment
PATHDIM = 6
K = int(os.environ.get("INR_K", 2000))
NSHARD = 8


def _sample_prop(pts, prop, vs, trilinear):
    """Sample the (1,4,X,Y,Z) property volume at (n,NS,3) voxel coords -> (4,n,NS).

    trilinear=False reproduces the original nearest-voxel gather. It is exact on the grid but
    NOT differentiable w.r.t. the sample position: round().long() detaches, so a gradient taken
    w.r.t. srcpos flows only through the ds/r factor, with the tissue composition held frozen.
    Measured on sh001/C3 deep points, that leaves d(tau_eff)/d(srcpos) ANTI-PARALLEL to the true
    finite-difference gradient (cos -0.90, and -0.82 with the source moved 5 mm inward, so it is
    not a surface artefact) -- i.e. an inverse-design optimiser would be pushed the wrong way.

    trilinear=True samples with grid_sample, which is differentiable in the position: cos vs the
    finite-difference gradient becomes +0.998. The feature VALUES barely move on the three optical
    depths (tau_a/tau_sp/tau_eff agree to 1.6%, r=0.998); L_csf/L_skull shift more (15%/7%) because
    they threshold the field, and interpolation blends across tissue boundaries.
    """
    if not trilinear:
        idx = pts.round().long()
        for d in range(3):
            idx[..., d].clamp_(0, int(vs[d]) - 1)
        return prop[0][:, idx[..., 0], idx[..., 1], idx[..., 2]]
    import torch.nn.functional as F
    n, NSp = pts.shape[0], pts.shape[1]
    g = 2.0 * pts / (vs.float().view(1, 1, 3) - 1.0) - 1.0        # voxel -> [-1,1]
    grid = g.flip(-1).view(1, n, NSp, 1, 3)                       # grid_sample expects (z,y,x)
    return F.grid_sample(prop, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)[0, :, :, :, 0]


def path_features(xyz, srcpos, prop, vs, nseg=0, trilinear=False):
    """Ray-march srcpos -> xyz. Returns (n, 6 + nseg), already scaled to O(1).

    First 6 columns (scalars): tau_a, tau_sp, tau_eff (path integrals, dimensionless optical
    depths), L_csf, L_skull (mm through CSF-like / skull-like tissue), r (mm).
    tau_eff is the leading one: -tau_eff/ln10 is the dominant term of log10 Phi.

    With nseg>0 we also return the mu_eff PROFILE: the ray is split into nseg equal segments
    and each segment's mean mu_eff is a column. The scalars integrate the path and therefore
    throw away the ORDER of the tissues along it -- but light that crosses CSF and then white
    matter does not arrive like light that crosses them in the opposite order, so the ordering
    is real information. The profile keeps it. (This is why the scalars alone only bought a 9%
    deep-band RMSE gain, well short of what the linear probe suggested.)
    """
    n = xyz.shape[0]
    dev = xyz.device
    s = torch.linspace(0.0, 1.0, NS, device=dev).view(1, NS, 1)
    pts = srcpos.view(1, 1, 3) + s * (xyz.view(n, 1, 3) - srcpos.view(1, 1, 3))
    r = (xyz - srcpos.view(1, 3)).norm(dim=1)
    ds = (r / (NS - 1)).view(n, 1)

    P = _sample_prop(pts, prop, vs, trilinear)                   # (4,n,NS)
    mua, mus, g = P[0], P[1], P[2]
    musp = mus * (1.0 - g)
    mueff = torch.sqrt(torch.clamp(3.0 * mua * (mua + musp), min=1e-12))

    is_csf = ((mua < 0.005) & (musp < 0.5)).float()
    is_skull = (musp > 1.2).float()
    cols = [
        (mua * ds).sum(1),
        (musp * ds).sum(1) * 0.1,          # musp' is ~10x larger; keep columns comparable
        (mueff * ds).sum(1),
        (is_csf * ds).sum(1) * 0.1,
        (is_skull * ds).sum(1) * 0.1,
        r * 0.01,
    ]
    if nseg:
        assert NS % nseg == 0, f"NS={NS} must be divisible by nseg={nseg}"
        prof = mueff.view(n, nseg, NS // nseg).mean(2)           # (n,nseg), source -> point
        cols += list(prof.unbind(1))
    return torch.stack(cols, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bufdir")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nseg", type=int, default=0,
                    help="mu_eff profile segments (0 = the 6 scalars only)")
    ap.add_argument("--suffix", default="", help="write PATH<suffix>.npy")
    ap.add_argument("--trilinear", action="store_true",
                    help="sample the property volume with a differentiable trilinear grid_sample "
                         "instead of the nearest voxel. Required for inverse design: the nearest "
                         "gather detaches the gradient w.r.t. the source position (see _sample_prop).")
    a = ap.parse_args()
    PD = PATHDIM + a.nseg
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    meta = torch.load(os.path.join(a.bufdir, "meta.pt"), map_location="cpu")
    total = meta["meta"]["total"]; npin = meta["meta"]["npin"]
    XN = meta["XN"]                       # (total,3) normalized coords
    PXN = meta["P"]["XN"] if npin else None   # (npin,7,3) normalized stencil coords

    # ---- replay the precompute ordering exactly ----
    scenes = [s for s in T.discover_scenes() if T.label_of(s["head"]) != "test"]
    scenes.sort(key=lambda s: s["geom_key"])
    heads = sorted(set(s["head"] for s in scenes))
    order, torder = [], []
    for sh in range(NSHARD):
        mine = {h for i, h in enumerate(heads) if i % NSHARD == sh}
        sh_scenes = [s for s in scenes if s["head"] in mine]
        order += sh_scenes
        torder += [s for s in sh_scenes if T.label_of(s["head"]) == "train"]
    assert len(order) * K == total, f"replay mismatch: {len(order)}*{K} != {total}"
    PER = npin // len(torder) if torder else 0
    print(f"replayed {len(order)} scenes x K={K} = {total} rows | "
          f"{len(torder)} train scenes x PER={PER} = {npin} pinn rows", flush=True)

    out = np.lib.format.open_memmap(os.path.join(a.bufdir, f"PATH{a.suffix}.npy"), mode="w+",
                                    dtype=np.float16, shape=(total, PD))
    pout = (np.lib.format.open_memmap(os.path.join(a.bufdir, f"P_PATH{a.suffix}.npy"), mode="w+",
                                      dtype=np.float16, shape=(npin, 7, PD))
            if npin else None)
    print(f"path_dim={PD} (6 scalars + {a.nseg} profile segments) | sampling={'trilinear' if a.trilinear else 'nearest'}", flush=True)

    # Deliberately NOT SceneStore: it also loads the 688 MB fluence volume and the FM pyramid
    # for every scene, neither of which a path integral needs. prop depends only on the head
    # (205 of them for 7735 scenes), so cache it on geom_key -- the scene list is sorted by it.
    import json, scipy.io as sio
    cache = {"key": None, "prop": None, "vs": None}

    def geom(sc):
        if cache["key"] != sc["geom_key"]:
            p = sio.loadmat(sc["prop"])[T.PROP_KEY].astype(np.float32)
            p = np.transpose(p, (3, 0, 1, 2))
            cache["prop"] = torch.from_numpy(p).unsqueeze(0).to(dev)
            cache["vs"] = torch.tensor(cache["prop"].shape[-3:], device=dev)
            cache["key"] = sc["geom_key"]
        return cache["prop"], cache["vs"]

    tset = {id(s): i for i, s in enumerate(torder)}
    with torch.no_grad():
        for n, sc in enumerate(order):
            prop, vs = geom(sc)
            m = json.load(open(sc["meta"]))
            sd = {"prop": prop, "srcpos": torch.tensor(m["srcpos"], dtype=torch.float32, device=dev)}
            lo = n * K
            xn = XN[lo:lo + K].to(dev)
            xyz = (xn + 1.0) * 0.5 * (vs.float().view(1, 3) - 1.0)   # inverse of xyz_to_norm
            out[lo:lo + K] = path_features(xyz, sd["srcpos"], prop, vs, a.nseg, a.trilinear).half().cpu().numpy()

            j = tset.get(id(sc))
            if pout is not None and j is not None:
                pxn = PXN[j * PER:(j + 1) * PER].to(dev).reshape(-1, 3)
                pxyz = (pxn + 1.0) * 0.5 * (vs.float().view(1, 3) - 1.0)
                pf = path_features(pxyz, sd["srcpos"], prop, vs, a.nseg, a.trilinear)
                pout[j * PER:(j + 1) * PER] = pf.view(PER, 7, PD).half().cpu().numpy()
            if n % 1000 == 0:
                print(f"  scene {n}/{len(order)}", flush=True)
    out.flush()
    if pout is not None:
        pout.flush()
    print(f"done. PATH{a.suffix}.npy {(total*PD*2)/1e6:.0f} MB | "
          f"P_PATH{a.suffix}.npy {(npin*7*PD*2)/1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
