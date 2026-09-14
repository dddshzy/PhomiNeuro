#!/usr/bin/env python3
"""
0620V6 ablation runner — clean-architecture (C) + FM-encoder ablation (A).

Keeps the v5-b0n1 production recipe FIXED (deep diffusion-PINN + shallow data-weight
+ near-source sampling, NO analytic baseline) and varies only:

  --clean   {0,1} : 1 = drop the legacy dead inputs (domain embedding + log-spacing)
                    that are constant in the single-domain / isotropic-1mm setting.
                    0 = keep them (== v5-b0n1 architecture, for the C control).
  --pyramid {0,1} : 1 = use the FM feature pyramid (1488-d).  0 = NO-FM ablation
                    (coord + optics + light + srcfeat only -> MLP).
  --pyrsrc  {prod,bare,rand} : which pyramid cache to read (only if --pyramid 1):
                    prod = extracted_pyramids_v2/_v3 (fine-tuned FM, production)
                    bare = un-finetuned VISTA features (no LoRA/MAE/STRD)
                    rand = randomly-initialized encoder features
                    (bare/rand require extract_pyramids_v6.py to have run first)

Run matrix (this round, no re-extraction needed):
  python train_inr_0620v6.py --gpu 0 --clean 0 --pyramid 1        # dirty  (C control)
  python train_inr_0620v6.py --gpu 1 --clean 1 --pyramid 1        # full   (clean baseline)
  python train_inr_0620v6.py --gpu 2 --clean 1 --pyramid 0        # nofm   (A headline)

Saves inr_0620V6_{name}.pt + metrics_0620V6_{name}.json into INR_CKPT_V3.
"""
import os, sys, json, time, math, argparse
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
import v2_manifest as M
import inr_dataset as D1
import optical_config as OC
from train_inr import PositionalEncoding
from train_inr_v3 import xyz_to_norm, sample_raw

FEATDIM = sum((48, 96, 192, 384, 768)); LN10 = math.log(10.0)
DEEP_LO, DEEP_HI = 3.0, 6.5; SHALLOW_BOOST = 3.0
PINN_PER_SCENE = 160; PINN_H = 2.0
PINN_LAMBDA = float(os.environ.get("INR_PINN_LAMBDA", 0.05))
R_NEAR = 55.0; NEAR_FRAC = 0.35

# alternate pyramid caches for the bare/random encoder ablations
PYRAMID_V6_BARE = os.path.join(M.ROOT, "INR_design", "extracted_pyramids_v6_bare")
PYRAMID_V6_RAND = os.path.join(M.ROOT, "INR_design", "extracted_pyramids_v6_rand")
PYRAMID_V8 = os.path.join(M.ROOT, "INR_design", "extracted_pyramids_v8")  # re-finetuned FM encoder
PYRAMID_V9 = os.path.join(M.ROOT, "INR_design", "extracted_pyramids_v9")  # v9 encoder (clean 3-way split)
PYRSRC_DIR = {"bare": PYRAMID_V6_BARE, "rand": PYRAMID_V6_RAND, "v8": PYRAMID_V8, "v9": PYRAMID_V9}


class INRv6(nn.Module):
    """log10(fluence); pyramid optional (NO-FM ablation), legacy dead inputs optional."""
    def __init__(self, pyramid_channels=(48, 96, 192, 384, 768), light_dim=10,
                 srcfeat_dim=2, num_frequencies=8, num_domains=3, domain_dim=4,
                 clean=True, use_pyramid=True, mlp_width=512, mlp_extra=0):
        super().__init__()
        self.clean = clean; self.use_pyramid = use_pyramid
        self.pos = PositionalEncoding(num_frequencies)
        if use_pyramid:
            self.feat_norm = nn.LayerNorm(sum(pyramid_channels))
        if not clean:
            self.domain_emb = nn.Embedding(num_domains, domain_dim)
        in_dim = self.pos.out_dim + 4 + light_dim + srcfeat_dim
        if use_pyramid:
            in_dim += sum(pyramid_channels)
        if not clean:
            in_dim += domain_dim + 1
        h = mlp_width                                  # capacity sweep: width + extra depth
        layers = [nn.Linear(in_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU()]
        for _ in range(mlp_extra):                     # mlp_extra=0 -> original hero MLP
            layers += [nn.Linear(h, h), nn.GELU()]
        layers += [nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward_feats(self, raw, xyz_norm, optical, light, srcfeat, domain=None):
        n = xyz_norm.shape[0]
        if light.dim() == 1:
            light = light.unsqueeze(0).expand(n, -1)
        parts = []
        if self.use_pyramid:
            parts.append(self.feat_norm(raw))
        parts += [self.pos(xyz_norm), optical, light, srcfeat]
        if not self.clean:
            dom = torch.zeros(n, dtype=torch.long, device=xyz_norm.device) if domain is None else domain
            parts += [self.domain_emb(dom), torch.zeros(n, 1, device=xyz_norm.device)]
        return self.mlp(torch.cat(parts, dim=-1))


def sample_near(sd, n):
    idx = sd["valid_idx"].float()
    r = (idx - sd["srcpos"].view(1, 3)).norm(dim=1)
    near = idx[r < R_NEAR]
    if near.shape[0] < 8:
        near = idx
    sel = near[torch.randint(0, near.shape[0], (n,), device=idx.device)]
    xyz = sel + (torch.rand_like(sel) - 0.5)
    xyz = torch.minimum(xyz.clamp(min=0), (sd["vol_shape"].float() - 1).view(1, 3))
    opt = OC.normalize_points(D1.sample_volume(sd["prop"], xyz, sd["vol_shape"]))
    gt = D1.sample_volume(sd["logflu"], xyz, sd["vol_shape"])
    return xyz, opt, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--clean", type=int, default=1)
    ap.add_argument("--pyramid", type=int, default=1)
    ap.add_argument("--pyrsrc", choices=["prod", "bare", "rand", "v8", "v9"], default="prod")
    ap.add_argument("--name", default=None)
    ap.add_argument("--tag", default="0620V6", help="checkpoint tag prefix (e.g. v7 for scb+bw)")
    a = ap.parse_args()

    # redirect pyramid caches for the bare/random encoder ablations (before discover)
    if a.pyramid and a.pyrsrc != "prod":
        base = PYRSRC_DIR[a.pyrsrc]
        M.PYRAMID_V2 = os.path.join(base, "v2"); M.PYRAMID_V3 = os.path.join(base, "v3")
    import inr_dataset_v3 as D                              # imports AFTER redirect

    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    K = int(os.environ.get("INR_K", 4000)); EPOCHS = int(os.environ.get("INR_EPOCHS", 200))
    BATCH = int(os.environ.get("INR_BATCH", 65536)); LR = float(os.environ.get("INR_LR", 1e-3))
    BRIGHT = float(os.environ.get("INR_BRIGHT", 0.5))
    os.makedirs(M.INR_CKPT_V3, exist_ok=True)
    name = a.name or (("full" if a.clean else "dirty") if a.pyramid and a.pyrsrc == "prod"
                      else ("nofm" if not a.pyramid else a.pyrsrc))
    tag = f"{a.tag}_{name}"; use_pyr = bool(a.pyramid)

    scenes = D.discover_scenes_v3(); labels, _ = D.split_labels(scenes)
    order = sorted(range(len(scenes)), key=lambda i: scenes[i]["geom_key"])
    order = [i for i in order if labels[i] != "test_head"]   # LOCKED test never loaded
    maxsc = int(os.environ.get("INR_MAXSCENES", 0))
    if maxsc:
        order = order[:maxsc]
    from collections import Counter
    print(f"[{tag}] clean={a.clean} pyramid={a.pyramid} src={a.pyrsrc} | "
          f"{len(order)} scenes | {dict(Counter(labels[i] for i in order))}", flush=True)

    model = INRv6(clean=bool(a.clean), use_pyramid=use_pyr).to(dev)
    store = D.SceneStoreV3(dev); total = len(order) * K
    RAW = torch.empty(total, FEATDIM, dtype=torch.float16, device=dev) if use_pyr else None
    XN = torch.empty(total, 3, device=dev); OPT = torch.empty(total, 4, device=dev)
    LIGHT = torch.empty(total, D.LIGHT_DIM, device=dev); SF = torch.empty(total, 2, device=dev)
    GT = torch.empty(total, 1, device=dev); DEC = torch.zeros(total, 1, device=dev)
    SP = torch.zeros(total, dtype=torch.int8, device=dev)
    code = {"train": 0, "val_ang": 1, "val_head": 2}
    npin = sum(1 for i in order if labels[i] == "train") * PINN_PER_SCENE
    P = dict(XN=torch.empty(npin, 7, 3, device=dev), OPT=torch.empty(npin, 7, 4, device=dev),
             SF=torch.empty(npin, 7, 2, device=dev), LIGHT=torch.empty(npin, D.LIGHT_DIM, device=dev),
             D=torch.empty(npin, 1, device=dev), MUA=torch.empty(npin, 1, device=dev), n=0)
    P["RAW"] = torch.empty(npin, 7, FEATDIM, dtype=torch.float16, device=dev) if use_pyr else None
    off = torch.tensor([[0,0,0],[PINN_H,0,0],[-PINN_H,0,0],[0,PINN_H,0],[0,-PINN_H,0],[0,0,PINN_H],[0,0,-PINN_H]],
                       dtype=torch.float32, device=dev)

    t0 = time.time(); cur = 0
    with torch.no_grad():
        for n, i in enumerate(order):
            sc = scenes[i]; sd = store.get(sc); vs = sd["vol_shape"]; lm = math.log10(sd["fmax"])
            xyz, opt, gt = D.sample_points(sd, K, bright_frac=BRIGHT)
            if labels[i] == "train":                       # near-source enrichment (nearsamp=1)
                nn_ = int(NEAR_FRAC * K); keep = K - nn_
                xn2, on2, gn2 = sample_near(sd, nn_)
                xyz = torch.cat([xyz[:keep], xn2]); opt = torch.cat([opt[:keep], on2]); gt = torch.cat([gt[:keep], gn2])
            xnorm = xyz_to_norm(xyz, vs)
            if use_pyr:
                RAW[cur:cur+K] = sample_raw(None, xnorm, sd["pyramid"]).half()
            XN[cur:cur+K] = xnorm; OPT[cur:cur+K] = opt; GT[cur:cur+K] = gt
            LIGHT[cur:cur+K] = sd["light"].unsqueeze(0).expand(K, -1)
            SF[cur:cur+K] = D.src_features(xyz, sd["srcpos"], sd["srcdir"])
            DEC[cur:cur+K] = lm - gt; SP[cur:cur+K] = code[labels[i]]; cur += K
            if labels[i] == "train":
                vi = sd["valid_idx"].float(); lf = D1.sample_volume(sd["logflu"], vi, vs).squeeze(1)
                dec = lm - lf; deep = vi[(dec >= DEEP_LO) & (dec <= DEEP_HI)]
                if deep.shape[0] < PINN_PER_SCENE:
                    deep = vi if deep.shape[0] == 0 else deep
                sel = deep[torch.randint(0, deep.shape[0], (PINN_PER_SCENE,), device=dev)]
                pts = torch.minimum((sel.unsqueeze(1) + off.unsqueeze(0)).clamp(min=0),
                                    (vs.float()-1).view(1,1,3)).reshape(-1, 3)
                xnf = xyz_to_norm(pts, vs); phys = D1.sample_volume(sd["prop"], pts, vs)
                ps = slice(P["n"], P["n"]+PINN_PER_SCENE)
                if use_pyr:
                    P["RAW"][ps] = sample_raw(None, xnf, sd["pyramid"]).half().view(PINN_PER_SCENE,7,FEATDIM)
                P["XN"][ps] = xnf.view(PINN_PER_SCENE,7,3); P["OPT"][ps] = OC.normalize_points(phys).view(PINN_PER_SCENE,7,4)
                P["SF"][ps] = D.src_features(pts, sd["srcpos"], sd["srcdir"]).view(PINN_PER_SCENE,7,2)
                P["LIGHT"][ps] = sd["light"].unsqueeze(0).expand(PINN_PER_SCENE,-1)
                ph = phys.view(PINN_PER_SCENE,7,4)[:,0]; mua,mus,g = ph[:,0:1],ph[:,1:2],ph[:,2:3]
                P["MUA"][ps] = mua; P["D"][ps] = 1.0/(3.0*(mua+mus*(1-g)).clamp_min(1e-4))
                P["n"] += PINN_PER_SCENE
            if (n+1) % 300 == 0:
                print(f"  precompute {n+1}/{len(order)} {time.time()-t0:.0f}s", flush=True)
    print(f"[{tag}] precompute {time.time()-t0:.0f}s pinn={P['n']}", flush=True)

    def rawb(idx):
        return RAW[idx].float() if use_pyr else None

    tr = torch.nonzero(SP==0).squeeze(1); va = torch.nonzero(SP==1).squeeze(1); vh = torch.nonzero(SP==2).squeeze(1)
    opt_ = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=EPOCHS, eta_min=1e-5)

    def ev(idx):
        model.eval(); tot, nb = 0.0, 0
        with torch.no_grad():
            for j in range(0, idx.numel(), BATCH):
                b = idx[j:j+BATCH]
                p = model.forward_feats(rawb(b), XN[b], OPT[b], LIGHT[b], SF[b])
                tot += F.mse_loss(p, GT[b]).item(); nb += 1
        return tot/max(nb,1)

    def pinn_loss(bsz=4096):
        if P["n"] == 0:
            return torch.tensor(0.0, device=dev)
        m = torch.randint(0, P["n"], (bsz,), device=dev)
        praw = P["RAW"][m].float().reshape(-1, FEATDIM) if use_pyr else None
        pxn = P["XN"][m].reshape(-1,3); popt = P["OPT"][m].reshape(-1,4); psf = P["SF"][m].reshape(-1,2)
        pl = P["LIGHT"][m].unsqueeze(1).expand(-1,7,-1).reshape(-1, D.LIGHT_DIM)
        logphi = model.forward_feats(praw, pxn, popt, pl, psf).view(bsz, 7)
        phi = torch.pow(10.0, logphi)
        lap = (phi[:,1:].sum(1) - 6*phi[:,0]) / (PINN_H**2)
        Dc = P["D"][m].squeeze(1); mua = P["MUA"][m].squeeze(1); phic = phi[:,0]
        num = -Dc*lap + mua*phic; den = (Dc*lap).abs() + mua*phic + 1e-12
        return ((num/den)**2).mean()

    hist={"train":[],"val_ang":[],"val_head":[]}; best=float("inf"); t1=time.time()
    for ep in range(1, EPOCHS+1):
        model.train(); perm = tr[torch.randperm(tr.numel(), device=dev)]; tl, nb = 0.0, 0
        for j in range(0, perm.numel(), BATCH):
            b = perm[j:j+BATCH]
            pred = model.forward_feats(rawb(b), XN[b], OPT[b], LIGHT[b], SF[b])
            w = 1.0 + SHALLOW_BOOST*((DEEP_LO - DEC[b])/DEEP_LO).clamp(0,1)
            data = (w * (pred - GT[b])**2).sum() / w.sum().clamp_min(1e-6)
            loss = data + PINN_LAMBDA * pinn_loss()
            opt_.zero_grad(); loss.backward(); opt_.step(); tl += data.item(); nb += 1
        sched.step(); va_l = ev(va); vh_l = ev(vh)
        hist["train"].append(tl/nb); hist["val_ang"].append(va_l); hist["val_head"].append(vh_l)
        if vh_l < best:                              # SELECT on VAL subjects (cross-subject)
            best = vh_l
            torch.save({"model": model.state_dict(), "epoch": ep, "val_ang": va_l, "val_head": vh_l,
                        "clean": a.clean, "pyramid": a.pyramid, "pyrsrc": a.pyrsrc,
                        "light_dim": D.LIGHT_DIM, "srcfeat": True, "history": hist},
                       os.path.join(M.INR_CKPT_V3, f"inr_{tag}.pt"))
        if ep % 20 == 0 or ep == 1:
            print(f"  [{tag}] ep {ep}/{EPOCHS} tr {tl/nb:.4f} va {va_l:.4f} vh {vh_l:.4f} best {best:.4f} {time.time()-t1:.0f}s", flush=True)
    json.dump({"clean": a.clean, "pyramid": a.pyramid, "pyrsrc": a.pyrsrc, "name": name,
               "best_val_head": best, "selected_on": "val_head",
               "final_val_ang": hist["val_ang"][-1], "history": hist},
              open(os.path.join(M.INR_CKPT_V3, f"metrics_{tag}.json"), "w"))
    print(f"[{tag}] DONE best_val_head={best:.4f} (selected on VAL) {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
