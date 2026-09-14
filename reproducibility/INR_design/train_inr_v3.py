#!/usr/bin/env python3
"""
v3 INR training: augmented data + source-relative features (r, cos theta), with an
ablation over fluence-magnitude WEIGHTING and a finite-difference diffusion PINN.

One process trains ONE ablation variant on ONE GPU (launch 4 in parallel for the
2x2 grid). Run:
  python train_inr_v3.py --gpu 0 --weight 0 --pinn 0   # features+aug baseline
  python train_inr_v3.py --gpu 1 --weight 1 --pinn 0   # + fluence weighting
  python train_inr_v3.py --gpu 2 --weight 0 --pinn 1   # + diffusion PINN
  python train_inr_v3.py --gpu 3 --weight 1 --pinn 1   # + both

Saves inr_checkpoints_v3/inr_v3_w{w}p{p}.pt and metrics_v3_w{w}p{p}.json.

Held-out sets (preserved): val_ang (unseen angles, seen heads), val_head (scb11).
Aug (rot+pert) only enters TRAIN.
"""
import os, sys, json, time, math, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
import inr_dataset_v3 as D
import v2_manifest as M
from train_inr import PositionalEncoding

FEATDIM = sum((48, 96, 192, 384, 768))
WEIGHT_ALPHA = 0.5
PINN_PER_SCENE = 128
PINN_H = 2.0            # finite-difference step (voxels = mm)
PINN_RMIN = 15.0        # exclude points within 15 mm of source (source region)
PINN_LAMBDA = float(os.environ.get("INR_PINN_LAMBDA", 0.05))


class INRv3(nn.Module):
    """log10(fluence) from pyramid + coord + optics + light + source-relative (r,cos)."""
    def __init__(self, pyramid_channels=(48, 96, 192, 384, 768), light_dim=10,
                 srcfeat_dim=2, num_frequencies=8, num_domains=3, domain_dim=4):
        super().__init__()
        self.pos = PositionalEncoding(num_frequencies)
        self.feat_norm = nn.LayerNorm(sum(pyramid_channels))
        self.domain_emb = nn.Embedding(num_domains, domain_dim)
        in_dim = (self.pos.out_dim + 4 + light_dim + srcfeat_dim
                  + sum(pyramid_channels) + domain_dim + 1)
        h = 512
        self.mlp = nn.Sequential(nn.Linear(in_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))

    def forward_feats(self, raw, xyz_norm, optical, light, srcfeat, domain=None):
        n = raw.shape[0]
        if light.dim() == 1:
            light = light.unsqueeze(0).expand(n, -1)
        dom = torch.zeros(n, dtype=torch.long, device=raw.device) if domain is None else domain
        emb = self.domain_emb(dom)
        sp = torch.zeros(n, 1, device=raw.device)         # log10(spacing=1mm)=0
        parts = [self.feat_norm(raw), self.pos(xyz_norm), optical, light, srcfeat, emb, sp]
        return self.mlp(torch.cat(parts, dim=-1))


def xyz_to_norm(xyz, vs):
    return torch.stack([(xyz[:, 0] / (vs[0] - 1)) * 2 - 1,
                        (xyz[:, 1] / (vs[1] - 1)) * 2 - 1,
                        (xyz[:, 2] / (vs[2] - 1)) * 2 - 1], dim=-1)


def sample_raw(model_pos_sampler, xyz_norm, pyramid):
    grid = torch.stack([xyz_norm[:, 2], xyz_norm[:, 1], xyz_norm[:, 0]], -1).view(1, 1, 1, -1, 3)
    feats = [F.grid_sample(fg, grid, mode="bilinear", align_corners=True
                           ).squeeze(0).squeeze(1).squeeze(1).permute(1, 0) for fg in pyramid]
    return torch.cat(feats, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--weight", type=int, default=0)
    ap.add_argument("--pinn", type=int, default=0)
    a = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    K = int(os.environ.get("INR_K", 4000)); EPOCHS = int(os.environ.get("INR_EPOCHS", 200))
    BATCH = int(os.environ.get("INR_BATCH", 65536)); LR = float(os.environ.get("INR_LR", 1e-3))
    BRIGHT = float(os.environ.get("INR_BRIGHT", 0.5))
    os.makedirs(M.INR_CKPT_V3, exist_ok=True)
    tag = f"w{a.weight}p{a.pinn}"

    scenes = D.discover_scenes_v3()
    labels, held = D.split_labels(scenes)
    order = sorted(range(len(scenes)), key=lambda i: scenes[i]["geom_key"])
    maxsc = int(os.environ.get("INR_MAXSCENES", 0))   # smoke-test subsample
    if maxsc:
        order = order[:maxsc]
    from collections import Counter
    print(f"[v3 {tag}] {len(scenes)} scenes {dict(Counter(labels))} {dict(Counter(s['kind'] for s in scenes))}", flush=True)

    model = INRv3().to(dev)
    store = D.SceneStoreV3(dev)
    total = len(order) * K
    RAW = torch.empty(total, FEATDIM, dtype=torch.float16, device=dev)
    XN = torch.empty(total, 3, device=dev); OPT = torch.empty(total, 4, device=dev)
    LIGHT = torch.empty(total, D.LIGHT_DIM, device=dev); SF = torch.empty(total, 2, device=dev)
    GT = torch.empty(total, 1, device=dev); WB = torch.zeros(total, 1, device=dev)
    SP = torch.zeros(total, dtype=torch.int8, device=dev)
    code = {"train": 0, "val_ang": 1, "val_head": 2}

    # PINN buffers (only if needed)
    P = None
    if a.pinn:
        npin = sum(1 for i in order if labels[i] == "train") * PINN_PER_SCENE
        P = dict(RAW=torch.empty(npin, 7, FEATDIM, dtype=torch.float16, device=dev),
                 XN=torch.empty(npin, 7, 3, device=dev), OPT=torch.empty(npin, 7, 4, device=dev),
                 SF=torch.empty(npin, 7, 2, device=dev), LIGHT=torch.empty(npin, D.LIGHT_DIM, device=dev),
                 D=torch.empty(npin, 1, device=dev), MUA=torch.empty(npin, 1, device=dev))
        pcur = 0
    off = np.array([[0, 0, 0], [PINN_H, 0, 0], [-PINN_H, 0, 0], [0, PINN_H, 0],
                    [0, -PINN_H, 0], [0, 0, PINN_H], [0, 0, -PINN_H]], dtype=np.float32)
    off_t = torch.tensor(off, device=dev)

    t0 = time.time(); cur = 0
    with torch.no_grad():
        for n, i in enumerate(order):
            sc = scenes[i]; sd = store.get(sc); vs = sd["vol_shape"]
            xyz, opt, gt = D.sample_points(sd, K, bright_frac=BRIGHT)
            xn = xyz_to_norm(xyz, vs)
            raw = sample_raw(None, xn, sd["pyramid"]).half()
            sl = slice(cur, cur + K)
            RAW[sl] = raw; XN[sl] = xn; OPT[sl] = opt; GT[sl] = gt
            LIGHT[sl] = sd["light"].unsqueeze(0).expand(K, -1)
            SF[sl] = D.src_features(xyz, sd["srcpos"], sd["srcdir"])
            WB[sl] = gt - math.log10(sd["fmax"])
            SP[sl] = code[labels[i]]
            cur += K
            # PINN interior anchors for train scenes
            if a.pinn and labels[i] == "train":
                idx = sd["valid_idx"].float()
                r = (idx - sd["srcpos"].view(1, 3)).norm(dim=1)
                cand = idx[r > PINN_RMIN]
                if cand.shape[0] < PINN_PER_SCENE:
                    cand = idx
                sel = cand[torch.randint(0, cand.shape[0], (PINN_PER_SCENE,), device=dev)]
                pts = sel.unsqueeze(1) + off_t.unsqueeze(0)            # (Kp,7,3)
                pts = pts.clamp(min=0)
                pts = torch.minimum(pts, (vs.float() - 1).view(1, 1, 3))
                flat = pts.reshape(-1, 3)
                xnf = xyz_to_norm(flat, vs)
                praw = sample_raw(None, xnf, sd["pyramid"]).half()
                popt_phys = D1opt(sd["prop"], flat, vs)               # physical (Kp*7,4)
                ps = slice(pcur, pcur + PINN_PER_SCENE)
                P["RAW"][ps] = praw.view(PINN_PER_SCENE, 7, FEATDIM)
                P["XN"][ps] = xnf.view(PINN_PER_SCENE, 7, 3)
                import optical_config as OC
                P["OPT"][ps] = OC.normalize_points(popt_phys).view(PINN_PER_SCENE, 7, 4)
                P["SF"][ps] = D.src_features(flat, sd["srcpos"], sd["srcdir"]).view(PINN_PER_SCENE, 7, 2)
                P["LIGHT"][ps] = sd["light"].unsqueeze(0).expand(PINN_PER_SCENE, -1)
                mua = popt_phys.view(PINN_PER_SCENE, 7, 4)[:, 0, 0:1]
                mus = popt_phys.view(PINN_PER_SCENE, 7, 4)[:, 0, 1:2]
                g = popt_phys.view(PINN_PER_SCENE, 7, 4)[:, 0, 2:3]
                P["MUA"][ps] = mua
                P["D"][ps] = 1.0 / (3.0 * (mua + mus * (1 - g)).clamp_min(1e-4))
                pcur += PINN_PER_SCENE
            if (n + 1) % 200 == 0:
                print(f"  precompute {n+1}/{len(scenes)} {time.time()-t0:.0f}s", flush=True)
    print(f"[v3 {tag}] precompute {time.time()-t0:.0f}s", flush=True)

    tr = torch.nonzero(SP == 0).squeeze(1); va = torch.nonzero(SP == 1).squeeze(1)
    vh = torch.nonzero(SP == 2).squeeze(1)
    opt_ = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=EPOCHS, eta_min=1e-5)

    def ev(idx):
        model.eval(); tot, nb = 0.0, 0
        with torch.no_grad():
            for j in range(0, idx.numel(), BATCH):
                b = idx[j:j + BATCH]
                p = model.forward_feats(RAW[b].float(), XN[b], OPT[b], LIGHT[b], SF[b])
                tot += F.mse_loss(p, GT[b]).item(); nb += 1
        return tot / max(nb, 1)

    def pinn_loss(bsz=4096):
        if not a.pinn:
            return torch.tensor(0.0, device=dev)
        m = torch.randint(0, P["RAW"].shape[0], (bsz,), device=dev)
        praw = P["RAW"][m].float().reshape(-1, FEATDIM)
        pxn = P["XN"][m].reshape(-1, 3); popt = P["OPT"][m].reshape(-1, 4)
        psf = P["SF"][m].reshape(-1, 2)
        plight = P["LIGHT"][m].unsqueeze(1).expand(-1, 7, -1).reshape(-1, D.LIGHT_DIM)
        logphi = model.forward_feats(praw, pxn, popt, plight, psf).view(bsz, 7)
        phi = torch.pow(10.0, logphi)                              # (bsz,7)
        lap = (phi[:, 1:].sum(1) - 6 * phi[:, 0]) / (PINN_H ** 2)  # neighbors - 6*center
        Dc = P["D"][m].squeeze(1); mua = P["MUA"][m].squeeze(1); phic = phi[:, 0]
        res = (-Dc * lap + mua * phic) / (mua * phic + 1e-9)       # relative residual
        return (res ** 2).mean()

    hist = {"train": [], "val_ang": [], "val_head": []}; best = float("inf"); t1 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = tr[torch.randperm(tr.numel(), device=dev)]
        tl, nb = 0.0, 0
        for j in range(0, perm.numel(), BATCH):
            b = perm[j:j + BATCH]
            pred = model.forward_feats(RAW[b].float(), XN[b], OPT[b], LIGHT[b], SF[b])
            if a.weight:
                w = torch.pow(10.0, WEIGHT_ALPHA * WB[b])           # (Phi/Phimax)^alpha
                data = (w * (pred - GT[b]) ** 2).sum() / w.sum().clamp_min(1e-6)
            else:
                data = F.mse_loss(pred, GT[b])
            loss = data + (PINN_LAMBDA * pinn_loss() if a.pinn else 0.0)
            opt_.zero_grad(); loss.backward(); opt_.step()
            tl += data.item(); nb += 1
        sched.step()
        va_l = ev(va); vh_l = ev(vh)
        hist["train"].append(tl / nb); hist["val_ang"].append(va_l); hist["val_head"].append(vh_l)
        if va_l < best:
            best = va_l
            torch.save({"model": model.state_dict(), "epoch": ep, "val_ang": va_l, "val_head": vh_l,
                        "weight": a.weight, "pinn": a.pinn, "light_dim": D.LIGHT_DIM,
                        "srcfeat": True, "history": hist}, os.path.join(M.INR_CKPT_V3, f"inr_v3_{tag}.pt"))
        if ep % 20 == 0 or ep == 1:
            print(f"  [{tag}] ep {ep}/{EPOCHS} tr {tl/nb:.4f} va {va_l:.4f} vh {vh_l:.4f} best {best:.4f} {time.time()-t1:.0f}s", flush=True)
    json.dump({"weight": a.weight, "pinn": a.pinn, "best_val_ang": best,
               "final_val_head": hist["val_head"][-1], "history": hist},
              open(os.path.join(M.INR_CKPT_V3, f"metrics_v3_{tag}.json"), "w"))
    print(f"[v3 {tag}] DONE best_val_ang={best:.4f} final_val_head={hist['val_head'][-1]:.4f} total {time.time()-t0:.0f}s", flush=True)


# physical optical sampling helper (grid_sample on physical prop volume)
def D1opt(prop, xyz, vs):
    import inr_dataset as D1
    return D1.sample_volume(prop, xyz, vs)


if __name__ == "__main__":
    main()
