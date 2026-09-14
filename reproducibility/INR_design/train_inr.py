#!/usr/bin/env python3
"""
FM-informed INR surrogate — steady-state (CW) fluence training.

Pipeline (matches the FM-INR research diagram):
  Foundation model (FM_tune_t4) → feature pyramid  ─┐
  coordinate (x,y,z)  ──────────────────────────────┼─► INR (MLP) ─► log10 fluence
  optical params (mu_a,mu_s,g,n) at point ──────────┤
  illumination (srcpos, srcdir) ────────────────────┘
  Loss = data (log-fluence MSE)  +  λ · PINN (steady-state diffusion residual)

Design:
  - Steady-state only (no time).
  - Wavelength is NOT an input (encoded in optical params) → all wl scenes pooled.
  - One scene (sample,wl) processed at a time; several point-batches per scene.
"""
import os
import math
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import inr_dataset as D


# ----------------------------------------------------------------------------
# 1. Model
# ----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=8):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.out_dim = 3 + 3 * 2 * num_frequencies

    def forward(self, x):                       # x: (N,3) in [-1,1]
        freqs = 2 ** torch.arange(self.num_frequencies, device=x.device, dtype=x.dtype)
        enc = [x]
        for fr in freqs:
            enc.append(torch.sin(math.pi * fr * x))
            enc.append(torch.cos(math.pi * fr * x))
        return torch.cat(enc, dim=-1)


class OpticalFluenceINR(nn.Module):
    """Predicts log10(fluence) at a point from pyramid features + coord + optics + light."""
    def __init__(self, pyramid_channels=(48, 96, 192, 384, 768),
                 light_dim=6, num_frequencies=8,
                 use_cond=True, num_domains=3, domain_dim=4):
        super().__init__()
        self.pos_encoder = PositionalEncoding(num_frequencies)
        self.feat_norm = nn.LayerNorm(sum(pyramid_channels))

        # domain-aware conditioning: a small learned per-domain embedding + the
        # log physical voxel spacing (the real cross-species scale variable).
        self.use_cond = use_cond
        cond_extra = 0
        if use_cond:
            self.domain_emb = nn.Embedding(num_domains, domain_dim)
            cond_extra = domain_dim + 1               # + log10(spacing_mm)

        in_dim = (self.pos_encoder.out_dim + 4 + light_dim
                  + sum(pyramid_channels) + cond_extra)
        h = 512
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h), nn.GELU(),
            nn.Linear(h, h),       nn.GELU(),
            nn.Linear(h, h // 2),  nn.GELU(),
            nn.Linear(h // 2, 1),                 # raw output = log10 fluence
        )

    def sample_pyramid_raw(self, xyz_norm, pyramid):
        """xyz_norm: (N,3) in [-1,1] (x,y,z order). Returns RAW (pre-LayerNorm) (N,sumC)."""
        N = xyz_norm.shape[0]
        grid = torch.stack([xyz_norm[:, 2], xyz_norm[:, 1], xyz_norm[:, 0]], dim=-1)
        grid = grid.view(1, 1, 1, N, 3)
        feats = []
        for fg in pyramid:
            s = F.grid_sample(fg, grid, mode='bilinear', align_corners=True)
            s = s.squeeze(0).squeeze(1).squeeze(1).permute(1, 0)   # (N,C)
            feats.append(s)
        return torch.cat(feats, dim=-1)

    def sample_pyramid(self, xyz_norm, pyramid, detach=False):
        out = self.sample_pyramid_raw(xyz_norm, pyramid)
        if detach:
            out = out.detach()
        return self.feat_norm(out)

    @staticmethod
    def xyz_to_norm(xyz_phys, vol_shape):
        D_, H_, W_ = vol_shape
        return torch.stack([
            (xyz_phys[:, 0] / (D_ - 1)) * 2 - 1,
            (xyz_phys[:, 1] / (H_ - 1)) * 2 - 1,
            (xyz_phys[:, 2] / (W_ - 1)) * 2 - 1,
        ], dim=-1)

    def _cond(self, n, device, domain=None, spacing=None):
        """Per-point conditioning [domain_embedding, log10(voxel_spacing_mm)]."""
        if not self.use_cond:
            return None
        if domain is None:
            domain = torch.zeros(n, dtype=torch.long, device=device)
        else:
            domain = torch.as_tensor(domain, device=device, dtype=torch.long)
            if domain.dim() == 0:
                domain = domain.view(1).expand(n)
        emb = self.domain_emb(domain)                          # (n, domain_dim)
        if spacing is None:
            sp = torch.zeros(n, 1, device=device)              # log10(1.0)=0
        else:
            spacing = torch.as_tensor(spacing, device=device, dtype=torch.float32)
            sp = torch.log10(spacing.reshape(-1, 1).expand(n, 1).clamp_min(1e-6))
        return torch.cat([emb, sp], dim=-1)

    def forward_features(self, raw_feat, xyz_norm, optical, light, domain=None, spacing=None):
        """Head on PRECOMPUTED raw pyramid features (for fast decoupled training)."""
        n = raw_feat.shape[0]
        pyramid_feat = self.feat_norm(raw_feat)
        pos_feat = self.pos_encoder(xyz_norm)
        if light.dim() == 1:
            light = light.unsqueeze(0).expand(n, -1)
        parts = [pyramid_feat, pos_feat, optical, light]
        cond = self._cond(n, raw_feat.device, domain, spacing)
        if cond is not None:
            parts.append(cond)
        return self.mlp(torch.cat(parts, dim=-1))

    def forward(self, xyz_phys, vol_shape, optical, light, pyramid,
                detach_features=False, domain=None, spacing=None):
        """
        xyz_phys : (N,3) voxel coords (may require grad for PINN)
        vol_shape: (3,) tensor [X,Y,Z]
        optical  : (N,4) normalized optical params
        light    : (6,) or (N,6) illumination vector
        pyramid  : list of 5 (1,C,d,h,w)
        domain/spacing : optional per-scene category id / voxel spacing (mm)
        Returns  : (N,1) predicted log10 fluence
        """
        n = xyz_phys.shape[0]
        xyz_norm = self.xyz_to_norm(xyz_phys, vol_shape)
        pyramid_feat = self.sample_pyramid(xyz_norm, pyramid, detach=detach_features)
        pos_feat = self.pos_encoder(xyz_norm)
        if light.dim() == 1:
            light = light.unsqueeze(0).expand(n, -1)
        parts = [pyramid_feat, pos_feat, optical, light]
        cond = self._cond(n, xyz_phys.device, domain, spacing)
        if cond is not None:
            parts.append(cond)
        return self.mlp(torch.cat(parts, dim=-1))


# ----------------------------------------------------------------------------
# 2. PINN — steady-state diffusion residual
#    -D ∇²Φ + μa Φ = 0   (source-free interior; constant-D local approximation)
#    D = 1 / (3 (μa + μs(1-g)))
# ----------------------------------------------------------------------------
def diffusion_pinn_residual(model, scene_data, n_points, src_exclude_vox=15.0):
    sd = scene_data
    idx = sd['valid_idx']
    M = idx.shape[0]
    sel = torch.randint(0, M, (n_points,), device=idx.device)
    xyz = idx[sel].float() + (torch.rand(n_points, 3, device=idx.device) - 0.5)
    xyz = xyz.clamp(min=0)
    xyz = torch.minimum(xyz, sd['vol_shape'].float() - 1)

    # exclude points near the source (where S != 0)
    dist = (xyz - sd['srcpos']).norm(dim=-1)
    keep = dist > src_exclude_vox
    if keep.sum() < 8:
        return torch.zeros((), device=idx.device)
    xyz = xyz[keep].requires_grad_(True)

    optical = D.sample_volume(sd['prop'], xyz, sd['vol_shape'])            # physical units
    mu_a = optical[:, 0:1]
    mu_s = optical[:, 1:2]
    g    = optical[:, 2:3]
    Dcoef = 1.0 / (3.0 * (mu_a + mu_s * (1.0 - g)) + 1e-6)

    optical_n = D.OC.normalize_points(optical)            # fixed-range (single definition)
    y = model(xyz, sd['vol_shape'], optical_n, sd['light'], sd['pyramid'],
              detach_features=True)                                        # (N,1) log10 Φ

    a = math.log(10.0)
    grad_y = torch.autograd.grad(y.sum(), xyz, create_graph=True)[0]       # (N,3)
    lap_y = 0.0
    for i in range(3):
        gi = torch.autograd.grad(grad_y[:, i].sum(), xyz, create_graph=True)[0][:, i]
        lap_y = lap_y + gi
    lap_y = lap_y.unsqueeze(-1)                                            # (N,1)

    Phi = torch.pow(10.0, y)
    lap_Phi = a * Phi * (a * (grad_y ** 2).sum(-1, keepdim=True) + lap_y)
    residual = -Dcoef * lap_Phi + mu_a * Phi
    # normalize residual scale by Φ to keep it dimensionless & stable
    res_norm = residual / (Phi + 1e-12)
    return (res_norm ** 2).mean()


# ----------------------------------------------------------------------------
# 3. Training
# ----------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from repro_config import INR_CHECKPOINT_DIR
    OUT_DIR = INR_CHECKPOINT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- hyperparams (env-overridable for the full run) ---
    EPOCHS          = int(os.environ.get("INR_EPOCHS", 15))
    STEPS_PER_SCENE = int(os.environ.get("INR_STEPS", 50))   # steps per (geom,illum) scene
    N_DATA          = int(os.environ.get("INR_NDATA", 4096))
    N_PINN          = 1024
    LR              = float(os.environ.get("INR_LR", 2e-4))
    PINN_WEIGHT     = float(os.environ.get("INR_PINN_W", 1e-3))
    PINN_WARMUP_EP  = max(1, EPOCHS // 3)
    HOLDOUT_GEOMS   = int(os.environ.get("INR_HOLDOUT", 1))  # whole geometries held out

    scenes = D.discover_scenes()
    assert len(scenes) > 0, "No scenes found — run extract_pyramids.py first."
    groups = D.group_by_geometry(scenes)
    # reproducible geometry-level split (don't leak a geometry's illuminations into val)
    geom_keys = sorted(groups.keys())
    random.Random(0).shuffle(geom_keys)
    val_keys = set(geom_keys[:HOLDOUT_GEOMS]) if len(geom_keys) > HOLDOUT_GEOMS else set()
    train_scenes = [s for s in scenes if s['geom_key'] not in val_keys]
    val_scenes   = [s for s in scenes if s['geom_key'] in val_keys]
    n_illum = [len(v) for v in groups.values()]
    print(f"Discovered {len(scenes)} scenes / {len(groups)} geometries "
          f"(illum/geom min={min(n_illum)} max={max(n_illum)})")
    print(f"Train: {len(train_scenes)} scenes  Val: {len(val_scenes)} scenes "
          f"({HOLDOUT_GEOMS} held-out geometries)\n")

    store = D.SceneStore(device)
    model = OpticalFluenceINR().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    history = {'data': [], 'pinn': [], 'val': []}
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # geometry-major order: all illuminations of one geometry consecutively, so
        # the 2-level SceneStore cache reuses each ~2.5 GB pyramid across its sources
        train_groups = D.group_by_geometry(train_scenes)
        gkeys = list(train_groups.keys()); random.shuffle(gkeys)
        ep_data, ep_pinn, n_steps = 0.0, 0.0, 0
        pinn_w = PINN_WEIGHT if epoch > PINN_WARMUP_EP else 0.0

        for gk in gkeys:
            gscenes = train_groups[gk][:]; random.shuffle(gscenes)
            for scene in gscenes:
                sd = store.get(scene)
                for _ in range(STEPS_PER_SCENE):
                    opt.zero_grad()
                    xyz, optical, gt_log = D.sample_points(sd, N_DATA)
                    pred = model(xyz, sd['vol_shape'], optical, sd['light'], sd['pyramid'])
                    data_loss = F.mse_loss(pred, gt_log)

                    if pinn_w > 0:
                        pinn_loss = diffusion_pinn_residual(model, sd, N_PINN)
                    else:
                        pinn_loss = torch.zeros((), device=device)

                    loss = data_loss + pinn_w * pinn_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                    ep_data += data_loss.item()
                    ep_pinn += float(pinn_loss.item())
                    n_steps += 1

        sched.step()
        ep_data /= n_steps
        ep_pinn /= n_steps
        history['data'].append(ep_data)
        history['pinn'].append(ep_pinn)

        # --- validation (data loss only) ---
        val_loss = float('nan')
        if val_scenes:
            model.eval()
            sd = store.get(val_scenes[0])
            with torch.no_grad():
                xyz, optical, gt_log = D.sample_points(sd, N_DATA * 2, jitter=False)
                pred = model(xyz, sd['vol_shape'], optical, sd['light'], sd['pyramid'])
                val_loss = F.mse_loss(pred, gt_log).item()
            history['val'].append(val_loss)

        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS} | data(logMSE): {ep_data:.4f} | "
              f"pinn: {ep_pinn:.3e} | val: {val_loss:.4f} | {dt:.0f}s")

        if epoch % 10 == 0 or epoch == EPOCHS:
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'history': history},
                       os.path.join(OUT_DIR, 'inr_model.pt'))
            with open(os.path.join(OUT_DIR, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

    print(f"\n[DONE] saved to {OUT_DIR}")
    print(f"  final data logMSE: {history['data'][-1]:.4f}  "
          f"val: {history['val'][-1] if history['val'] else float('nan'):.4f}")


if __name__ == "__main__":
    main()
