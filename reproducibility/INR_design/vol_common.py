#!/usr/bin/env python3
"""
Shared volume-grid utilities for the VOXEL/OPERATOR fluence baselines, so the
input-channel construction is byte-identical between the cache builder, the
trainer, and the evaluator (no silent train/eval skew).

Grid convention: each native head volume (256^3 scb / ~244x280x242 bw) is
resampled to a fixed GRID^3 cube. Normalized [-1,1] coords map 1:1 between native
and the cube (per-axis rescale), so a query at native coord xyz can be grid_sampled
from the GRID^3 prediction using coords normalized by the NATIVE shape.

Input channels (4): [mu_a/0.25, mu_s/45, source-Gaussian blob, cos(theta) beam field]
— the SAME physical information our model sees (optical props + source pos/dir),
just rasterized onto the grid.
"""
import torch
import torch.nn.functional as F

MUA_HI, MUS_HI = 0.25, 45.0     # fixed optical normalization (optical_config.PHYS_RANGE)
SRC_SIGMA = 3.0                 # source blob width (grid voxels)


def source_channels(srcpos, srcdir, vol_shape, grid, sigma=SRC_SIGMA):
    """(2, grid, grid, grid): [exp(-r^2/2sigma^2) entry blob, cos(theta) beam alignment]."""
    dev = srcpos.device
    a = torch.arange(grid, device=dev, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(a, a, a, indexing="ij")          # each (g,g,g), axis order X,Y,Z
    pos = torch.stack([gx, gy, gz], 0)                            # (3,g,g,g)
    vs = vol_shape.float().to(dev)
    sp_grid = (srcpos.to(dev) / (vs - 1).clamp_min(1)) * (grid - 1)
    u = pos - sp_grid.view(3, 1, 1, 1)                           # (3,g,g,g) grid-voxel offset
    r = u.norm(dim=0)                                            # (g,g,g)
    blob = torch.exp(-(r ** 2) / (2 * sigma ** 2))
    d = srcdir.to(dev) * (grid - 1) / (vs - 1).clamp_min(1)      # map dir into cube axes
    d = d / d.norm().clamp_min(1e-6)
    cos = (u * d.view(3, 1, 1, 1)).sum(0) / r.clamp_min(1e-6)
    return torch.stack([blob, cos], 0)                          # (2,g,g,g)


def light_code(srcpos, srcdir, vol_shape):
    """The 10-d 'light' source code = [srcpos_norm(3), srcdir(3), sin/cos of elev&azim angles(4)],
    identical to train_inr_v11.py:230-239. Pure source geometry (position + direction)."""
    dev = srcpos.device
    vs = vol_shape.float().to(dev)
    spn = srcpos.to(dev).float() / (vs - 1).clamp_min(1)
    sd = srcdir.to(dev).float()
    nrm = -sd
    A = torch.asin(nrm[2].clamp(-1, 1))
    B = torch.atan2(nrm[0], nrm[1])
    ang = torch.stack([torch.sin(A), torch.cos(A), torch.sin(B), torch.cos(B)])
    return torch.cat([spn, sd, ang])                            # (10,)


def light_channels(srcpos, srcdir, vol_shape, grid):
    """(10, grid, grid, grid): the light code broadcast to constant channels (REPRESENTATION parity —
    the grid already carries source pos/dir via source_channels; this gives grid models the identical
    explicit source vector that coord/ours receive)."""
    lc = light_code(srcpos, srcdir, vol_shape)                  # (10,)
    return lc.view(10, 1, 1, 1).expand(10, grid, grid, grid)


def build_grid_inputs(sd, grid, with_light=False):
    """(1, C, grid, grid, grid) model input from a scene dict. C=4 (V10) or 14 (V14 CW, +light broadcast)."""
    prop = sd["prop"]                                            # (1,4,X,Y,Z) physical
    ms = prop[:, :2]                                             # mu_a, mu_s
    ms_ds = F.interpolate(ms, size=(grid,) * 3, mode="trilinear", align_corners=True)
    ms_ds = torch.stack([ms_ds[:, 0] / MUA_HI, ms_ds[:, 1] / MUS_HI], 1)  # (1,2,g,g,g)
    src = source_channels(sd["srcpos"], sd["srcdir"], sd["vol_shape"], grid).unsqueeze(0)
    parts = [ms_ds, src]
    if with_light:
        parts.append(light_channels(sd["srcpos"], sd["srcdir"], sd["vol_shape"], grid).unsqueeze(0))
    return torch.cat(parts, 1)                                  # (1, 4 or 14, g,g,g)


def downsample_target(sd, grid):
    """(1, 1, grid, grid, grid) downsampled floored log10 Phi target."""
    return F.interpolate(sd["logflu"], size=(grid,) * 3, mode="trilinear", align_corners=True)


def grid_masks(sd, grid):
    """tissue (mu_s>0) and CSF masks resampled (nearest) to the cube. Returns (tissue, csf) bool (g,g,g)."""
    tis = F.interpolate(sd["prop"][:, 1:2], size=(grid,) * 3, mode="nearest")[0, 0] > 0
    csf = F.interpolate(sd["csf"].float()[None, None], size=(grid,) * 3, mode="nearest")[0, 0] > 0.5
    return tis, csf
