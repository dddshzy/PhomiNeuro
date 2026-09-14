"""Unit tests for the fixed-range optical normalization (single source of truth)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optical_config as OC


def _random_phys_volume(shape=(4, 6, 7, 8), channels=OC.CHANNELS, seed=0):
    """Random physical volume with each channel drawn within its PHYS_RANGE."""
    rng = np.random.default_rng(seed)
    lo, hi = OC.CONFIG.lo_hi(channels)
    vol = np.empty(shape, dtype=np.float32)
    for c in range(len(channels)):
        vol[c] = rng.uniform(lo[c], hi[c], size=shape[1:])
    return vol


def test_metadata_sane():
    assert OC.CHANNELS == ("mu_a", "mu_s", "g", "n")
    assert OC.MU_S_IS_REDUCED is False            # mu_s is RAW scattering
    for c in OC.CHANNELS:
        lo, hi = OC.PHYS_RANGE[c]
        assert hi > lo, f"degenerate range for {c}"
    # bounds must cover the empirical dataset span (with headroom)
    assert OC.PHYS_RANGE["mu_a"][1] >= 0.110
    assert OC.PHYS_RANGE["mu_s"][1] >= 40.10
    assert OC.PHYS_RANGE["g"][0] <= 0.85
    assert OC.PHYS_RANGE["n"][1] >= 1.371


def test_volume_roundtrip_numpy():
    v = _random_phys_volume(seed=1)
    out = OC.denormalize(OC.normalize(v, channel_dim=0), channel_dim=0)
    assert np.allclose(out, v, atol=1e-5), np.abs(out - v).max()


def test_volume_roundtrip_torch():
    v = torch.from_numpy(_random_phys_volume(seed=2))
    out = OC.denormalize(OC.normalize(v, channel_dim=0), channel_dim=0)
    assert torch.allclose(out, v, atol=1e-5), (out - v).abs().max().item()


def test_normalize_known_values():
    # channel-by-channel min/max should land on 0 and 1
    lo, hi = OC.CONFIG.lo_hi(OC.CHANNELS)
    vlo = torch.tensor(lo).view(4, 1, 1, 1) * torch.ones(4, 2, 2, 2)
    vhi = torch.tensor(hi).view(4, 1, 1, 1) * torch.ones(4, 2, 2, 2)
    assert torch.allclose(OC.normalize(vlo), torch.zeros_like(vlo), atol=1e-6)
    assert torch.allclose(OC.normalize(vhi), torch.ones_like(vhi), atol=1e-6)


def test_normalize_points_matches_volume():
    # per-point (N,C) normalization must equal the volume path channel-for-channel
    v = _random_phys_volume(shape=(4, 3, 3, 3), seed=3)
    vol_norm = OC.normalize(v, channel_dim=0)                 # (4,3,3,3)
    pts = torch.from_numpy(v.reshape(4, -1).T)                # (N,4) physical
    pts_norm = OC.normalize_points(pts)                       # (N,4)
    assert torch.allclose(
        pts_norm, torch.from_numpy(vol_norm.reshape(4, -1).T), atol=1e-6)


def test_build_volume_differentiable():
    D, H, W = 4, 5, 6
    mu_a = (torch.rand(D, H, W) * 0.1).requires_grad_(True)   # physical mu_a (leaf)
    fixed = torch.stack([                                     # mu_s, g, n physical
        torch.rand(D, H, W) * 30.0,
        0.8 + torch.rand(D, H, W) * 0.2,
        1.0 + torch.rand(D, H, W) * 0.3,
    ], dim=0)
    vol = OC.build_optical_volume(mu_a, fixed)
    assert vol.shape == (1, 4, D, H, W)
    assert vol.grad_fn is not None                            # differentiable
    # gradient actually reaches mu_a, and ONLY the mu_a channel depends on it
    vol.sum().backward()
    assert mu_a.grad is not None and torch.isfinite(mu_a.grad).all()
    # mu_a channel normalized by span -> d(sum)/d(mu_a) == 1/span everywhere
    span = OC.PHYS_RANGE["mu_a"][1] - OC.PHYS_RANGE["mu_a"][0]
    assert torch.allclose(mu_a.grad, torch.full_like(mu_a.grad, 1.0 / span), atol=1e-5)


def test_build_volume_matches_normalize():
    # build_optical_volume must reproduce the plain normalize() of the same phys volume
    v = _random_phys_volume(shape=(4, 4, 4, 4), seed=5)
    mu_a = torch.from_numpy(v[0])
    fixed = torch.from_numpy(v[1:])
    built = OC.build_optical_volume(mu_a, fixed).squeeze(0)   # (4,4,4,4)
    ref = torch.from_numpy(OC.normalize(v, channel_dim=0))
    assert torch.allclose(built, ref, atol=1e-6), (built - ref).abs().max().item()
