#!/usr/bin/env python3
"""
Single source of truth for optical-channel metadata + FIXED physical-range
normalization for the FM-INR optical-fluence pipeline.

WHY FIXED RANGES (not per-volume min/max):
  The FM encoder and the INR consume normalized optical volumes.  Historically
  both the encoder-training dataset (FM_tune_t4.OpticalPropertyDataset) and
  extract_pyramids.py normalized *each volume by its own per-channel min/max*.
  That is undefined at INVERSION time, because mu_a is the unknown we are solving
  for — its min/max depend on the answer.  This module replaces that with FIXED
  physical bounds so the encoder-input pipeline is well-defined whether mu_a is
  known (training) or unknown/being-optimized (inversion).

CHANNEL SEMANTICS (confirmed against the data + forward model):
  Order: (mu_a, mu_s, g, n).  Units: mu_a, mu_s in mm^-1; g, n dimensionless.
    - Source: pmcx_sim/mcx_fluence_simv2.py  ("格式: [mua, mus, g, n]  单位: mm⁻¹",
      and the property volumes carry key 'vol_prop_eye_aseg' with this axis order).
    - mu_s is RAW scattering, NOT reduced scattering.  g is a *separate* channel,
      and the reduced coefficient is formed downstream as  mu_s' = mu_s * (1 - g)
      (see the diffusion PINN:  D = 1/(3*(mu_a + mu_s*(1-g)))).  Hence
      MU_S_IS_REDUCED = False.

PHYS_RANGE: fixed (lo, hi) per channel.  Chosen to cover the empirical span of
  the current 16-volume training set with headroom, and to stay physically
  plausible for brain/soft tissue at NIR wavelengths.  Empirical global span
  measured across all dataset/*.mat:
      mu_a in [0,   0.110]    mu_s in [0,  40.10]    g in [0.85, 1.0]    n in [1.0, 1.371]
  Plausibility cross-check (Jacques 2013, "Optical properties of biological
  tissues: a review", Phys. Med. Biol. 58:R37): soft-tissue mu_a at NIR spans
  ~1e-3..~0.1+ mm^-1, reduced scattering mu_s' ~ 0.5..5 mm^-1 (here raw mu_s is
  larger because g ~ 0.9), g ~ 0.8..0.95, tissue refractive index n ~ 1.33..1.40.
  Bounds below bracket the data with inversion headroom; they are NOT invented
  tight values — widen here (single edit) if a new species/wavelength is added.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Sequence

import numpy as np

try:                                    # torch is optional for the pure-numpy paths
    import torch
    _HAS_TORCH = True
except Exception:                       # pragma: no cover
    torch = None
    _HAS_TORCH = False


CHANNELS: Tuple[str, ...] = ("mu_a", "mu_s", "g", "n")

# mu_s is raw scattering; reduced scattering is derived as mu_s*(1-g) downstream.
MU_S_IS_REDUCED: bool = False

UNITS: Dict[str, str] = {"mu_a": "mm^-1", "mu_s": "mm^-1", "g": "", "n": ""}

# Fixed physical (lo, hi) bounds — see module docstring for sourcing.
# These are the UNION over all training data sources (head MRI volumes +
# rigid-augmented heads + virtual phantoms).  They MUST be final before the
# encoder/INR retrain bakes normalization in.  Sourcing:
#   - heads (empirical, all dataset/*.mat): mu_a<=0.110, mu_s<=40.10, g>=0.85, n<=1.371
#   - rigid augmentation reuses head optical values -> no new range
#   - phantoms (data_expansion): background mu_a in [0.005,0.02], mu_s in [5,30],
#     g in [0.85,0.95], n in [1.33,1.40]; absorbing inclusions mu_a up to 0.20.
#     -> mu_a upper bound RAISED 0.15 -> 0.25 so high-contrast phantom inclusions
#        (and inversion excursions) stay strictly inside the range with headroom.
#        Heads then occupy mu_a's lower ~44%; phantoms supply the high-mu_a signal
#        the head set lacks. Jacques 2013 cross-check: tissue/blood mu_a at NIR
#        can reach ~0.2-0.5 mm^-1, so 0.25 is physically conservative.
PHYS_RANGE: Dict[str, Tuple[float, float]] = {
    "mu_a": (0.0, 0.25),     # heads<=0.110, phantom inclusions<=0.20, +headroom
    "mu_s": (0.0, 45.0),     # heads<=40.10, phantom bg<=30
    "g":    (0.80, 1.0),     # heads>=0.85, phantom 0.85-0.95
    "n":    (1.0, 1.40),     # heads<=1.371, phantom 1.33-1.40
}


@dataclass(frozen=True)
class OpticalConfig:
    """Immutable bundle describing the optical channels and their fixed ranges."""
    channels: Tuple[str, ...] = CHANNELS
    phys_range: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(PHYS_RANGE))
    mu_s_is_reduced: bool = MU_S_IS_REDUCED
    units: Dict[str, str] = field(default_factory=lambda: dict(UNITS))

    def lo_hi(self, channels: Sequence[str]):
        lo = [self.phys_range[c][0] for c in channels]
        hi = [self.phys_range[c][1] for c in channels]
        return lo, hi


CONFIG = OpticalConfig()


# ---------------------------------------------------------------------------
# normalize / denormalize against FIXED ranges (type-agnostic: numpy or torch)
# ---------------------------------------------------------------------------
def _is_torch(x) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def _channel_slices(vol, channel_dim):
    """Yield each channel slice of `vol` along `channel_dim` (keeps grad for torch)."""
    nd = vol.ndim
    cd = channel_dim % nd
    for c in range(vol.shape[cd]):
        idx = [slice(None)] * nd
        idx[cd] = c
        yield vol[tuple(idx)]


def _stack(slices, channel_dim, like):
    if _is_torch(like):
        return torch.stack(list(slices), dim=channel_dim)
    return np.stack(list(slices), axis=channel_dim)


def normalize(vol, channels: Sequence[str] = CHANNELS, channel_dim: int = 0,
              config: OpticalConfig = CONFIG, clamp: bool = False):
    """
    Min-max normalize against FIXED PHYS_RANGE (NOT per-volume statistics).
    `vol` may be numpy or torch; the channel axis is `channel_dim` and must have
    length == len(channels).  Returns the same type/shape; differentiable for torch.
    """
    assert vol.shape[channel_dim] == len(channels), (
        f"channel_dim {channel_dim} has size {vol.shape[channel_dim]}, "
        f"expected {len(channels)} for channels {channels}")
    lo, hi = config.lo_hi(channels)
    out = []
    for c, ch in enumerate(_channel_slices(vol, channel_dim)):
        span = hi[c] - lo[c]
        nc = (ch - lo[c]) / span
        if clamp:
            nc = nc.clamp(0.0, 1.0) if _is_torch(nc) else np.clip(nc, 0.0, 1.0)
        out.append(nc)
    return _stack(out, channel_dim, vol)


def denormalize(vol_norm, channels: Sequence[str] = CHANNELS, channel_dim: int = 0,
                config: OpticalConfig = CONFIG):
    """Inverse of `normalize`: map [0,1] back to physical units."""
    assert vol_norm.shape[channel_dim] == len(channels)
    lo, hi = config.lo_hi(channels)
    out = []
    for c, ch in enumerate(_channel_slices(vol_norm, channel_dim)):
        out.append(ch * (hi[c] - lo[c]) + lo[c])
    return _stack(out, channel_dim, vol_norm)


# ---------------------------------------------------------------------------
# Differentiable volume builder for inversion (mu_a is the optimized unknown)
# ---------------------------------------------------------------------------
def build_optical_volume(mu_a_phys, fixed_props_phys,
                         channels: Sequence[str] = CHANNELS,
                         config: OpticalConfig = CONFIG):
    """
    Assemble a NORMALIZED (1, C, D, H, W) optical volume from a (possibly
    requires_grad) physical mu_a field and the fixed physical mu_s/g/n channels.

    Args:
      mu_a_phys       : torch tensor, physical mu_a (mm^-1).  Shape (D,H,W) or
                        (1,1,D,H,W) or (1,D,H,W).  May require grad.
      fixed_props_phys: torch tensor of the NON-mu_a channels in `channels` order,
                        physical units.  Shape (C-1, D, H, W) or (1, C-1, D, H, W).

    Returns:
      (1, C, D, H, W) normalized tensor.  Differentiable in mu_a_phys
      (pure tensor ops — no .numpy()/.item()/in-place on leaves).
    """
    assert _HAS_TORCH, "build_optical_volume requires torch"
    # squeeze mu_a to (D,H,W)
    ma = mu_a_phys
    while ma.dim() > 3:
        ma = ma.squeeze(0)
    # squeeze fixed props to (C-1, D, H, W)
    fp = fixed_props_phys
    if fp.dim() == 5:
        fp = fp.squeeze(0)
    assert fp.shape[0] == len(channels) - 1, (
        f"fixed_props_phys has {fp.shape[0]} channels, expected {len(channels) - 1}")

    # interleave mu_a with the fixed channels in `channels` order
    mu_a_name = "mu_a"
    assert channels[0] == mu_a_name or mu_a_name in channels
    fixed_iter = iter(fp[i] for i in range(fp.shape[0]))
    stacked = []
    for c in channels:
        stacked.append(ma if c == mu_a_name else next(fixed_iter))
    vol_phys = torch.stack(stacked, dim=0)              # (C, D, H, W)
    vol_norm = normalize(vol_phys, channels=channels, channel_dim=0, config=config)
    return vol_norm.unsqueeze(0)                        # (1, C, D, H, W)


# ---------------------------------------------------------------------------
# INR per-point optical normalization (single definition shared with volume path)
# ---------------------------------------------------------------------------
def normalize_points(optical_phys, channels: Sequence[str] = CHANNELS,
                     config: OpticalConfig = CONFIG):
    """Normalize a per-point (N, C) physical optical tensor against PHYS_RANGE."""
    return normalize(optical_phys, channels=channels, channel_dim=-1, config=config)


if __name__ == "__main__":          # quick self-check
    print("CHANNELS:", CHANNELS, "| MU_S_IS_REDUCED:", MU_S_IS_REDUCED)
    for c in CHANNELS:
        print(f"  {c:5s} range {PHYS_RANGE[c]}  unit '{UNITS[c]}'")
