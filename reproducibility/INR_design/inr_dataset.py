#!/usr/bin/env python3
"""
Real INR training data for the steady-state (CW) FM-informed fluence surrogate.

A "scene" = one (sample, wavelength) pair for which we have:
  - property volume  dataset/{sample}_copmri_withHermiteF{wl}.mat   (X,Y,Z,4)
  - feature pyramid  extracted_pyramids/{...}_pyramid.pt             (5 scales)
  - GT fluence       pmcx_sim/sim_t1/data/wl{wl}/fluence_{sample}_F{wl}_th75_ph75_r10.mat
  - light config     pmcx_sim/sim_t1/data/wl{wl}/meta_{sample}_F{wl}_th75_ph75_r10.json

Design choices (per project spec):
  - Steady-state only (no time axis).
  - Wavelength is NOT an explicit input — it is encoded in (mu_a, mu_s, g, n),
    so scenes of all wavelengths are pooled together.
  - Property volume and fluence share the SAME (X,Y,Z) grid (verified), so a
    voxel coordinate indexes both directly.

Fluence spans many orders of magnitude → we supervise on log10(fluence).
"""
import os
import re
import sys
import glob
import json
import torch
import numpy as np
import scipy.io as sio

# Shared optical-channel config / FIXED-range normalization (single source of truth).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optical_config as OC
import repro_config as RC

DATASET_DIR = RC.DATASET_DIR
PYRAMID_DIR = RC.PYRAMID_DIR
SIM_DIR = RC.SIM_DATA_DIR

# The INR's per-point optical normalization uses the SAME fixed-range definition
# as the encoder input (optical_config.normalize_points), so there is one
# normalization convention across the whole pipeline.  The old divide-by-scale
# OPT_SCALE = [0.10, 40, 1, 1.5] was inconsistent (no lower bound for g/n) and is
# retired in favour of min-max against optical_config.PHYS_RANGE.

# Fluence floor for the log10 supervision.
# Light fluence decays exponentially, so deep voxels reach the Monte-Carlo noise
# floor (and true zeros) ~1e-8..0 — taking log10 of those injects huge negative
# outliers (and an artificial spike at log10(1e-12)=-12) that dominate the MSE and
# corrupt training.  We therefore truncate to a RELATIVE per-scene dynamic range:
# everything below (peak / 10**FLU_FLOOR_DECADES) is clamped to that floor, keeping
# only the reliable top decades.  This both removes the noisy tail and avoids the
# inflated -12 dynamic range.  Tune via env FLU_FLOOR_DECADES.
# Measured tissue distribution (head, F810): median log10(fluence) ~ -11.7 and a
# tail down to ~1e-41 — i.e. >50% of tissue is unphysical Monte-Carlo dust (sub-
# single-photon accumulation) + 6% exact zeros.  We clamp ~8 decades below the
# per-scene peak (≈ the MC noise floor) so the supervised range is the genuinely
# illuminated tissue; everything darker becomes a single defined "dark" level
# instead of -41/-12 outliers.
FLU_FLOOR_DECADES = float(os.environ.get("FLU_FLOOR_DECADES", 8.0))
FLU_EPS = 1e-12     # absolute floor (only used if a scene is all-zero)

# ---- domain-aware conditioning (shared encoder + lightweight domain awareness) ----
# We keep ONE shared encoder/INR and inject category + physical-scale conditioning,
# rather than forking per species.  The physically meaningful cross-species variable
# is the VOXEL SPACING (mm/voxel): a 64^3 crop spans ~6 cm of a 1 mm human head but
# ~6 mm of a 0.1 mm mouse, so the same normalized coordinate is a different physical
# distance and fluence (decay is in mm).  All current data is 1 mm; mouse will differ.
DOMAIN_NAMES = ("head", "phantom", "mouse")
DEFAULT_SPACING_MM = 1.0


def scene_domain(sample):
    """0 = human head (continuous / augmented / scatterBrains), 1 = phantom, 2 = mouse."""
    if sample.startswith("phantom"):
        return 1
    if sample.startswith("mouse") or sample.startswith("ms_"):
        return 2
    return 0


def scene_spacing_mm(sample):
    """Physical voxel spacing (mm). All current sources are 1 mm isotropic."""
    return DEFAULT_SPACING_MM


def discover_scenes():
    """
    Return one scene per (geometry, illumination): a geometry = (property volume +
    feature pyramid) keyed by (sample, wl); each geometry may have MULTIPLE
    illuminations (distinct fluence + meta files tagged by source angles/radius).
    The shared `pyramid`/`prop` paths let SceneStore cache the geometry once and
    swap only the (cheap) per-illumination fluence — so multi-illumination training
    does not reload the ~2.5 GB pyramid per source.
    Backward compatible: a geometry with one fluence yields one scene.
    """
    scenes = []
    pyr_files = sorted(glob.glob(os.path.join(PYRAMID_DIR, "*_pyramid.pt")))
    for pyr in pyr_files:
        base = os.path.basename(pyr)
        m = re.match(r"(.+)_copmri_withHermiteF(\d+)_pyramid\.pt$", base)
        if not m:
            continue
        sample, wl = m.group(1), m.group(2)
        prop = os.path.join(DATASET_DIR, f"{sample}_copmri_withHermiteF{wl}.mat")
        if not os.path.isfile(prop):
            continue
        wld = os.path.join(SIM_DIR, f"wl{wl}")
        for flu in sorted(glob.glob(os.path.join(wld, f"fluence_{sample}_F{wl}_*.mat"))):
            fm = re.match(rf"fluence_{re.escape(sample)}_F{wl}_(.+)\.mat$",
                          os.path.basename(flu))
            if not fm:
                continue
            illum = fm.group(1)                                  # e.g. th45_ph90_r10
            meta = os.path.join(wld, f"meta_{sample}_F{wl}_{illum}.json")
            if os.path.isfile(meta):
                scenes.append(dict(sample=sample, wl=wl, illum=illum, geom_key=pyr,
                                   pyramid=pyr, prop=prop, fluence=flu, meta=meta))
    return scenes


def group_by_geometry(scenes):
    """Group scenes by geometry (shared pyramid) -> {geom_key: [scene, ...]}.
    Iterate geometry-major in training so the cached pyramid is reused across
    that geometry's illuminations."""
    groups = {}
    for s in scenes:
        groups.setdefault(s['geom_key'], []).append(s)
    return groups


class SceneStore:
    """
    Two-level cache for multi-illumination training.

    GEOMETRY (property volume + ~2.5 GB feature pyramid + tissue mask) is shared by
    all illuminations of a (sample, wl) and cached once (MRU, one at a time to bound
    GPU memory).  ILLUMINATION (fluence -> log10 + lit voxels + 6-d light vector) is
    cheap and swapped per scene.  So iterating a geometry's illuminations does NOT
    reload the pyramid.  Iterate geometry-major (see group_by_geometry) for best reuse.
    """
    def __init__(self, device):
        self.device = device
        self._geo_key = None
        self._geo = None
        self._illum_key = None
        self._illum = None

    def _load_geometry(self, scene):
        prop = sio.loadmat(scene['prop'])['vol_prop_eye_aseg'].astype(np.float32)
        prop = np.transpose(prop, (3, 0, 1, 2))                 # (4,X,Y,Z)
        prop_t = torch.from_numpy(prop).unsqueeze(0).to(self.device)  # (1,4,X,Y,Z)
        X, Y, Z = prop_t.shape[-3:]
        tissue_np = prop[1] > 0                                  # (X,Y,Z) mu_s>0
        valid_idx = torch.nonzero(prop_t[0, 1] > 0, as_tuple=False)   # (M,3)
        pyr_raw = torch.load(scene['pyramid'], map_location='cpu')['pyramid']
        pyramid = [p.to(self.device) for p in pyr_raw]
        return dict(prop=prop_t, pyramid=pyramid, valid_idx=valid_idx,
                    tissue_np=tissue_np,
                    vol_shape=torch.tensor([X, Y, Z], device=self.device))

    def _load_illumination(self, scene, geo):
        X, Y, Z = [int(v) for v in geo['vol_shape'].tolist()]
        # --- GT fluence -> log10 with a per-scene relative dynamic-range floor ---
        flu = sio.loadmat(scene['fluence'])['fluence'].astype(np.float32)
        fmax = float(flu.max())
        floor = max(FLU_EPS, fmax * (10.0 ** (-FLU_FLOOR_DECADES)))
        flu = np.clip(flu, floor, None)
        logflu = np.log10(flu)
        logflu_t = torch.from_numpy(logflu).unsqueeze(0).unsqueeze(0).to(self.device)

        # "lit" voxels: tissue AND above the floor (for stratified sampling)
        bright_np = geo['tissue_np'] & (logflu > np.log10(floor) + 1e-4)
        bright_idx = torch.from_numpy(np.argwhere(bright_np)).to(self.device)

        # --- light vector [srcpos_norm(3), srcdir(3)] ---
        with open(scene['meta']) as f:
            meta = json.load(f)
        srcpos = torch.tensor(meta['srcpos'], dtype=torch.float32)
        srcdir = torch.tensor(meta['srcdir'], dtype=torch.float32)
        srcpos_norm = srcpos / (torch.tensor([X, Y, Z], dtype=torch.float32) - 1)
        light = torch.cat([srcpos_norm, srcdir]).to(self.device)         # (6,)
        return dict(logflu=logflu_t, bright_idx=bright_idx, light=light,
                    srcpos=srcpos.to(self.device))

    def get(self, scene):
        if scene['geom_key'] != self._geo_key:
            self._geo = None                       # free old geometry (pyramid)
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self._geo = self._load_geometry(scene)
            self._geo_key = scene['geom_key']
            self._illum_key = None                 # force illumination reload
        if scene['fluence'] != self._illum_key:
            self._illum = self._load_illumination(scene, self._geo)
            self._illum_key = scene['fluence']
        return {**self._geo, **self._illum, 'vol_shape': self._geo['vol_shape']}


def grid_from_xyz(xyz_phys, vol_shape):
    """
    Map physical (D,H,W)=(X,Y,Z) coords to a grid_sample grid.
    xyz_phys: (N,3) in voxel units (x,y,z).  Returns (1,1,1,N,3) grid in [-1,1]
    with last-dim order (w,h,d) as required by F.grid_sample, align_corners=True.
    """
    D, H, W = vol_shape
    xn = (xyz_phys[:, 0] / (D - 1)) * 2 - 1
    yn = (xyz_phys[:, 1] / (H - 1)) * 2 - 1
    zn = (xyz_phys[:, 2] / (W - 1)) * 2 - 1
    grid = torch.stack([zn, yn, xn], dim=-1)        # (N,3) -> (w,h,d)
    return grid.view(1, 1, 1, -1, 3)


def sample_volume(vol, xyz_phys, vol_shape, mode='bilinear'):
    """Trilinearly sample a (1,C,X,Y,Z) volume at (N,3) coords -> (N,C)."""
    grid = grid_from_xyz(xyz_phys, vol_shape)
    out = torch.nn.functional.grid_sample(vol, grid, mode=mode, align_corners=True)
    return out.squeeze(0).squeeze(1).squeeze(1).permute(1, 0)   # (N,C)


def sample_points(scene_data, n_points, jitter=True, generator=None, bright_frac=0.5):
    """
    Draw n_points continuous coordinates from inside the tissue mask, STRATIFIED:
    a fraction `bright_frac` from "informative" (illuminated, above-floor) voxels
    and the rest from all tissue, so the large clamped dark region does not dominate
    the data loss.  bright_frac=0 reproduces uniform tissue sampling.
    Returns:
      xyz   : (N,3) float coords (voxel units)
      optical: (N,4) normalized optical params
      gt_log : (N,1) log10 GT fluence
    """
    idx = scene_data['valid_idx']
    bidx = scene_data.get('bright_idx')
    dev = idx.device
    if bidx is None or bidx.shape[0] < 8:
        bright_frac = 0.0
    n_bright = int(round(bright_frac * n_points))
    parts = []
    if n_bright > 0:
        s = torch.randint(0, bidx.shape[0], (n_bright,), device=dev, generator=generator)
        parts.append(bidx[s].float())
    if n_points - n_bright > 0:
        s = torch.randint(0, idx.shape[0], (n_points - n_bright,), device=dev, generator=generator)
        parts.append(idx[s].float())
    xyz = torch.cat(parts, dim=0)
    if jitter:
        xyz = xyz + (torch.rand_like(xyz) - 0.5)    # +-0.5 voxel sub-voxel jitter
        xyz = xyz.clamp(min=0)
        maxc = (scene_data['vol_shape'].float() - 1)
        xyz = torch.minimum(xyz, maxc)

    optical = sample_volume(scene_data['prop'], xyz, scene_data['vol_shape'])  # (N,4) physical
    optical = OC.normalize_points(optical)                                     # fixed-range
    gt_log  = sample_volume(scene_data['logflu'], xyz, scene_data['vol_shape'])  # (N,1)
    return xyz, optical, gt_log
