"""
Structured (theta, phi, r) illumination sampling.

Light fluence depends strongly on the source, so the surrogate must see MANY source
configurations to generalize over illumination (and to support illumination
optimization downstream).  This module defines a STRUCTURED grid of sources and
places each on the head surface.

The geometry mirrors pmcx_sim/mcx_fluence_simv2.py EXACTLY (compute_src_dir +
find_source_position) so sources generated here are consistent with the existing
ground truth, but is reimplemented here in pure numpy so this module is importable
and testable without loading the MCX OpenCL .so.

  theta : polar angle (deg)   phi : azimuth (deg)   r : disk-source radius (mm)
  src direction points INWARD; src position is just outside the head surface along
  the outward ray from the volume centroid.

Usage:
    from data_expansion.illumination import structured_grid, place_source, STARTER_GRID
    for cfg in STARTER_GRID:
        illum = place_source(occupancy_vol, **cfg)
"""
from dataclasses import dataclass
from itertools import product
from typing import List, Optional

import numpy as np

from data_expansion.solvers import Illumination

# Source-search parameters (match mcx_fluence_simv2 SRC_R_INI / SRC_R_MARGIN).
SRC_R_INI = 50
SRC_R_MARGIN = 15


# --- "精简起步集" starter grid: 6 sources spread around the head -------------
STARTER_THETAS = (45.0, 90.0, 135.0)     # top-ish / equatorial / bottom-ish
STARTER_PHIS = (45.0, 135.0)             # two azimuths
STARTER_RADII = (10.0,)                  # single disk radius to start


def structured_grid(thetas=STARTER_THETAS, phis=STARTER_PHIS,
                    radii=STARTER_RADII) -> List[dict]:
    """Cartesian product of (theta, phi, r) -> list of source-config dicts."""
    return [dict(theta=float(t), phi=float(p), radius=float(r))
            for t, p, r in product(thetas, phis, radii)]


STARTER_GRID = structured_grid()


def illum_tag(theta: float, phi: float, radius: float) -> str:
    """Filename tag matching the existing convention: th{θ}_ph{φ}_r{r}."""
    return f"th{theta:.0f}_ph{phi:.0f}_r{radius:.0f}"


def compute_src_dir(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Inward-pointing unit direction for polar/azimuth angles (mirrors mcx)."""
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)
    raw = np.array([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)])
    return -raw / np.linalg.norm(raw)            # 指向颅内 (inward)


def mask_centroid(occupancy: np.ndarray) -> np.ndarray:
    return np.argwhere(occupancy).mean(axis=0)


def find_source_position(occupancy: np.ndarray, src_dir: np.ndarray,
                         origin: np.ndarray, r_ini: int = SRC_R_INI,
                         margin: int = SRC_R_MARGIN) -> np.ndarray:
    """
    March outward from `origin` along -src_dir to the first air voxel (occupancy==0)
    that stays clear for 10 more steps, then add `margin`.  Mirrors the mcx
    double-while logic exactly.  `occupancy`: nonzero = tissue, 0 = air/background.
    """
    vol_size = occupancy.shape
    r = float(r_ini)
    max_r = int(max(vol_size) * 2)
    while r < max_r:
        S = (origin + np.round(r * (-src_dir))).astype(int)
        if not all(0 <= S[i] < vol_size[i] for i in range(3)):
            break
        if occupancy[S[0], S[1], S[2]] == 0:
            clear = True
            for step in range(1, 11):
                Sc = (origin + np.round((r + step) * (-src_dir))).astype(int)
                if not all(0 <= Sc[i] < vol_size[i] for i in range(3)):
                    break
                if occupancy[Sc[0], Sc[1], Sc[2]] != 0:
                    clear = False
                    break
            if clear:
                break
        r += 1
    r += margin
    S = np.clip((origin + np.round(r * (-src_dir))).astype(int),
                0, np.array(vol_size) - 1)
    return S


def place_source(occupancy: np.ndarray, theta: float, phi: float, radius: float,
                 origin: Optional[np.ndarray] = None) -> Illumination:
    """Place a disk source for (theta, phi, radius) on the surface of `occupancy`."""
    if origin is None:
        origin = mask_centroid(occupancy)
    src_dir = compute_src_dir(theta, phi)
    src_pos = find_source_position(occupancy, src_dir, origin)
    return Illumination(srcpos=tuple(float(x) for x in src_pos),
                        srcdir=tuple(float(x) for x in src_dir),
                        radius_mm=float(radius))
