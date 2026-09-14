"""
Virtual geometric phantoms (PART B input).

A phantom is a homogeneous highly-scattering background (physiological mu_s/g/n)
filling the whole cubic grid, with embedded absorbing inclusions (spheres /
cylinders / slabs) of randomised position, size and absorption contrast.  g and n
are kept HOMOGENEOUS across the phantom so the diffusion and Monte-Carlo solvers
see matched boundary/refraction physics (cleaner solver agreement); mu_a and mu_s
vary by region.  All optical values are drawn to lie strictly inside
optical_config.PHYS_RANGE.

Illumination (srcpos voxel / srcdir unit / disk radius) is randomised per phantom:
a disk source sits just inside a randomly chosen face, pointing inward.

generate_phantom(rng, cfg) -> (optical_volume (X,Y,Z,4) physical, Illumination, meta)
"""
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .solvers import Illumination


@dataclass
class PhantomConfig:
    shape: Tuple[int, int, int] = (96, 96, 96)
    # background (physiological soft tissue, NIR)
    bg_mu_a: Tuple[float, float] = (0.005, 0.02)
    bg_mu_s: Tuple[float, float] = (5.0, 30.0)
    g_range: Tuple[float, float] = (0.85, 0.95)     # homogeneous per phantom
    n_range: Tuple[float, float] = (1.33, 1.40)     # homogeneous per phantom
    # absorbing inclusions (higher mu_a contrast)
    n_inclusions: Tuple[int, int] = (1, 3)
    inc_mu_a: Tuple[float, float] = (0.05, 0.20)
    inc_mu_s: Tuple[float, float] = (5.0, 30.0)
    inc_radius_vox: Tuple[float, float] = (6.0, 16.0)
    src_radius_mm: Tuple[float, float] = (5.0, 10.0)
    src_inset_vox: int = 2                           # source depth inside the face
    src_tilt: float = 0.25                           # max lateral tilt of srcdir


def _u(rng, lohi):
    return float(rng.uniform(lohi[0], lohi[1]))


def _sphere_mask(shape, center, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((xx - center[2]) ** 2 + (yy - center[1]) ** 2
            + (zz - center[0]) ** 2) <= radius ** 2


def _cylinder_mask(shape, center, radius, axis):
    grids = np.ogrid[:shape[0], :shape[1], :shape[2]]
    perp = [a for a in range(3) if a != axis]
    d2 = sum((grids[a] - center[a]) ** 2 for a in perp)   # size-1 on cyl axis
    return np.broadcast_to(d2 <= radius ** 2, shape)      # broadcast along cyl axis


def _slab_mask(shape, axis, pos, thickness):
    idx = np.arange(shape[axis])
    sel = np.abs(idx - pos) <= thickness / 2.0
    sh = [1, 1, 1]
    sh[axis] = shape[axis]
    m = sel.reshape(sh)
    return np.broadcast_to(m, shape)


def generate_phantom(rng: np.random.Generator, cfg: PhantomConfig = PhantomConfig()):
    """Return (optical_volume (X,Y,Z,4) physical, Illumination, meta dict)."""
    shape = cfg.shape
    g = _u(rng, cfg.g_range)
    n = _u(rng, cfg.n_range)
    mu_a_bg = _u(rng, cfg.bg_mu_a)
    mu_s_bg = _u(rng, cfg.bg_mu_s)

    mu_a = np.full(shape, mu_a_bg, dtype=np.float64)
    mu_s = np.full(shape, mu_s_bg, dtype=np.float64)

    n_inc = int(rng.integers(cfg.n_inclusions[0], cfg.n_inclusions[1] + 1))
    inclusions = []
    for _ in range(n_inc):
        kind = rng.choice(["sphere", "cylinder", "slab"])
        radius = _u(rng, cfg.inc_radius_vox)
        center = [rng.integers(int(radius), shape[a] - int(radius)) for a in range(3)]
        if kind == "sphere":
            m = _sphere_mask(shape, center, radius)
        elif kind == "cylinder":
            axis = int(rng.integers(0, 3))
            m = _cylinder_mask(shape, center, radius, axis)
        else:
            axis = int(rng.integers(0, 3))
            m = _slab_mask(shape, axis, center[axis], max(4.0, radius))
        ia = _u(rng, cfg.inc_mu_a)
        is_ = _u(rng, cfg.inc_mu_s)
        mu_a[m] = ia
        mu_s[m] = is_
        inclusions.append(dict(kind=str(kind), center=[int(c) for c in center],
                               radius=float(radius), mu_a=ia, mu_s=is_))

    optical = np.stack([mu_a, mu_s,
                        np.full(shape, g), np.full(shape, n)], axis=-1)  # (X,Y,Z,4)

    # --- illumination: disk source just inside a random face, pointing inward ---
    axis = int(rng.integers(0, 3))
    side = int(rng.choice([0, 1]))                    # 0 = low face, 1 = high face
    srcpos = np.array([rng.integers(int(0.25 * s), int(0.75 * s)) for s in shape],
                      dtype=np.float64)
    inset = cfg.src_inset_vox
    srcpos[axis] = inset if side == 0 else shape[axis] - 1 - inset
    inward = 1.0 if side == 0 else -1.0
    srcdir = np.zeros(3)
    srcdir[axis] = inward
    # small lateral tilt
    for a in range(3):
        if a != axis:
            srcdir[a] = rng.uniform(-cfg.src_tilt, cfg.src_tilt)
    srcdir = srcdir / np.linalg.norm(srcdir)
    illum = Illumination(srcpos=tuple(srcpos.tolist()),
                         srcdir=tuple(srcdir.tolist()),
                         radius_mm=_u(rng, cfg.src_radius_mm))

    meta = dict(shape=list(shape), g=g, n=n, mu_a_bg=mu_a_bg, mu_s_bg=mu_s_bg,
                inclusions=inclusions, src_axis=axis, src_side=side)
    return optical, illum, meta
