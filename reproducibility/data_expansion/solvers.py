"""
Forward fluence solvers (PART B) behind one interface:  solve_fluence(...).

Two backends:
  (i)  'diffusion' — steady-state diffusion approximation, variable coefficient:
            -div(D grad Phi) + mu_a Phi = S,   D = 1/(3(mu_a + mu_s(1-g)))
       Finite-VOLUME discretisation on the voxel grid: harmonic-mean face
       diffusion, Robin (partial-current / extrapolated) boundary, SPD system
       solved by preconditioned CG.  Fast; valid in the highly-scattering /
       diffusive regime.  Source = isotropic point at one transport mean free
       path (1/mu_s') below the entry point (the standard diffusion source).
       Absolute scale is arbitrary (a free constant log-offset).

  (ii) 'mcx' — Monte Carlo via the existing pmcxcl pipeline (mcx_fluence_simv2).
       MC-quality ground truth, in the SAME physical units as the head GT
       (flux * 1000 mJ/mm^2).  Reused, not reimplemented.

The two are compared in tests/test_solver_agreement (RMSE in log10 Phi over the
diffusive interior; near-source / deep low-fluence mismatch is expected and
reported, not hidden).
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optical_config as OC
import repro_config as RC

PMCX_SIM_DIR = os.path.join(str(RC.REPRO_ROOT), "pmcx_sim")


# ---------------------------------------------------------------------------
@dataclass
class Illumination:
    """Source: voxel position (3,), unit direction (3,), disk radius (mm)."""
    srcpos: Tuple[float, float, float]
    srcdir: Tuple[float, float, float]
    radius_mm: float = 10.0

    def dir_unit(self) -> np.ndarray:
        d = np.asarray(self.srcdir, dtype=np.float64)
        return d / (np.linalg.norm(d) + 1e-12)


@dataclass
class DiffusionConfig:
    h_mm: float = 1.0            # voxel size (unitinmm)
    cg_tol: float = 1e-8
    cg_maxiter: int = 5000
    src_depth_mfp: float = 1.0   # diffusion source depth in transport mfp'


# ---------------------------------------------------------------------------
# optical helpers
# ---------------------------------------------------------------------------
def _unpack(optical_volume: np.ndarray):
    """(X,Y,Z,4) physical -> mu_a, mu_s, g, n, mu_s', D (all (X,Y,Z))."""
    assert optical_volume.ndim == 4 and optical_volume.shape[-1] == 4
    o = optical_volume.astype(np.float64)
    mu_a, mu_s, g, n = o[..., 0], o[..., 1], o[..., 2], o[..., 3]
    mu_sp = mu_s * (1.0 - g)                       # reduced scattering
    denom = 3.0 * (mu_a + mu_sp)
    D = np.zeros_like(mu_a)
    np.divide(1.0, denom, out=D, where=denom > 0)
    return mu_a, mu_s, g, n, mu_sp, D


def _A_coef(n: np.ndarray) -> np.ndarray:
    """Boundary mismatch coefficient A=(1+Reff)/(1-Reff) from refractive index n.

    Reff via the Groenhuis/Egan polynomial fit (valid ~1.0-1.5)."""
    Reff = -1.440 / (n ** 2) + 0.710 / n + 0.668 + 0.0636 * n
    Reff = np.clip(Reff, 0.0, 0.999)
    return (1.0 + Reff) / (1.0 - Reff)


# ---------------------------------------------------------------------------
# diffusion finite-volume solver
# ---------------------------------------------------------------------------
def _find_source_voxel(mask, mu_sp, illum: Illumination, h: float,
                       depth_mfp: float) -> Tuple[int, int, int]:
    """Entry = first tissue voxel along the ray; source = +depth_mfp * mfp' inside."""
    shape = np.array(mask.shape)
    pos = np.asarray(illum.srcpos, dtype=np.float64)
    d = illum.dir_unit()

    def in_grid(p):
        return np.all(p >= 0) and np.all(p < shape)

    # march to first tissue voxel
    entry = None
    p = pos.copy()
    for _ in range(int(2 * shape.max())):
        ip = np.round(p).astype(int)
        if in_grid(ip) and mask[tuple(ip)]:
            entry = ip
            break
        p = p + d
    if entry is None:                              # fallback: nearest tissue voxel
        idx = np.argwhere(mask)
        entry = idx[np.argmin(np.linalg.norm(idx - pos, axis=1))]

    mfp_vox = 1.0 / (mu_sp[tuple(entry)] + 1e-9) / h
    s0 = np.round(entry + d * depth_mfp * mfp_vox).astype(int)
    if not (in_grid(s0) and mask[tuple(s0)]):
        s0 = entry                                 # keep it inside tissue
    return tuple(int(x) for x in s0)


def solve_diffusion(optical_volume: np.ndarray, illum: Illumination,
                    cfg: DiffusionConfig = DiffusionConfig(),
                    return_info: bool = False, src_voxel=None):
    """Variable-coefficient steady-state diffusion. Returns Phi (X,Y,Z).

    `src_voxel` (optional) pins the isotropic source voxel, bypassing ray-march
    source finding — used by the equivariance test to isolate the PDE solver."""
    h = cfg.h_mm
    mu_a, mu_s, g, n, mu_sp, D = _unpack(optical_volume)
    mask = mu_sp > 0                                # scattering tissue
    Nt = int(mask.sum())
    assert Nt > 0, "no scattering tissue voxels"

    lin = -np.ones(mask.shape, dtype=np.int64)
    lin[mask] = np.arange(Nt)
    Acoef = _A_coef(n)

    # pad by 1 (outside = non-tissue) so every face is captured incl. grid borders
    Dp = np.pad(D, 1)
    linp = np.pad(lin, 1, constant_values=-1)
    Ap = np.pad(Acoef, 1, mode='edge')

    rows, cols, vals = [], [], []
    diag = np.zeros(Nt)
    diag[:] = mu_a[mask]                            # absorption term
    eps = 1e-30

    for ax in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[ax] = slice(0, -1)
        s1[ax] = slice(1, None)
        ia = linp[tuple(s0)].ravel()
        ib = linp[tuple(s1)].ravel()
        Da = Dp[tuple(s0)].ravel()
        Db = Dp[tuple(s1)].ravel()
        Aa = Ap[tuple(s0)].ravel()
        Ab = Ap[tuple(s1)].ravel()
        ta = ia >= 0
        tb = ib >= 0

        # interior face (both tissue): harmonic-mean conductance
        m = ta & tb
        Dface = 2.0 * Da[m] * Db[m] / (Da[m] + Db[m] + eps)
        cond = Dface / (h * h)
        ria, rib = ia[m], ib[m]
        rows.extend([ria, rib, ria, rib])
        cols.extend([ria, rib, rib, ria])
        vals.extend([cond, cond, -cond, -cond])

        # boundary face for a (neighbor non-tissue): Robin sink
        ba = ta & ~tb
        np.add.at(diag, ia[ba], 1.0 / (2.0 * Aa[ba] * h))
        # boundary face for b
        bb = (~ta) & tb
        np.add.at(diag, ib[bb], 1.0 / (2.0 * Ab[bb] * h))

    rows = np.concatenate(rows + [np.arange(Nt)])
    cols = np.concatenate(cols + [np.arange(Nt)])
    vals = np.concatenate(vals + [diag])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(Nt, Nt)).tocsr()
    A.sum_duplicates()

    # source term: isotropic point at one transport mfp' inside
    s0vox = tuple(int(x) for x in src_voxel) if src_voxel is not None \
        else _find_source_voxel(mask, mu_sp, illum, h, cfg.src_depth_mfp)
    b = np.zeros(Nt)
    b[lin[s0vox]] = 1.0 / (h ** 3)

    M = spla.LinearOperator(A.shape, matvec=lambda x: x / A.diagonal())  # Jacobi
    phi_t, info = spla.cg(A, b, rtol=cfg.cg_tol, maxiter=cfg.cg_maxiter, M=M)
    phi_t = np.clip(phi_t, 0.0, None)

    phi = np.zeros(mask.shape)
    phi[mask] = phi_t
    if return_info:
        return phi, dict(cg_info=info, src_voxel=s0vox, n_tissue=Nt)
    return phi


# ---------------------------------------------------------------------------
# MCX backend (reuse mcx_fluence_simv2's pmcxcl)
# ---------------------------------------------------------------------------
_MCX = None


def _get_mcx():
    global _MCX
    if _MCX is None:
        sys.path.insert(0, PMCX_SIM_DIR)
        import mcx_fluence_simv2 as M           # triggers the .so load workaround
        _MCX = M
    return _MCX


def _build_label_volume(optical_volume: np.ndarray, mask: np.ndarray,
                        max_materials: int = 4096):
    """Map a (mostly piecewise-homogeneous) phantom to (label_vol, prop_table)."""
    o = optical_volume.astype(np.float64)
    flat = o[mask]                                  # (Nt,4)
    uniq, inv = np.unique(np.round(flat, 6), axis=0, return_inverse=True)
    assert len(uniq) <= max_materials, (
        f"{len(uniq)} unique materials > {max_materials}; phantom not piecewise "
        f"homogeneous (use MCX per-voxel mapping instead)")
    label = np.zeros(mask.shape, dtype=np.uint32)   # 0 = background/vacuum
    label[mask] = inv.astype(np.uint32) + 1         # materials 1..K
    prop = np.zeros((len(uniq) + 1, 4), dtype=np.float64)
    prop[0] = [0.0, 0.0, 1.0, 1.0]                  # background (vacuum)
    prop[1:] = uniq
    return label, prop


def solve_mcx(optical_volume: np.ndarray, illum: Illumination,
              nphoton: float = 5e6, gpuid: int = 1, time_gate: float = 5e-8,
              h_mm: float = 1.0, seed: int = 29012392, return_info: bool = False):
    """Monte Carlo fluence (mJ/mm^2, same units as head GT). Returns Phi (X,Y,Z)."""
    M = _get_mcx()
    mu_a, mu_s, g, n, mu_sp, D = _unpack(optical_volume)
    mask = (mu_s > 0)                               # match head tissue convention
    label, prop = _build_label_volume(optical_volume, mask)

    cfg = {
        "vol": label, "prop": prop,
        "srcpos": list(map(float, illum.srcpos)),
        "srcdir": list(map(float, illum.dir_unit())),
        "srctype": "disk", "srcparam1": [float(illum.radius_mm), 0.0, 0.0, 0.0],
        "tstart": 0.0, "tend": float(time_gate), "tstep": float(time_gate),
        "issrcfrom0": 1, "seed": int(seed), "nphoton": int(nphoton),
        "unitinmm": float(h_mm), "outputtype": "fluence",
        "isreflect": 0, "isspecular": 0, "gpuid": gpuid, "autopilot": 1,
    }
    res = M.pmcx.run(cfg)
    flux = res["flux"]
    if flux.ndim == 4 and flux.shape[3] == 1:
        flux = flux[:, :, :, 0]
    phi = flux * 1000.0                             # mJ/mm^2 (head-GT convention)
    if return_info:
        return phi, dict(n_materials=len(prop) - 1)
    return phi


# ---------------------------------------------------------------------------
# unified interface
# ---------------------------------------------------------------------------
def solve_fluence(optical_volume: np.ndarray, illumination: Illumination,
                  backend: str = "diffusion", **kw) -> np.ndarray:
    """Forward solve. backend in {'diffusion','mcx'}. Returns Phi (X,Y,Z)."""
    if backend == "diffusion":
        return solve_diffusion(optical_volume, illumination,
                               cfg=kw.get("cfg", DiffusionConfig()),
                               return_info=kw.get("return_info", False))
    if backend == "mcx":
        return solve_mcx(optical_volume, illumination,
                         **{k: v for k, v in kw.items() if k != "cfg"})
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# comparison utility (used by the solver-agreement test & calibration)
# ---------------------------------------------------------------------------
def diffusive_interior_mask(phi_d, phi_m, src_voxel, mu_sp,
                            src_exclude_mfp: float = 3.0, boundary: int = 8,
                            rel_floor: float = 1e-5):
    """
    Region where a diffusion-vs-MC comparison is physically meaningful:
      - at least `src_exclude_mfp` transport mfp' from the source (diffusion
        approximation breaks down near the collimated entry),
      - at least `boundary` voxels from the grid edge,
      - BOTH fields within `rel_floor` of their own maxima (exclude deep,
        photon-starved / numerically-underflowed tails — documented mismatch).
    """
    shape = phi_d.shape
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    src = np.asarray(src_voxel)
    dist = np.sqrt((zz - src[0]) ** 2 + (yy - src[1]) ** 2 + (xx - src[2]) ** 2)
    mfp = 1.0 / (np.median(mu_sp[mu_sp > 0]) + 1e-12)
    interior = np.zeros(shape, bool)
    interior[boundary:-boundary, boundary:-boundary, boundary:-boundary] = True
    return (interior & (dist > src_exclude_mfp * mfp)
            & (phi_d > phi_d.max() * rel_floor)
            & (phi_m > phi_m.max() * rel_floor))


def log10_agreement(phi_a: np.ndarray, phi_b: np.ndarray, mask: np.ndarray,
                    floor: float = 1e-12):
    """
    RMSE in log10 between two fluence fields over `mask`, after removing the
    single best additive log-offset (absolute scale is a free constant of the
    diffusion source).  Returns (rmse, offset, n_vox).
    """
    a = np.log10(np.clip(phi_a[mask], floor, None))
    b = np.log10(np.clip(phi_b[mask], floor, None))
    offset = np.median(b - a)                       # align scales
    resid = (a + offset) - b
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return rmse, float(offset), int(mask.sum())
