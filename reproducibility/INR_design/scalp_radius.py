#!/usr/bin/env python3
"""Differentiable scalp radius rm(A,B) for gradient-based illumination inverse design.

THE PROBLEM. `illum_opt*.py` places the source at srcpos = origin + rm * n(A,B), where rm comes from
`illumination.find_source_position` -- a voxel march, run in numpy on a DETACHED direction. So rm is a
constant to autograd, and d(srcpos)/d(A,B) keeps only the tangential term rm * dn/d(A,B), dropping the
radial term drm/d(A,B) * n.

WHY THAT IS NOT HARMLESS. For the true physics a collimated beam is invariant to sliding the source
along its own axis, so the dropped radial term ought to contribute nothing. The SURROGATE is not
invariant: srcpos enters it three ways (light[0:3], src_features r/cos, path_features), and it was only
ever trained with the source ON the scalp, so off-scalp is extrapolation. Measured on sh001 at the
grid optimum (A=73, B=-170), target depth 30 mm:
    dJ/d(radial)          = -0.129 per mm          (physics says 0)
    dJ/dA full FD         = +0.0787                (the true gradient of J(A,B))
    dJ/dA with rm pinned  = -0.0316                (what autograd currently computes)
    => the dropped rm term is +0.110, i.e. 140% of the gradient, and it FLIPS THE SIGN.
That reproduces the archived "autograd dJ/dA has the wrong sign" bug. drm/dA is about -1.09 mm/deg,
so a fixed-radius sphere is also a bad forward approximation: over the A in [-8,85] optimisation box
the scalp radius moves by ~100 mm.

THE FIX. rm(A,B) is a smooth-ish property of the head surface, so tabulate it once per head with the
EXACT same find_source_position call (no convention drift) and interpolate bilinearly at query time.
The interpolation is differentiable, B is periodic, A is clamped. Note we are differentiating a
SMOOTHED version of a function that is intrinsically discrete (voxel march + round + 15-voxel margin);
the table spacing sets how much of that staircase is smoothed away.

  rad = ScalpRadius(occ, origin, dev)          # ~6 s per head at the 2-degree default
  rm  = rad(A_deg, B_deg)                      # torch scalar, differentiable in A_deg/B_deg
"""
import os, sys, math
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "data_expansion")):
    sys.path.insert(0, p)
import illumination as ILL


def _n_of(A_deg, B_deg):
    """numpy twin of illum_opt.srcdir_torch's outward normal n (srcdir = -n)."""
    A, B = math.radians(A_deg), math.radians(B_deg)
    horiz = math.cos(B) * np.array([0., 1., 0.]) + math.sin(B) * np.array([1., 0., 0.])
    n = math.cos(A) * horiz + math.sin(A) * np.array([0., 0., 1.])
    return n / np.linalg.norm(n)


class ScalpRadius:
    """Tabulated radius r(A,B) from `origin` along the outward normal, bilinearly interpolated.

    mode="source": the mcx source point (find_source_position, 15 voxels outside the scalp).
    mode="entry" : the V17 beam ENTRY point on the scalp, i.e. what make_entryfeat.entry_of returns.
                   This is the one an entry-referenced (V17) surrogate needs, and tabulating it is
                   what makes the entry a SMOOTH function of (A,B): entry_of itself inherits the
                   integer-voxel staircase of find_source_position, which is precisely what turned
                   J(A,B) into a 0.26-decade sawtooth. Requires occ_t (torch) and vs.
    """

    def __init__(self, occ, origin, dev, dA=2.0, dB=2.0, A_range=(-90.0, 90.0),
                 mode="source", occ_t=None, vs=None):
        self.dev = dev
        self.A0, self.A1 = A_range
        self.dA, self.dB = dA, dB
        self.mode = mode
        self.nA = int(round((self.A1 - self.A0) / dA)) + 1
        self.nB = int(round(360.0 / dB))                       # B is periodic: [-180, 180)
        origin = np.asarray(origin, dtype=float)
        if mode == "entry":
            import illumination as _ILL
            MARGIN = _ILL.SRC_R_MARGIN
            shp = np.asarray(occ.shape)

            def _entry_r(sp, n):
                """numpy twin of make_entryfeat.entry_of, in RADIUS form. Same algorithm (round to a
                voxel, 0.5-step walk, both directions) so the table matches what the trainer used;
                done in numpy because the torch version syncs the GPU on every step, which makes a
                65k-cell table take many minutes."""
                srcdir = -n
                e = np.asarray(sp, float) + MARGIN * srcdir

                def ins(q):
                    i = np.rint(q).astype(int)
                    if np.any(i < 0) or np.any(i >= shp):
                        return False
                    return bool(occ[i[0], i[1], i[2]])

                if ins(e):
                    for k in range(1, 40):
                        q = e - 0.5 * k * srcdir
                        if not ins(q):
                            return np.linalg.norm(q + 0.5 * srcdir - origin)
                    return np.linalg.norm(e - origin)
                for k in range(40):
                    q = e + 0.5 * k * srcdir
                    i = np.rint(q).astype(int)
                    if np.any(i < 0) or np.any(i >= shp):
                        break
                    if ins(q):
                        return np.linalg.norm(q - origin)
                return np.linalg.norm(e - origin)
        tab = np.empty((self.nA, self.nB), dtype=np.float32)
        for i in range(self.nA):
            A = self.A0 + i * dA
            for j in range(self.nB):
                B = -180.0 + j * dB
                n = _n_of(A, B)
                sp = ILL.find_source_position(occ, -n, origin)
                tab[i, j] = (np.linalg.norm(np.asarray(sp, float) - origin) if mode == "source"
                             else _entry_r(sp, n))
        self.tab = torch.from_numpy(tab).to(dev)

    def exact(self, A_deg, B_deg):
        """The un-interpolated value, for verifying the table against the real march."""
        raise NotImplementedError                                # caller has occ/origin; see tests

    def __call__(self, A_deg, B_deg):
        """Bilinear lookup. A_deg/B_deg are torch scalars; result is differentiable in both."""
        fi = (A_deg - self.A0) / self.dA
        fi = fi.clamp(0, self.nA - 1 - 1e-6)
        fj = (B_deg + 180.0) / self.dB
        fj = torch.remainder(fj, float(self.nB))                 # periodic in B
        i0 = fi.floor(); j0 = fj.floor()
        ti = fi - i0; tj = fj - j0
        i0 = i0.long(); j0 = j0.long()
        i1 = (i0 + 1).clamp(max=self.nA - 1)
        j1 = torch.remainder(j0 + 1, self.nB)
        v00 = self.tab[i0, j0]; v01 = self.tab[i0, j1]
        v10 = self.tab[i1, j0]; v11 = self.tab[i1, j1]
        return ((1 - ti) * ((1 - tj) * v00 + tj * v01)
                + ti * ((1 - tj) * v10 + tj * v11))
