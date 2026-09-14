"""
Equivariant rigid augmentation (PART A) — free, no re-simulation.

Light transport is equivariant under a rigid motion applied JOINTLY to medium +
source: if you flip/rotate the optical volume AND the source AND the fluence field
together, the transformed (optical, Phi) pair is still an exact solution of the
transported problem.  So we can synthesize new valid scenes from existing ones
without running any solver.

We restrict to the 48 exact lattice symmetries of a cubic grid — the octahedral
group with reflections — represented as SIGNED AXIS PERMUTATIONS.  A transform is
(perm, sign): new spatial axis k is taken from old axis perm[k], negated if
sign[k] == -1.  This covers every axis flip and 90-degree rotation EXACTLY (pure
index remapping, no interpolation), so the augmented fluence is exact GT.

Conventions (consistent across volume / point / direction — verified by
tests/test_equivariance):
  - volume (X,Y,Z[,C]) spatial axes 0..2: transpose by perm, then flip negated axes.
  - point  p (voxel coords): new_p[k] = p[perm[k]] (then (N-1-.) if flipped).
  - vector d (e.g. srcdir):  new_d[k] = sign[k] * d[perm[k]]   (linear part only).

Medium-changing augmentations (mu_a jitter, added lesions) are intentionally NOT
here — they change the medium and require re-simulation (route them through
solvers.solve_fluence instead). Never pair a changed medium with an old Phi.
"""
from dataclasses import dataclass
from itertools import permutations, product
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class RigidTransform:
    """A signed axis permutation: an exact cubic-lattice symmetry."""
    perm: Tuple[int, int, int]
    sign: Tuple[int, int, int]

    @property
    def is_proper_rotation(self) -> bool:
        """True for proper rotations (det=+1), False for reflections (det=-1)."""
        # determinant = parity(perm) * prod(sign)
        p = self.perm
        parity = (np.sign((p[1] - p[0]) * (p[2] - p[0]) * (p[2] - p[1])))
        return parity * self.sign[0] * self.sign[1] * self.sign[2] > 0

    # -- apply to a spatial volume -------------------------------------------
    def apply_volume(self, vol: np.ndarray) -> np.ndarray:
        """vol: (X,Y,Z) or (X,Y,Z,C). Returns the rigidly-transformed volume."""
        nd = vol.ndim
        assert nd in (3, 4), f"expected (X,Y,Z[,C]), got {vol.shape}"
        axes = list(self.perm) + ([3] if nd == 4 else [])
        out = np.transpose(vol, axes)
        for k in range(3):
            if self.sign[k] < 0:
                out = np.flip(out, axis=k)
        return np.ascontiguousarray(out)

    # -- apply to a point (voxel coordinates) --------------------------------
    def apply_point(self, p, shape_old: Tuple[int, int, int]) -> np.ndarray:
        """p: (3,) voxel coords in the old frame. Returns new (3,) coords."""
        p = np.asarray(p, dtype=np.float64)
        out = np.empty(3, dtype=np.float64)
        for k in range(3):
            c = p[self.perm[k]]
            if self.sign[k] < 0:
                c = (shape_old[self.perm[k]] - 1) - c
            out[k] = c
        return out

    # -- apply to a direction (unit vector) ----------------------------------
    def apply_dir(self, d) -> np.ndarray:
        """d: (3,) direction. Linear part only (no offset)."""
        d = np.asarray(d, dtype=np.float64)
        return np.array([self.sign[k] * d[self.perm[k]] for k in range(3)],
                        dtype=np.float64)

    def new_shape(self, shape_old: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(int(shape_old[self.perm[k]]) for k in range(3))


def all_transforms(proper_only: bool = False) -> List[RigidTransform]:
    """The 48 signed axis permutations (24 proper rotations + 24 reflections)."""
    out = []
    for perm in permutations((0, 1, 2)):
        for sign in product((1, -1), repeat=3):
            t = RigidTransform(perm, sign)
            if proper_only and not t.is_proper_rotation:
                continue
            out.append(t)
    return out


IDENTITY = RigidTransform((0, 1, 2), (1, 1, 1))


def transform_scene(prop_vol: np.ndarray, fluence: np.ndarray,
                    srcpos, srcdir, t: RigidTransform):
    """
    Apply the SAME rigid transform jointly to the optical volume, the fluence,
    and the illumination — the operation that keeps (optical, Phi) physical.

    Args:
      prop_vol : (X,Y,Z,4) optical channels (physical units)
      fluence  : (X,Y,Z) fluence field (same grid)
      srcpos   : (3,) source voxel position;  srcdir: (3,) unit direction
    Returns (prop_vol_t, fluence_t, srcpos_t, srcdir_t).
    """
    assert prop_vol.shape[:3] == fluence.shape, \
        f"grid mismatch: {prop_vol.shape[:3]} vs {fluence.shape}"
    shape_old = prop_vol.shape[:3]
    prop_t = t.apply_volume(prop_vol)
    flu_t = t.apply_volume(fluence)
    pos_t = t.apply_point(srcpos, shape_old)
    dir_t = t.apply_dir(srcdir)
    return prop_t, flu_t, pos_t, dir_t


def sample_nonidentity(rng: np.random.Generator, k: int,
                       proper_only: bool = False) -> List[RigidTransform]:
    """Pick k distinct non-identity transforms."""
    pool = [t for t in all_transforms(proper_only) if t != IDENTITY]
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [pool[i] for i in idx]
