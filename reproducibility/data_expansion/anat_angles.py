"""
v2 light-source angle geometry (anatomical, frame-standardized).

After orientation standardization (see standardize_orientation.py), every head
lives in ONE canonical anatomical frame:

    +X = right (left<->right temporal axis)   (sign is don't-care: B sweep is symmetric)
    +Y = anterior (face / "forward")
    +Z = superior (toward vertex)

Two illumination angles (user spec):
  A (elevation)  in [45, 90] deg : 0 deg = straight forward (in the horizontal
        plane), lifting toward the vertex; A=90 -> source straight above (vertex).
  B (azimuth)    in [-90, 90] deg: 0 = forward, -90 = left temporal, +90 = right
        temporal (orientation within the horizontal plane).

The source sits on the scalp at outward normal n (head-center -> source); the
illumination direction (into the head) is srcdir = -n:

    horiz = cos(B) * f + sin(B) * r          # forward direction rotated in horiz plane
    n     = cos(A) * horiz + sin(A) * u      # lift from horizontal toward vertex
    srcdir = -n / |n|

These (A, B) are frame-independent anatomical labels, identical across all heads,
which is exactly the "standardized light-source parameter" the task requires.
"""
import numpy as np

# Canonical anatomical unit axes (voxel-index space, post-standardization).
F_ANT = np.array([0.0, 1.0, 0.0])   # anterior / forward
U_SUP = np.array([0.0, 0.0, 1.0])   # superior / up
R_RIGHT = np.array([1.0, 0.0, 0.0]) # right

# Formal v2 grid (10 deg steps).
A_GRID = [45.0, 55.0, 65.0, 75.0, 85.0, 90.0]      # elevation
B_GRID = [float(b) for b in range(-90, 91, 10)]     # azimuth (19 values)


def angles_to_srcdir(A_deg, B_deg,
                     f=F_ANT, u=U_SUP, r=R_RIGHT):
    """Return (srcdir, n_out): inward illumination unit vector and the outward
    scalp normal (head-center -> source), both unit length, in voxel-index axes."""
    A = np.radians(A_deg)
    B = np.radians(B_deg)
    horiz = np.cos(B) * f + np.sin(B) * r
    n = np.cos(A) * horiz + np.sin(A) * u
    n = n / np.linalg.norm(n)
    return (-n).astype(np.float64), n.astype(np.float64)


def angle_grid(a_grid=A_GRID, b_grid=B_GRID):
    """List of (A, B) pairs. At A=90 (vertex) B is degenerate -> keep only B=0."""
    out = []
    for a in a_grid:
        if a >= 90.0 - 1e-6:
            out.append((float(a), 0.0))
        else:
            for b in b_grid:
                out.append((float(a), float(b)))
    return out


def angle_tag(A_deg, B_deg):
    """Filename-safe tag, e.g. A45_Bm30 / A75_Bp00 (m=minus, p=plus)."""
    def fmt(v):
        s = "m" if v < 0 else "p"
        return f"{s}{abs(int(round(v))):02d}"
    return f"A{int(round(A_deg)):02d}_B{fmt(B_deg)}"


if __name__ == "__main__":
    g = angle_grid()
    print(f"angle grid: {len(g)} (A,B) pairs")
    for a, b in g[:3] + g[-3:]:
        sd, n = angles_to_srcdir(a, b)
        print(f"  A={a:5.1f} B={b:6.1f} {angle_tag(a,b):12s} srcdir={np.round(sd,3).tolist()}")
