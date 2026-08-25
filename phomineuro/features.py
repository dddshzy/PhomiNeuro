"""Input-feature assembly.

Everything here is transcribed from the production tree and each function names its origin. The
model is a function of these features; getting one of them subtly wrong does not raise, it just
degrades the prediction silently. Two have bitten this project before and carry the scar:

  * `path_features(..., trilinear=)` -- a model trained with trilinear ray-march sampling but
    evaluated with nearest-voxel gathers is a train/eval mismatch. Always pass cfg["path_tri"].
  * the entry-point convention -- under src_ref="entry" every source-dependent feature references
    the beam's entry point on the scalp, NOT the mcx source, which sits 15 voxels outside it.
"""
import math
import torch
import torch.nn.functional as F

R_SCALE = 128.0          # train_inr_v11.py:39
NS = 96                  # make_pathfeat.py:37 -- ray-march samples per source->point segment
SRC_R_MARGIN = 15        # illumination.py -- the mcx source stand-off, in voxels


# --- train_inr_v3.py:64 -------------------------------------------------------------------------
def xyz_to_norm(xyz, vs):
    return torch.stack([(xyz[:, 0] / (vs[0] - 1)) * 2 - 1,
                        (xyz[:, 1] / (vs[1] - 1)) * 2 - 1,
                        (xyz[:, 2] / (vs[2] - 1)) * 2 - 1], dim=-1)


# --- train_inr_v3.py:70 -------------------------------------------------------------------------
def sample_raw(xyz_norm, pyramid):
    """Trilinearly sample the 5-scale FM pyramid at normalised coords -> (N, 1488)."""
    grid = torch.stack([xyz_norm[:, 2], xyz_norm[:, 1], xyz_norm[:, 0]], -1).view(1, 1, 1, -1, 3)
    feats = [F.grid_sample(fg, grid, mode="bilinear", align_corners=True
                           ).squeeze(0).squeeze(1).squeeze(1).permute(1, 0) for fg in pyramid]
    return torch.cat(feats, -1)


# --- inr_dataset.py:197,211 ---------------------------------------------------------------------
def sample_volume(vol, xyz_phys, vol_shape, mode="bilinear"):
    """Trilinearly sample a (1,C,X,Y,Z) volume at (N,3) voxel coords -> (N,C)."""
    D, H, W = vol_shape
    xn = (xyz_phys[:, 0] / (D - 1)) * 2 - 1
    yn = (xyz_phys[:, 1] / (H - 1)) * 2 - 1
    zn = (xyz_phys[:, 2] / (W - 1)) * 2 - 1
    grid = torch.stack([zn, yn, xn], dim=-1).view(1, 1, 1, -1, 3)
    out = F.grid_sample(vol, grid, mode=mode, align_corners=True)
    return out.squeeze(0).squeeze(1).squeeze(1).permute(1, 0)


# --- train_inr_v11.py:386 -----------------------------------------------------------------------
def src_features(xyz, srcpos, srcdir):
    u = xyz - srcpos.view(1, 3)
    r = u.norm(dim=1, keepdim=True).clamp_min(1e-3)
    cos = (u * srcdir.view(1, 3)).sum(1, keepdim=True) / r
    return torch.cat([r / R_SCALE, cos], dim=1)


# --- make_pathfeat.py:43 ------------------------------------------------------------------------
def _sample_prop(pts, prop, vs, trilinear):
    if not trilinear:
        idx = pts.round().long()
        for d in range(3):
            idx[..., d].clamp_(0, int(vs[d]) - 1)
        return prop[0][:, idx[..., 0], idx[..., 1], idx[..., 2]]
    n, NSp = pts.shape[0], pts.shape[1]
    g = 2.0 * pts / (vs.float().view(1, 1, 3) - 1.0) - 1.0        # voxel -> [-1,1]
    grid = g.flip(-1).view(1, n, NSp, 1, 3)                       # grid_sample expects (z,y,x)
    return F.grid_sample(prop, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)[0, :, :, :, 0]


# --- make_pathfeat.py:71 ------------------------------------------------------------------------
def path_features(xyz, srcpos, prop, vs, nseg=0, trilinear=False):
    """Ray-march srcpos -> xyz. Returns (n, 6 + nseg): tau_a, tau_sp, tau_eff, L_csf, L_skull, r."""
    n = xyz.shape[0]
    dev = xyz.device
    s = torch.linspace(0.0, 1.0, NS, device=dev).view(1, NS, 1)
    pts = srcpos.view(1, 1, 3) + s * (xyz.view(n, 1, 3) - srcpos.view(1, 1, 3))
    r = (xyz - srcpos.view(1, 3)).norm(dim=1)
    ds = (r / (NS - 1)).view(n, 1)

    P = _sample_prop(pts, prop, vs, trilinear)                   # (4,n,NS)
    mua, mus, g = P[0], P[1], P[2]
    musp = mus * (1.0 - g)
    mueff = torch.sqrt(torch.clamp(3.0 * mua * (mua + musp), min=1e-12))

    is_csf = ((mua < 0.005) & (musp < 0.5)).float()
    is_skull = (musp > 1.2).float()
    cols = [
        (mua * ds).sum(1),
        (musp * ds).sum(1) * 0.1,          # musp' is ~10x larger; keep columns comparable
        (mueff * ds).sum(1),
        (is_csf * ds).sum(1) * 0.1,
        (is_skull * ds).sum(1) * 0.1,
        r * 0.01,
    ]
    if nseg:
        assert NS % nseg == 0, f"NS={NS} must be divisible by nseg={nseg}"
        prof = mueff.view(n, nseg, NS // nseg).mean(2)           # (n,nseg), source -> point
        cols += list(prof.unbind(1))
    return torch.stack(cols, 1)


# --- make_entryfeat.py --------------------------------------------------------------------------
def entry_of(srcpos, srcdir, occ=None, vs=None, snap=True, max_step=40):
    """The beam's entry point on the scalp.

    `entry = srcpos + SRC_R_MARGIN * srcdir` is exact only when the source sits at the nominal
    stand-off; where it does not, the point is snapped to the first tissue voxel in 0.5-voxel steps.
    The walk goes BOTH ways -- advancing only would drift by |dr| - 0.5 whenever the source is
    closer than nominal and `entry` therefore already lies inside tissue.
    """
    e = srcpos + SRC_R_MARGIN * srcdir
    if not snap or occ is None:
        return e

    def inside(p):
        i = p.round().long()
        if not bool((i >= 0).all() and (i < vs.long()).all()):
            return False
        return bool(occ[int(i[0]), int(i[1]), int(i[2])])

    if inside(e):                       # started in tissue -> back out
        for k in range(1, max_step):
            p = e - 0.5 * k * srcdir
            if not inside(p):
                return p + 0.5 * srcdir
    for k in range(max_step):           # started in air -> advance
        p = e + 0.5 * k * srcdir
        if inside(p):
            return p
    return e


def light_vector(srcpos, srcdir, vol_shape):
    """The 10-element illumination vector: srcpos_norm(3) + srcdir(3) + sin/cos of elevation and
    azimuth(4). train_inr_v11.py:434.

    Note the redundancy, which is exact and provable: with outward normal n = -srcdir,
    A = asin(n_z) and B = atan2(n_x, n_y), so srcdir = -[cosA sinB, cosA cosB, sinA] and, since
    A is in [-pi/2, pi/2] and hence cosA >= 0, the inversion is unique. Columns 3..9 carry 2
    degrees of freedom in 7 numbers. hero3 drops 6:10 for exactly this reason.
    """
    X, Y, Z = [int(v) for v in vol_shape]
    srcpos_norm = srcpos / (torch.tensor([X, Y, Z], dtype=torch.float32, device=srcpos.device) - 1)
    nrm = -srcdir
    A = math.asin(float(torch.clamp(nrm[2], -1, 1)))
    B = math.atan2(float(nrm[0]), float(nrm[1]))
    ang = torch.tensor([math.sin(A), math.cos(A), math.sin(B), math.cos(B)],
                       device=srcpos.device, dtype=srcpos.dtype)
    return torch.cat([srcpos_norm, srcdir, ang])


# gate index -> time code in [-1,1]. train_inr_v11.py:486, with N_STEP = 10.
N_STEP = 10
T_ENC = (2.0 * (torch.arange(N_STEP).float() + 0.5) / N_STEP - 1.0)
GATE_NS = tuple(round(0.2 * (k + 0.5), 2) for k in range(N_STEP))   # gate centres, nanoseconds


def t_code(t_ns):
    """Nanoseconds -> the model's time code. Verified against T_ENC: t_code = t_ns - 1.

    The model is continuous in t, so this accepts values BETWEEN gate centres -- something the
    grid-output baselines cannot do at all. Interpolated values carry a real cost, largest near
    t = 0.4-0.6 ns; treat off-gate queries as an extrapolation of a 10-sample curve, not as free.
    """
    return float(t_ns) - 1.0
