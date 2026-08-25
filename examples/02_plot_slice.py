#!/usr/bin/env python3
"""The published slice figure, surrogate panel only.

    python examples/02_plot_slice.py --head scb16 --electrode Cz --gate 1

Reproduces the LEFT ("Ours") panel of the published slice figure: a plane cut through the
mcx source along the beam, tissue map underneath, log10 fluence on top, one white contour per decade.
Figure grammar -- colours, contour levels, arrow, tissue key, axis convention -- is transcribed from
the production script so the two are comparable side by side.

Two things differ from the production script, both on purpose:

  * ONE PANEL, not two. The Monte-Carlo panel needs a 14 MB reference field per scene, and only
    scb16/Cz ships with one. `--with-mc` draws it where available.
  * THE COLOUR SCALE COMES FROM THE PREDICTION unless MC is drawn. The published figure keys the
    scale to the Monte-Carlo peak, which is unavailable for most demo scenes. The scale is printed
    with every run so a reader always knows which convention produced the picture.
"""
import argparse
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Patch                                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phomineuro import GATE_NS, ROOT, WEIGHTS, Predictor, Scene               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--head", default="scb16")
ap.add_argument("--electrode", default="Cz")
ap.add_argument("--gate", type=int, default=1, help="0-indexed; gate 1 = 0.3 ns")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--decades", type=float, default=6.0, help="colour span below the peak")
ap.add_argument("--step", type=float, default=0.5, help="in-plane sampling, mm")
ap.add_argument("--z-below-brain", type=float, default=30.0)
ap.add_argument("--with-mc", action="store_true", help="also draw the Monte-Carlo panel if shipped")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--cache", default=os.path.join(ROOT, "pyramid_cache"))
ap.add_argument("--vista3d", default=os.environ.get("VISTA3D_CKPT", ""))
ap.add_argument("--out", default=os.path.join(ROOT, "out"))
a = ap.parse_args()

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.linewidth": 0.9, "figure.dpi": 200,
})
# magma with the near-black bottom trimmed: the darkest decades stay legible against the tissue map
FLU_CMAP = LinearSegmentedColormap.from_list(
    "magma_deep", plt.get_cmap("magma")(np.linspace(0.20, 1.0, 256)))
TNAME = ["Scalp", "Skull", "CSF", "GM", "WM"]
TCOL = ["#D9C4B0", "#E4E0D6", "#BFD3D6", "#B9C4B0", "#D7CBBD"]
# (mu_a, mu_s) at 810 nm for scalp/skull/CSF/GM/WM -- read off optical_tissue_lut.tissue_lut("F810")
# labels 1-5, not estimated. Used only to COLOUR the background: each plane voxel is assigned the
# nearest of the five in standardised units. Getting these wrong does not fail, it just mislabels the
# tissue map (a first pass with plausible-looking values painted the scalp as grey matter).
REF = np.array([[0.0306, 11.6944],     # 1 skin
                [0.0110, 17.4545],     # 2 skull
                [0.0026,  0.0909],     # 3 CSF
                [0.0280,  7.3000],     # 4 GM
                [0.0920, 38.0000]])    # 5 WM
TSCALE = REF.std(0) + 1e-9

dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
pred = Predictor(os.path.join(WEIGHTS, f"phomineuro_s{a.seed}.pt"), dev)
scene = Scene(a.head, a.electrode, ROOT, dev, src_ref=pred.src_ref, cache_dir=a.cache,
              vista3d_ckpt=a.vista3d, weights_dir=os.path.join(WEIGHTS, "fm_encoder"))

# ---- the plane ---------------------------------------------------------------------------------
# ORIGIN IS THE PHYSICAL MCX SOURCE, not scene.srcpos. Under the entry convention scene.srcpos IS the
# entry point, and using it would slide the whole panel along the beam, breaking comparability with
# the published figure.
src = scene.mcx_srcpos.cpu().numpy().astype(float)
d = scene.srcdir.cpu().numpy().astype(float)
d /= np.linalg.norm(d) + 1e-9
e1 = np.array([d[0], d[1], 0.0])                        # horizontal part of the beam
e1 = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(e1) < 1e-6 else e1 / np.linalg.norm(e1)
e2 = np.array([0.0, 0.0, 1.0])                          # plane is parallel to z

prop = scene.prop_phys                                  # (4,X,Y,Z)
dims = np.array(prop.shape[1:])
R = 150.0
ss = np.arange(-R, R + 1e-6, a.step)
ww = np.arange(-R, R + 1e-6, a.step)
S, W = np.meshgrid(ss, ww, indexing="xy")
pts = src[None, None, :] + S[..., None] * e1[None, None, :] + W[..., None] * e2[None, None, :]
vox = np.round(pts).astype(int)
inside = np.all((vox >= 0) & (vox < dims[None, None, :]), axis=-1)
vc = np.clip(vox, 0, dims - 1)
opt = prop[:2, vc[..., 0], vc[..., 1], vc[..., 2]]      # (2,H,W)
tissue_mask = inside & (opt[1] > 1e-6)
cls = np.argmin((((opt.transpose(1, 2, 0)[..., None, :] - REF[None, None]) / TSCALE) ** 2
                 ).sum(-1), axis=-1)

rows = np.where(tissue_mask.any(1))[0]
cols = np.where(tissue_mask.any(0))[0]
m = int(round(6.0 / a.step))
r0, r1 = max(rows[0] - m, 0), min(rows[-1] + m + 1, tissue_mask.shape[0])
c0, c1 = max(cols[0] - m, 0), min(cols[-1] + m + 1, tissue_mask.shape[1])
z0_ref = ww[r0]                                          # z = 0 always means the bottom of the head
if a.z_below_brain is not None:
    # Crop off neck and face, which the beam never reaches. The brain mask must NOT come from `cls`:
    # that is a nearest-neighbour match on (mu_a, mu_s), and neck/temporalis muscle lands on the GM
    # reference, so "lowest GM row" would be the bottom of the neck. Without the FreeSurfer parcel-
    # lation the production script uses, this falls back to the same approximation -- and says so.
    brain = tissue_mask & np.isin(cls, (3, 4))
    br = np.where(brain.any(1))[0]
    if br.size:
        r0 = max(r0, int(br[0] - round(a.z_below_brain / a.step)))
        print(f"[plot] z-crop by cls(GM/WM) -- muscle may be misread as GM")
sl = (slice(r0, r1), slice(c0, c1))
ext = [ss[c0], ss[c1 - 1], ww[r0], ww[r1 - 1]]

t_entry = 0.0
for tt in np.arange(0.0, 80.0, 0.25):
    pv = np.clip(np.round(src + tt * d).astype(int), 0, dims - 1)
    if prop[1, pv[0], pv[1], pv[2]] > 1e-6:
        t_entry = float(tt)
        break

# ---- predict on the plane ----------------------------------------------------------------------
q = torch.tensor(pts[tissue_mask], dtype=torch.float32, device=dev)
P = np.full(tissue_mask.shape, np.nan)
P[tissue_mask] = pred.at_points(scene, q, t_ns=GATE_NS[a.gate])[:, 0].cpu().numpy()

G = None
mc = scene.mc_reference(ROOT) if a.with_mc else None
if a.with_mc and mc is None:
    print(f"[plot] no Monte-Carlo reference shipped for {a.head}/{a.electrode}; drawing one panel")
if mc is not None:
    g = mc[vc[..., 0], vc[..., 1], vc[..., 2], a.gate].astype(float)
    G = np.where(tissue_mask & (g > 0), np.log10(np.clip(g, 1e-300, None)), np.nan)

pk = float(np.nanmax(G)) if G is not None else float(np.nanmax(P))
floor = pk - a.decades
print(f"[plot] colour scale [{floor:.2f}, {pk:.2f}] keyed to "
      f"{'the Monte-Carlo peak' if G is not None else 'the PREDICTION peak'}")
panels = [(P, "Ours")] + ([(G, "MC (GT)")] if G is not None else [])
P = np.where(P >= floor, P, np.nan)
if G is not None:
    G = np.where(G >= floor, G, np.nan)
    panels = [(P, "Ours"), (G, "MC (GT)")]
else:
    panels = [(P, "Ours")]

# ---- draw ---------------------------------------------------------------------------------------
aspect = (ext[1] - ext[0]) / (ext[3] - ext[2])
h_panel = 3.3
lpad, rpad = 0.62, 1.95
panels_w = len(panels) * aspect * h_panel
fig_w = panels_w + lpad + rpad
fig = plt.figure(figsize=(fig_w, h_panel + 0.60))
gs = fig.add_gridspec(1, len(panels), wspace=0.10, left=lpad / fig_w,
                      right=(lpad + panels_w) / fig_w, top=0.94, bottom=0.145)
tcmap = ListedColormap(TCOL)
tnorm = BoundaryNorm(np.arange(-0.5, 5.1, 1), tcmap.N)
tshow = np.where(tissue_mask, cls, np.nan)

axes, im = [], None
for i, (F, title) in enumerate(panels):
    ax = fig.add_subplot(gs[i], sharex=axes[0] if axes else None,
                         sharey=axes[0] if axes else None)
    axes.append(ax)
    ax.imshow(tshow[sl], origin="lower", extent=ext, cmap=tcmap, norm=tnorm, interpolation="nearest")
    im = ax.imshow(F[sl], origin="lower", extent=ext, cmap=FLU_CMAP, vmin=floor, vmax=pk,
                   interpolation="bilinear")
    lv = np.arange(np.ceil(floor), pk + 1e-9, 1.0)        # one contour per decade
    with np.errstate(invalid="ignore"):
        ax.contour(np.linspace(ext[0], ext[1], F[sl].shape[1]),
                   np.linspace(ext[2], ext[3], F[sl].shape[0]),
                   np.nan_to_num(F[sl], nan=floor - 1), levels=lv,
                   colors="white", linewidths=0.45, alpha=0.55)
    din = np.array([np.dot(d, e1), np.dot(d, e2)]) * (t_entry + 3.0)
    ax.annotate("", xy=(din[0], din[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#D1483F", lw=2.0,
                                mutation_scale=13, shrinkA=0, shrinkB=0), zorder=6)
    ax.plot(0, 0, marker="o", ms=5.5, mfc="#D1483F", mec="white", mew=1.1, zorder=7)
    ax.set_title(title, loc="left", fontsize=11, color="#3A3A38", pad=4)
    ax.set_xlabel("Along beam (mm)")
    ax.set_aspect("equal")
    for s_ in ax.spines.values():
        s_.set_color("#9A9A98")
if len(axes) > 1:
    axes[1].tick_params(labelleft=False)

zspan = ext[3] - ext[2]
zstep = 50.0 if zspan > 120 else 25.0
k0 = np.ceil((ext[2] - z0_ref) / zstep - 1e-9)
zt = z0_ref + np.arange(k0, (ext[3] - z0_ref) / zstep + 1e-9) * zstep
axes[0].set_yticks(zt)
axes[0].set_yticklabels([f"{v - z0_ref:.0f}" for v in zt])
axes[0].set_ylabel("z (mm)")

cax = fig.add_axes([(lpad + panels_w + 0.16) / fig_w, 0.145, 0.16 / fig_w, 0.94 - 0.145])
cb = fig.colorbar(im, cax=cax)
cb.set_label("log$_{10}$ fluence (a.u.)")
fig.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(TCOL, TNAME)],
           loc="upper right", frameon=False, bbox_to_anchor=(0.999, 0.94), handlelength=1.2)

os.makedirs(a.out, exist_ok=True)
stem = f"slice_{a.head}_{a.electrode}_g{a.gate}_s{a.seed}"
for e in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"{stem}.{e}"), dpi=600, bbox_inches="tight")
print(f"[plot] wrote {os.path.join(a.out, stem)}.png / .pdf   "
      f"(gate {a.gate}, t = {GATE_NS[a.gate]} ns)")
