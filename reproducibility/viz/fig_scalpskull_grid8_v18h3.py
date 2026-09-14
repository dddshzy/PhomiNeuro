"""Plot scalp and skull layers for representative cohort participants.

One participant is selected near the scalp-plus-skull thickness median of each
``{AD, CU} x {Aβ+, Aβ-} x {M, F}`` cell. The display plane contains the measured
local inward normal, and the GRACE segmentation is overlaid on the corresponding
T1 image with a physically scaled caliper.
"""
import argparse, csv, os, sys
from collections import defaultdict

import numpy as np
import scipy.io as sio
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import Patch                                       # noqa: E402
from matplotlib.colors import to_rgb                                       # noqa: E402
import matplotlib.patheffects as pe                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data_expansion"))
import repro_config as RC                                                     # noqa: E402
import mni_normalize as MN                                                 # noqa: E402
import fig_skull_split as SK                                               # noqa: E402
import electrode_layer_thickness as ELT                                    # noqa: E402
import pet_to_mni as PET                                                   # noqa: E402
import generate_oasis as GO                                                # noqa: E402

THICK_CSV = str(RC.PUP_MNI_DIR / "cohort_electrode_thickness.csv")
STRAT_CSV = str(RC.OASIS_METADATA_DIR / "cohort_stratification.csv")
CACHE = str(RC.PUP_MNI_DIR)
MNI_DIR = str(RC.MNI_DATASET_DIR)
SRC_DIR = str(RC.OASIS_DATASET_DIR)
OUT = str(RC.RESULTS_DIR / "v18h3")
STEM = "scalpskull_grid8_v18h3"
CL_POS = 26.0

# Medical-imaging palette: dense bone brightest (as on CT), and no two ADJACENT layers in the same
# hue family. Outside in: skin -> fat -> muscle -> cortical -> diploe -> cortical -> CSF -> GM -> WM.
# Air is never painted, so the T1 background shows through unaltered.
# Bone uses a cool near-white tone and scalp layers use warmer mid-tones so their
# boundaries remain visible over a grayscale T1 image. Diploe uses a distinct hue.
COL = {"air": None,
       "WM": "#C9CBD1", "GM": "#8C9099", "CSF": "#3FA9F5", "vessel": "#C0392B",
       "fat": "#D9A404", "muscle": "#C97A68", "skin": "#E9967A", "skull": "#F7FBFD",
       "cortical bone": "#F7FBFD", "cancellous bone (diploe)": "#A86EDB",
       "bone, unresolved": "#35C46B", "eyes": "#59C7B8"}

# GRACE's own 11 classes -- the segmentation electrode_layer_thickness actually measures on.
GRACE_NAME = {1: "WM", 2: "GM", 3: "eyes", 4: "CSF", 5: "air", 6: "vessel",
              7: "cancellous bone (diploe)", 8: "cortical bone", 9: "skin", 10: "fat", 11: "muscle"}

CELLS = [("AD", "pos", "M"), ("AD", "pos", "F"), ("AD", "neg", "M"), ("AD", "neg", "F"),
         ("HC", "pos", "M"), ("HC", "pos", "F"), ("HC", "neg", "M"), ("HC", "neg", "F")]
DXLAB = {"AD": "AD", "HC": "CU"}
AMYLAB = {"pos": "Aβ+", "neg": "Aβ−"}


def t1_in_mni(sid):
    """The phantom's own T1 on the common MNI grid, via the same chain as SK.grace_in_mni."""
    p = os.path.join(CACHE, f"{sid}_t1mni_v18h3.npy")
    if os.path.isfile(p):
        return np.load(p)
    import SimpleITK as sitk
    iso, _ = GO.to_iso1mm(PET.find_t1(sid.upper()))
    ref_shape = sio.loadmat(os.path.join(
        SRC_DIR, f"{sid}_copmri_withHermiteF810.mat"))[SK.PROP_KEY].shape[:3]
    if iso.shape != ref_shape:
        raise SystemExit(f"[v18h3] {sid}: iso T1 {iso.shape} != pre-MNI phantom {ref_shape}")
    T = sitk.ReadTransform(os.path.join(MNI_DIR, f"{sid}_v11mni_rigid.tfm"))
    src = iso[::-1].copy() if MN.DATASET_FLIP[MN.dataset_of(sid)] else iso
    ref = sitk.Image(int(MN.GDIM[0]), int(MN.GDIM[1]), int(MN.GDIM[2]), sitk.sitkFloat32)
    ref.SetSpacing((1, 1, 1)); ref.SetOrigin(tuple(float(-x) for x in MN.AC))
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    res = sitk.Resample(MN._to_sitk(src.astype(np.float32), (0, 0, 0)), ref, T,
                        sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    out = sitk.GetArrayFromImage(res).transpose(2, 1, 0).astype(np.float32)
    np.save(p, out)
    return out


def cohort(site):
    """Per-head scalp+skull mm at `site`, joined to diagnosis / amyloid / sex."""
    th = {}
    for r in csv.DictReader(open(THICK_CSV)):
        if r["electrode"] != site:
            continue
        th[r["subject"].lower()] = (float(r["skin_mm"]) + float(r["muscle_mm"])
                                    + float(r["fat_mm"]), float(r["skull_mm"]), r["flag"].strip())
    out = defaultdict(list)
    for r in csv.DictReader(open(STRAT_CSV)):
        h = r["subject"].lower()
        if h not in th:
            continue
        try:
            cl = float(r["centiloid"])
        except (TypeError, ValueError):
            continue                                  # no Centiloid -> no amyloid class
        scalp, skull, flag = th[h]
        out[(r["group"], "pos" if cl >= CL_POS else "neg", r["sex"])].append(
            dict(head=h, scalp=scalp, skull=skull, total=scalp + skull, flag=flag))
    return out


def pick_median(members):
    """The head whose total is closest to the cell median -- deterministic for even n too."""
    v = np.array([m["total"] for m in members])
    return members[int(np.argmin(np.abs(v - float(np.median(v)))))]


def ray(head, site):
    """Rerun the thickness algorithm and return its ray, its sample indices and its layer sums."""
    g = SK.grace_in_mni(head)
    solid = (g > 0) & (g != ELT.AIR)
    grad = ELT.local_normal_field(solid)
    d_out = MN._load_mni()["dirs"][site]
    start = MN.snap_scalp(solid, d_out)
    m = ELT.march(g, start, -d_out, grad)
    labs = m["labs"]
    # the extracerebral column, defined exactly as march() defines it
    win = int(round(3.0 / ELT.STEP))
    brain = np.isin(labs, [ELT.WM, ELT.GM]).astype(int)
    col_end = len(labs)
    for i in range(len(labs) - win):
        if brain[i:i + win].mean() > 0.8:
            col_end = i
            break
    bone = np.where(np.isin(labs[:col_end], list(ELT.BONE)))[0]
    i0, i1 = (int(bone[0]), int(bone[-1])) if len(bone) else (0, 0)
    return dict(start=np.asarray(m["scalp"], float), n=np.asarray(m["normal"], float),
                i0=i0, i1=i1, scalp=m["skin"] + m["fat"] + m["muscle"], skull=m["skull"])


AXES = [(np.array([1.0, 0.0, 0.0]), "R", "L–R"),
        (np.array([0.0, 1.0, 0.0]), "A", "A–P"),
        (np.array([0.0, 0.0, 1.0]), "S", "S–I")]
PLANE_OF = {"L–R": "sagittal", "A–P": "coronal", "S–I": "axial"}


def basis(n, ref_dir):
    """In-plane horizontal for a ray direction n, conditioned for ANY electrode.

    cross(n, x_hat) degenerates for a lateral site such as T4, where n is nearly +-x_hat itself.
    Crossing against the global axis LEAST aligned with the ray is well conditioned everywhere and
    picks the natural frame at each site: x_hat for the midline electrodes (a sagittal plane) and
    z_hat at T4 (an axial one).

    The choice is made from the SITE's canonical outward direction, not from this head's own local
    normal. Using the local normal made the reference axis flip on individual heads -- at T4 one head
    landed on a near-CORONAL plane while its seven neighbours were near-axial, and the shared caption
    then described all eight as axial. v stays perpendicular to n for any reference, so the plane
    still contains the ray exactly; only the in-plane rotation is being pinned.
    """
    n = n / np.linalg.norm(n)
    ref = np.eye(3)[int(np.argmin(np.abs(np.asarray(ref_dir, float))))]
    v = np.cross(n, ref)
    return n, v / np.linalg.norm(v)


def orient(n, v):
    """(out-of-plane axis name, its tilt in deg, the two anatomical axes best inside the plane)."""
    m = np.cross(n, v); m /= np.linalg.norm(m)
    k = int(np.argmax(np.abs(m @ np.eye(3))))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(m[k])))))
    up = -n
    inplane = sorted(((float(np.hypot(e @ v, e @ up)), float(e @ v), float(e @ up), pos)
                      for e, pos, _ in AXES), reverse=True)[:2]
    return AXES[k][2], tilt, inplane


def ph_name(r):
    return {**{k: SK.P0ROW_NAME[k] for k in SK.P0ROW_NAME},
            SK.CANC: "cancellous", SK.CORT: "cortical", SK.UNRES: "bone?"}.get(r, r)


def plane(vol, start, n, v, ts, ss, order):
    """Sample `vol` on the plane {start + t*n + s*v}."""
    T, S = np.meshgrid(ts, ss, indexing="ij")
    P = start[None, None, :] + T[..., None] * n[None, None, :] + S[..., None] * v[None, None, :]
    return ndi.map_coordinates(vol, [P[..., 0], P[..., 1], P[..., 2]], order=order,
                               mode="constant", cval=0.0)


def tighten_columns(fig, axs, factor, wspace):
    """Cut the visible gap between columns by `factor`, measured on a real render.

    wspace alone cannot do it. Each panel is aspect="equal" and square, so the drawn image is only as
    wide as the axes are TALL; the rest of the axes box is blank. Measured on this figure the visible
    image-to-image gap was 0.49 in, of which wspace contributed 0.11 and that leftover padding 0.38 --
    halving wspace would have moved 11% of what the eye sees. The padding is removed by narrowing the
    canvas, which shrinks the axes box while the image width (set by the unchanged axes height) holds:

        gap = axes_w * (1 + wspace) - image_w

    is solved for the axes width that gives the target gap, then converted back to a figure width.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    row = axs[0]
    ims = [ax.images[0].get_window_extent(r) for ax in row]
    gap0 = float(np.mean([b.x0 - a.x1 for a, b in zip(ims, ims[1:])]))
    imgw = float(np.mean([b.width for b in ims]))
    ncol = len(row)
    aw = (factor * gap0 + imgw) / (1.0 + wspace)
    frac = fig.subplotpars.right - fig.subplotpars.left
    w_in = (ncol * aw + (ncol - 1) * wspace * aw) / frac / fig.dpi
    fig.set_size_inches(w_in, fig.get_size_inches()[1])
    fig.subplots_adjust(wspace=wspace)
    fig.canvas.draw()
    ims = [ax.images[0].get_window_extent(fig.canvas.get_renderer()) for ax in row]
    gap1 = float(np.mean([b.x0 - a.x1 for a, b in zip(ims, ims[1:])]))
    print(f"[v18h3] column gap {gap0 / fig.dpi:.3f} -> {gap1 / fig.dpi:.3f} in "
          f"(x{gap1 / gap0:.2f}); canvas width -> {w_in:.2f} in")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="Fz")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--stem", default=STEM)
    ap.add_argument("--alpha", type=float, default=0.50)
    ap.add_argument("--overlay", choices=["grace", "phantom"], default="grace")
    ap.add_argument("--cu-first", action="store_true", help="put the CU row on top")
    ap.add_argument("--col-gap", type=float, default=1.0,
                    help="scale the VISIBLE gap between columns (0.5 = halve it)")
    ap.add_argument("--half", type=float, default=18.0, help="mm either side of the ray")
    ap.add_argument("--up", type=float, default=6.0, help="mm outside the scalp")
    ap.add_argument("--res", type=float, default=0.5, help="mm per display pixel")
    ap.add_argument("--size", type=float, nargs=2, metavar=("W", "H"), default=[13.0, 7.0])
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    groups = cohort(a.site)
    missing = [c for c in CELLS if not groups.get(c)]
    if missing:
        raise SystemExit(f"[v18h3] empty cells at {a.site}: {missing}")
    chosen = {c: pick_median(groups[c]) for c in CELLS}
    cells = (CELLS[4:] + CELLS[:4]) if a.cu_first else CELLS      # which diagnosis leads the grid

    names = GRACE_NAME if a.overlay == "grace" else {
        **{r: SK.P0ROW_NAME[r] for r in SK.P0ROW_NAME},
        SK.CANC: "cancellous bone (diploe)", SK.CORT: "cortical bone",
        SK.UNRES: "bone, unresolved"}
    rgba = {r: (*to_rgb(COL[nm]), a.alpha) for r, nm in names.items() if COL.get(nm) is not None}

    down = 2 * a.half - a.up                               # square in mm
    ts = np.arange(down, -a.up - 1e-9, -a.res)             # index 0 = deepest -> origin="lower"
    ss = np.arange(-a.half, a.half + 1e-9, a.res)

    fig, axs = plt.subplots(2, 4, figsize=tuple(a.size))
    fig.subplots_adjust(left=0.006, right=0.994, top=0.945, bottom=0.115, wspace=0.035, hspace=0.115)
    seen, planes = set(), []

    for ax, c in zip(axs.flatten(), cells):
        m = chosen[c]
        R = ray(m["head"], a.site)
        n, v = basis(R["n"], MN._load_mni()["dirs"][a.site])
        axname, tilt, inplane = orient(n, v)
        planes.append((axname, tilt))

        lab = SK.grace_in_mni(m["head"]) if a.overlay == "grace" else SK.split_labels(m["head"])[0]
        G = plane(t1_in_mni(m["head"]), R["start"], n, v, ts, ss, order=1)
        L = plane(lab.astype(np.float32), R["start"], n, v, ts, ss, order=0).astype(np.int16)
        seen |= set(np.unique(L).tolist())

        lo, hi = np.percentile(G[G > 0], (1, 99)) if (G > 0).any() else (0.0, 1.0)
        # extent is (left, right, BOTTOM, TOP) and origin='lower' draws row 0 at the bottom, so the
        # bottom must carry ts[0] -- the deepest sample. Writing ts[-1] there inverts the vertical
        # coordinate against the pixels: the picture still looks right (scalp up) but t=0 lands near
        # the bottom, which put the caliper inside the brain instead of across the scalp and skull.
        ext = [ss[0], ss[-1], ts[0], ts[-1]]
        ax.imshow(G, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest",
                  extent=ext, aspect="equal")
        O = np.zeros(L.shape + (4,), np.float32)
        for r_, col in rgba.items():
            O[L == r_] = col
        ax.imshow(O, origin="lower", interpolation="nearest", extent=ext, aspect="equal")

        # Caliper AT the algorithm's own sample positions: t = 0 is index 0 (the voxel snap_scalp
        # returned) and t = i1*STEP is the last bone voxel of the extracerebral column. Because the
        # panel's vertical axis IS the marched ray, these are exact rather than projected. The span
        # can exceed the annotated sum by the voxels the algorithm does not count (air or soft
        # tissue interleaved inside the bone span); that residual is printed per head.
        # Sample i occupies [i*STEP - STEP/2, i*STEP + STEP/2], so the range the algorithm summed
        # runs from -STEP/2 to (i1 + 1/2)*STEP -- length (i1+1)*STEP. Using the sample CENTRES
        # instead makes the caliper exactly one step short of the number beside it (measured: a
        # uniform -0.25 mm on all eight heads, which is what exposed the off-by-one).
        t_out, t_in = -0.5 * ELT.STEP, (R["i1"] + 0.5) * ELT.STEP
        xc = -a.half * 0.42
        ax.annotate("", xy=(xc, t_out), xytext=(xc, t_in),
                    arrowprops=dict(arrowstyle="<|-|>", color="#FFE84D", lw=1.7,
                                    mutation_scale=10, shrinkA=0, shrinkB=0))
        for tt in (t_out, t_in):
            ax.plot([xc - 3.0, xc + 3.0], [tt, tt], color="#FFE84D", lw=1.7, solid_capstyle="butt")
        ax.text(xc + 4.0, (t_out + t_in) / 2, f"{m['total']:.1f} mm", ha="left", va="center",
                fontsize=14.0, fontweight="bold", color="#FFE84D",
                path_effects=[pe.withStroke(linewidth=3.0, foreground="#101010")])

        # Orientation compass. The panel axes are NOT anatomical: vertical is the electrode's own
        # inward normal, which at Fz points antero-superior (up = outward = (0, +0.56..0.69,
        # +0.70..0.83) over the eight heads), and horizontal is postero-superior. Labelling the frame
        # A/P/S/I would therefore be wrong. What IS true is that ONE anatomical axis is nearly
        # out-of-plane (orient() measures which and by how much), so only the other two need marking;
        # arrow is projected from that head's own basis, so it is exact per panel rather than shared.
        # Panel "up" is DECREASING t, hence the sign flip on the vertical component.
        o_s, o_t, L = ss[0] + 10.0, ts[0] - 5.5, 5.0
        for _, cs, ct, lab in inplane:
            ds, dt = cs * L, -ct * L               # panel "up" is DECREASING t, hence the sign flip
            ax.annotate("", xy=(o_s + ds, o_t + dt), xytext=(o_s, o_t),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.4, mutation_scale=8,
                                        shrinkA=0, shrinkB=0))
            ax.text(o_s + ds * 1.42, o_t + dt * 1.42, lab, ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color="white",
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="#101010")])

        ax.set_title(f"{DXLAB[c[0]]} {AMYLAB[c[1]]} {'Male' if c[2] == 'M' else 'Female'}",
                     fontsize=13.0, fontweight="bold", pad=4.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(ss[0], ss[-1]); ax.set_ylim(ts[0], ts[-1])
        for sp in ax.spines.values():
            sp.set_color("#9a9a9a"); sp.set_linewidth(0.8)
        # Hard check that the caliper's inner tick sits on bone in the segmentation being DRAWN.
        # It was a mismatch here that exposed the real issue: march() measures on raw GRACE, while
        # fig_skull_split's phantom labels give the phantom's own skull mask the final word on where
        # bone is. On oas31139 at T4 they disagree over 1.75 mm at the inner table, so a caliper
        # honouring the algorithm ended inside phantom CSF. Drawing GRACE removes the contradiction
        # instead of hiding it; the residual disagreement is still reported per head.
        gl = SK.grace_in_mni(m["head"])
        ph = SK.split_labels(m["head"])[0]
        tip = np.round(R["start"] + R["i1"] * ELT.STEP * n).astype(int)
        gtip, ptip = int(gl[tuple(tip)]), int(ph[tuple(tip)])
        if a.overlay == "grace" and gtip not in ELT.BONE:
            raise SystemExit(f"[v18h3] {m['head']}: caliper tip on GRACE label {gtip}, not bone")
        drawn = gtip if a.overlay == "grace" else ptip
        print(f"  {DXLAB[c[0]]:2s} {AMYLAB[c[1]]} {c[2]}  n={len(groups[c]):2d}  {m['head']}  "
              f"scalp {m['scalp']:.2f} + skull {m['skull']:.2f} = {m['total']:.2f} mm   "
              f"span {t_in - t_out:.2f} ({t_in - t_out - m['total']:+.2f})   "
              f"tip: GRACE={GRACE_NAME.get(gtip, gtip)} phantom={ph_name(ptip)}"
              f"{'  <-- masks differ' if (gtip in ELT.BONE) != (ptip in (SK.CANC, SK.CORT, SK.UNRES)) else ''}"
              f"{'   FLAG ' + m['flag'] if m['flag'] else ''}", flush=True)

    axset = {x[0] for x in planes}
    if len(axset) != 1:
        raise SystemExit(f"[v18h3] panels disagree on the out-of-plane axis: {sorted(axset)} -- the "
                         "caption cannot describe them as one plane family")

    # "near-sagittal/axial/coronal" is only claimed when the plane really is close to one. At F4 the
    # electrode normal has three comparable components, so the out-of-plane axis sits 31 deg off S-I
    # and calling it near-axial would oversell a plane that is simply oblique.
    tiltmax = max(x[1] for x in planes)
    plane_word = (f"near-{PLANE_OF[planes[0][0]]} oblique" if tiltmax <= 15.0 else "oblique")

    if a.col_gap != 1.0:
        tighten_columns(fig, axs, a.col_gap, 0.035 * a.col_gap)

    order = [i for i in sorted(seen) if i in rgba]
    fig.legend(handles=[Patch(facecolor=(*to_rgb(COL[names[i]]), 1.0), edgecolor="#606060",
                              linewidth=0.6, label=names[i]) for i in order],
               loc="lower center", bbox_to_anchor=(0.5, -0.006), ncol=len(order), fontsize=9.4,
               frameon=False, handlelength=1.5, handleheight=1.1, columnspacing=1.5,
               handletextpad=0.5)
    # Use two caption lines to keep the saved canvas close to the panel width.
    for y, s in ((0.076, f"segmentation at {a.alpha:.0%} opacity over the subject's own T1w   ·   "
                         f"{plane_word} plane containing the {a.site} inward normal "
                         f"(out-of-plane axis within {tiltmax:.1f}° of {planes[0][0]})"),
                 (0.046, "up = outward along that normal; white arrows give the two anatomical "
                         "directions best inside the plane"
                         "   ·   caliper spans the thickness algorithm's own sample range"
                         f"   ·   Aβ± = Centiloid ≥ / < {CL_POS:.0f}")):
        fig.text(0.5, y, s, ha="center", va="bottom", fontsize=8.8, color="#5a5a5a")

    for ext_ in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(a.outdir, f"{a.stem}.{ext_}"), dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(a.outdir, a.stem + ".png"))


if __name__ == "__main__":
    sys.exit(main())
