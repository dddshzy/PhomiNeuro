"""Generate V18 feature-correlation and parenchymal-deposition figures.

The analysis reads the V18 CSF-split and combined deposition tables and reuses
the common statistical and plotting routines in ``fig_v17_update0801_0803``.
"""
import argparse, csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPRO_ROOT)
sys.path.insert(0, HERE)
import repro_config as RC                                               # noqa: E402
os.environ.setdefault("MPLCONFIGDIR", str(RC.RESULTS_DIR / "v18h3" / "_mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import fig_v17_update0801_0803 as M                                   # noqa: E402

ROOT = str(RC.WORK_DIR)
PUP = str(RC.PUP_MNI_DIR)
DEP = os.path.join(PUP, "cohort_energy_deposition_v18h3.csv")
DEP_CSF = os.path.join(PUP, "cohort_energy_deposition_csf_v18h3.csv")
OUT = str(RC.RESULTS_DIR / "v18h3")


DROP = object()          # returned by a patch_text fn to suppress the call entirely


class _null:
    """No-op context, so an optional patch can be written as one `with` rather than two branches."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class patch_text:
    """Rewrite drawn strings without touching the V17 script and without copying its routines.

    The control group has to read CU (cognitively unimpaired), but "HC" is not just a label: in
    _pa_box it is also the internal group KEY (GRP, grp(), the four post-hoc brackets), so remapping
    the key would hand mannwhitneyu two empty samples. Only the rendered text is intercepted.

    Both Axes.text and Figure.text are wrapped, because the two figures use different ones --
    _pa_box draws the group name with ax.text (fig_v17_update0801_0803.py:469) while the Mantel
    cohort title uses fig.text (:629).

    ``fn(s)`` returns a replacement or ``None``; the expected hit count is asserted.
    """

    def __init__(self, fn, expect, what="text"):
        self.fn, self.expect, self.what, self.n, self.seen = fn, expect, what, 0, []

    def _wrap(self, orig):
        outer = self

        def text(obj, *args, **kw):
            s = args[2] if len(args) >= 3 else kw.get("s")
            if isinstance(s, str):
                new = outer.fn(s)
                if new is DROP:
                    outer.n += 1
                    outer.seen.append((s, "<dropped>"))
                    return None          # every ax.text return value in _pa_box is discarded
                if new is not None and new != s:
                    outer.n += 1
                    outer.seen.append((s, new))
                    if len(args) >= 3:
                        args = (args[0], args[1], new) + args[3:]
                    else:
                        kw = dict(kw, s=new)
            return orig(obj, *args, **kw)

        return text

    def __enter__(self):
        import matplotlib.axes, matplotlib.figure
        self.oa, self.of = matplotlib.axes.Axes.text, matplotlib.figure.Figure.text
        matplotlib.axes.Axes.text = self._wrap(self.oa)
        matplotlib.figure.Figure.text = self._wrap(self.of)
        return self

    def __exit__(self, *exc):
        import matplotlib.axes, matplotlib.figure
        matplotlib.axes.Axes.text, matplotlib.figure.Figure.text = self.oa, self.of
        if exc[0] is None:
            if self.n != self.expect:
                raise SystemExit(f"[v18h3] {self.what} rewrite fired {self.n}x, expected "
                                 f"{self.expect} -- the label moved. Saw: {self.seen}")
            print(f"[v18h3] {self.what} rewrite applied {self.n}x  e.g. "
                  f"{self.seen[0][0]!r} -> {self.seen[0][1]!r}")
        return False


def relabel_group():
    """AD / HC group names under the boxes: exactly six, one per panel of parenchyma_anova."""
    return patch_text(lambda s: "CU" if s == "HC" else None, 6, what="group label HC->CU")


def relabel_cohort_title():
    """The Mantel HC-only cohort title, composed as f'{gt}  (n = {len(heads)})' at :629, so it is a
    prefix match rather than an equality one."""
    return patch_text(lambda s: "Cognitively unimpaired" + s[len("Healthy controls"):]
                      if s.startswith("Healthy controls") else None, 1, what="cohort title")


class compact_group_axis:
    """Collapse _pa_box's three-deck x-axis into one row of merged labels.

    As drawn by the V17 routine each panel carries: the tick labels (Male / Female / Male / Female),
    a bold group name below them (AD / HC, :469) and a sample-size caption below that
    ("(n=42, M:F=23:19)", :471). That is three decks for two factors. This replaces the tick labels
    with merged ones (AD-M / AD-F / CU-M / CU-F) and drops the other two decks, which also removes
    the last place the group name was spelled -- so the HC->CU rewrite is no longer needed alongside.

    Every count is asserted per panel: 4 dropped strings (2 group names + 2 captions) and 1 tick-label
    substitution. The V17 routine is not edited; only what it draws is intercepted.
    """

    def __init__(self, xlabels, n_panels):
        self.xlabels, self.n = list(xlabels), n_panels
        drop = {"AD", "HC", "CU"}
        self.txt = patch_text(lambda s: DROP if (s in drop or s.startswith("(n=")) else None,
                              4 * n_panels, what="x-axis deck")
        self.hits = 0

    def __enter__(self):
        import matplotlib.axes
        self.txt.__enter__()
        self.orig = matplotlib.axes.Axes.set_xticklabels
        outer, lab = self, self.xlabels

        def set_xticklabels(ax, labels, *args, **kw):
            labels = list(labels)
            if len(labels) == len(lab):
                outer.hits += 1
                labels = lab
            return outer.orig(ax, labels, *args, **kw)

        matplotlib.axes.Axes.set_xticklabels = set_xticklabels
        return self

    def __exit__(self, *exc):
        import matplotlib.axes
        matplotlib.axes.Axes.set_xticklabels = self.orig
        self.txt.__exit__(*exc)
        if exc[0] is None:
            if self.hits != self.n:
                raise SystemExit(f"[v18h3] tick-label merge fired {self.hits}x, expected {self.n}")
            print(f"[v18h3] x-axis compacted on {self.hits} panels -> {self.xlabels}")
        return False


class with_amyloid_feature:
    """Add a binary amyloid-positivity variable to the Mantel figure's feature set.

    _load_env_tissue already carries Centiloid as a continuous feature, so the binary one is derived
    from that array rather than read again -- same heads, same order, same NaN pattern (the 12 heads
    with no Centiloid stay NaN and are dropped per-test by the existing mask, so this row is computed
    on 72 heads while the others use 84).

    REGISTER THE OBVIOUS CAVEAT: A-beta+ is a dichotomisation of Centiloid, so the Centiloid x
    amyloid+ cell of the triangle is collinear BY CONSTRUCTION and is not an independent finding.
    What the row is actually for is the other cells -- whether a binary clinical read of amyloid
    tracks the anatomy and the tissue-level delivery profiles the way the continuous scale does.
    """

    def __init__(self, cl_pos=26.0, label="Aβ+", after="centiloid"):
        self.cl_pos, self.label, self.after = cl_pos, label, after
        self.n = 0

    def __enter__(self):
        import numpy as np
        orig, outer = M._load_env_tissue, self

        def patched(group=None):
            FEATS, heads, env, tis = orig(group)
            keys = [k for k, _ in FEATS]
            if "centiloid" not in keys:
                raise SystemExit("[v18h3] _load_env_tissue no longer exposes 'centiloid'")
            cl = np.asarray(env["centiloid"], float)
            env["apos"] = np.where(np.isnan(cl), np.nan, (cl >= outer.cl_pos).astype(float))
            i = keys.index(outer.after) + 1
            outer.n += 1
            return FEATS[:i] + [("apos", outer.label)] + FEATS[i:], heads, env, tis

        self.orig = orig
        M._load_env_tissue = patched
        return self

    def __exit__(self, *exc):
        M._load_env_tissue = self.orig
        if exc[0] is None:
            print(f"[v18h3] {self.label} (Centiloid >= {self.cl_pos:.0f}) added to "
                  f"{self.n} feature sets")
        return False


class swap_pa_panels:
    """Exchange two panels of fig_parenchyma_anova without editing the V17 routine.

    Its panel list (PAN) is a local inside the function, so it cannot be rebound from outside. The
    panels are drawn in list order by successive _pa_box(ax, rows, key, ylab, bg) calls onto axes
    taken in order, so exchanging the (key, ylab) pair carried by two of those calls exchanges the
    two panels' positions exactly. Both members of the pair here share WARM_BG, so the row tint is
    unaffected; a swap across rows would also have to carry bg.

    The number of substitutions is asserted, so a renamed key fails loudly instead of silently
    leaving the figure in its old order.
    """

    def __init__(self, pairs):
        self.map = {}
        for (ka, la), (kb, lb) in pairs:
            self.map[ka] = (kb, lb)
            self.map[kb] = (ka, la)
        self.n = 0

    def __enter__(self):
        orig, outer = M._pa_box, self

        def pa_box(ax, rows, key, ylab, bg):
            if key in outer.map:
                key, ylab = outer.map[key]
                outer.n += 1
            return orig(ax, rows, key, ylab, bg)

        self.orig = orig
        M._pa_box = pa_box
        return self

    def __exit__(self, *exc):
        M._pa_box = self.orig
        if exc[0] is None:
            if self.n != len(self.map):
                raise SystemExit(f"[v18h3] panel swap fired {self.n}x, expected {len(self.map)} -- "
                                 f"a panel key changed")
            print(f"[v18h3] panels swapped: {' <-> '.join(sorted(set(self.map)))}")
        return False


def fit_xticklabels(fig, axes, start=12.0, floor=6.5, pad_frac=0.30):
    """Shrink the merged tick labels until no two of them touch, by MEASURING them.

    _pa_box's box positions (0, 0.584, 1.333, 1.917) were chosen for "Male"/"Female"; the merged
    two-factor labels are longer and can collide at the inherited 12 pt. The clearance depends on the
    font, the figure width and the label text, so it is measured on a real render rather than
    estimated: draw, take each label's window extent, and step the size down until the smallest
    horizontal gap within every panel exceeds pad_frac of the label height.
    """
    import numpy as np
    fs = start
    while fs >= floor:
        for ax in axes:
            for t in ax.get_xticklabels():
                t.set_fontsize(fs)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        worst = np.inf
        for ax in axes:
            bb = sorted((t.get_window_extent(r) for t in ax.get_xticklabels()), key=lambda b: b.x0)
            for a, b in zip(bb, bb[1:]):
                worst = min(worst, (b.x0 - a.x1) / max(a.height, 1e-9))
        if worst >= pad_frac:
            print(f"[v18h3] tick labels fit at {fs:.1f} pt (clearance {worst:.2f} x label height)")
            return fs
        fs -= 0.5
    print(f"[v18h3] tick labels forced to the {floor:.1f} pt floor -- still tight")
    return floor


def fit_ylabels(fig, axes, start=13.5, floor=7.0, frac=0.98):
    """Shrink the y-axis labels until each fits inside its own panel height.

    A y label is rotated, so its length is measured against the AXES HEIGHT, and shrinking a figure
    vertically is exactly what makes it overflow: point sizes do not shrink with the canvas. Left
    overflowing, bbox_inches="tight" would silently grow the saved image back out -- the label would
    stick out past the panel it belongs to instead of the figure being compressed. Measured, not
    estimated, for the same reason as the tick labels.
    """
    fs = start
    while fs >= floor:
        for ax in axes:
            ax.yaxis.label.set_fontsize(fs)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        ok = True
        for ax in axes:
            h = ax.get_window_extent(r).height
            if ax.yaxis.label.get_window_extent(r).height > frac * h:
                ok = False
                break
        if ok:
            print(f"[v18h3] y labels fit at {fs:.1f} pt")
            return fs
        fs -= 0.5
    print(f"[v18h3] y labels forced to the {floor:.1f} pt floor -- still overflowing")
    return floor


def post_adjust(fn):
    """Run fn(fig) immediately before the figure is written."""
    def wrap(orig):
        def _save(fig, name):
            fn(fig)
            return orig(fig, name)
        return _save
    return wrap


def relayout_compact(fig, bottom=0.05, hspace=0.26, legend_xy=(0.535, 0.5125), scale=(1.0, 1.0)):
    """Reclaim the vertical space the two removed x-axis decks used to occupy.

    The V17 layout reserves bottom=0.125 and hspace=0.43 to fit the group name and the sample-size
    caption under every panel. With those gone the reserved band is empty, so the panels are pushed
    small and the marker key floats in a wide gap. The gap must not close completely: the top row's
    tick labels live in it, and the key sits below them. With top=0.975 and two rows the row height
    is h = (top - bottom) / (2 + hspace), the gap spans hspace*h and its centre is bottom + h*(1 +
    hspace/2) -- which is where the key is re-anchored.
    """
    # `scale` compresses the canvas while leaving every font at its point size, so the figure gets
    # DENSER rather than merely smaller -- rescaling the fonts too would just be a zoom. figsize is
    # hard-coded inside the V17 routine, so it is rescaled here instead of edited there. Subplot
    # positions are fractions, so the layout follows; only the text has to be refitted afterwards.
    sx, sy = scale
    if (sx, sy) != (1.0, 1.0):
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * sx, h * sy)
        print(f"[v18h3] canvas {w:.2f}x{h:.2f} -> {w * sx:.2f}x{h * sy:.2f} in "
              f"(width x{sx:g}, height x{sy:g})")
    fig.subplots_adjust(bottom=bottom, hspace=hspace)
    for leg in fig.legends:
        leg.set_bbox_to_anchor(legend_xy, transform=fig.transFigure)


def save_as(stem_map):
    """Rename figures at save time: {v17_stem: new_stem}. The Mantel all-84 figure is also called
    'feature_correlation', which would overwrite the Spearman-matrix figure already in this
    directory, so the two must be told apart by name."""
    def wrap(orig):
        def _save(fig, name):
            return orig(fig, stem_map.get(name, name))
        return _save
    return wrap


def check(path, cols):
    """1596 rows = 84 held-out OASIS heads x 19 electrodes, and the columns the figures index by."""
    if not os.path.isfile(path):
        raise SystemExit(f"[v18h3] missing {path}")
    rows = list(csv.DictReader(open(path)))
    missing = [c for c in cols if c not in (rows[0] if rows else {})]
    if missing:
        raise SystemExit(f"[v18h3] {os.path.basename(path)} lacks columns {missing}")
    if len(rows) != 1596:
        raise SystemExit(f"[v18h3] {os.path.basename(path)} has {len(rows)} rows, want 1596")
    heads = {r["subject"].lower() for r in rows}
    if len(heads) != 84:
        raise SystemExit(f"[v18h3] {os.path.basename(path)} covers {len(heads)} heads, want 84")
    print(f"[v18h3] {os.path.basename(path)}: 1596 rows, 84 heads, columns OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["corr", "anova", "mantel"], default=None)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--keep-hc", action="store_true",
                    help="draw the control group as HC, exactly as the V17 original (default: CU)")
    ap.add_argument("--no-amyloid-row", dest="amyloid_row", action="store_false", default=True,
                    help="omit the binary Aβ+ variable from the Mantel figures")
    ap.add_argument("--cl-pos", type=float, default=26.0,
                    help="Centiloid amyloid-positivity cutline (project-wide value: 26)")
    ap.add_argument("--scale", type=float, nargs=2, metavar=("SX", "SY"), default=[0.85, 0.70],
                    help="compress the anova canvas: width x SX, height x SY (fonts unchanged)")
    ap.add_argument("--wide-labels", action="store_true",
                    help="keep the V17 three-deck x-axis (Male/Female + group name + n caption)")
    a = ap.parse_args()

    check(DEP, ["subject", "group", "electrode", "gm_frac", "wm_frac", "other_frac"])
    check(DEP_CSF, ["subject", "group", "electrode", "csf_frac", "gm_frac", "wm_frac", "other_frac"])

    if not os.path.isfile(M.C.CSV):
        raise SystemExit(f"[v18h3] demographics table not found: {M.C.CSV}")
    sids = {r["sid"].lower() for r in csv.DictReader(open(M.C.CSV))}
    print(f"[v18h3] demographics: {len(sids)} sids, covering "
          f"{len({r['subject'].lower() for r in csv.DictReader(open(DEP))} & sids)}/84 heads")

    os.makedirs(a.outdir, exist_ok=True)
    M.DEP_CSV, M.DEP_CSF_CSV, M.OUT = DEP, DEP_CSF, a.outdir
    _save0 = M._save
    M._save = lambda fig, name: _save0(fig, f"{name}_v18h3")
    print(f"[v18h3] OUT={a.outdir}")

    if a.only in (None, "corr"):
        M.fix_feature_correlation()          # no group label in this figure -- nothing to relabel

    if a.only in (None, "anova"):
        if a.keep_hc:
            M.fig_parenchyma_anova()
        elif a.wide_labels:
            with relabel_group():                       # V17 three-deck axis, HC renamed to CU
                M.fig_parenchyma_anova()
        else:
            def finish(fig):
                relayout_compact(fig, scale=tuple(a.scale))
                fit_ylabels(fig, fig.axes)          # fig.axes is the 6 panels; the key is a fig.legend
                fit_xticklabels(fig, fig.axes)
            keep = M._save
            M._save = post_adjust(finish)(M._save)
            swap = swap_pa_panels([(("skull", "Skull thickness (mm)"),
                                    ("scalp", "Scalp thickness (mm)"))])
            with compact_group_axis(["AD-M", "AD-F", "CU-M", "CU-F"], 6), swap:
                M.fig_parenchyma_anova()
            M._save = keep

    if a.only in (None, "mantel"):
        # Three panels, matching what the V17 run left on disk: the all-84 network plus the two
        # single-cohort ones. The V17 all-84 Mantel figure was never actually written --
        # feature_correlation.png there is the Spearman matrix from fix_feature_correlation (17:46),
        # while fig_mantel_correlation only ran with a group and produced _AD42/_HC42 (18:21).
        stems = {"feature_correlation": "feature_correlation_mantel"}
        if not a.keep_hc:
            stems["feature_correlation_HC42"] = "feature_correlation_CU42"
        M._save = save_as(stems)(M._save)
        rng_note = f"{M.PERMS if hasattr(M, 'PERMS') else 999} permutations"
        with with_amyloid_feature(cl_pos=a.cl_pos) if a.amyloid_row else _null():
            for grp in (None, "AD", "HC"):
                lab = grp or "all-84"
                print(f"[v18h3] Mantel network: {lab}  ({rng_note})", flush=True)
                if grp == "HC" and not a.keep_hc:
                    with relabel_cohort_title():
                        M.fig_mantel_correlation(grp)
                else:
                    M.fig_mantel_correlation(grp)


if __name__ == "__main__":
    sys.exit(main())
