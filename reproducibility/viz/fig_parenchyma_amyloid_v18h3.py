"""Plot V18 deposition by diagnosis and amyloid status.

The main panel uses the full ``{AD, CU} x {Aβ+, Aβ-}`` design and a Centiloid
cutoff of 26. Subjects without Centiloid measurements are excluded and counted
in the figure sidecar.
"""
import argparse, csv, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPRO_ROOT)
sys.path.insert(0, HERE)
import repro_config as RC                                               # noqa: E402
os.environ.setdefault("MPLCONFIGDIR", str(RC.RESULTS_DIR / "v18h3" / "_mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import fig_v17_update0801_0803 as M                                       # noqa: E402
import fig_update0803_v18h3 as W                                          # noqa: E402
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.lines import Line2D                                       # noqa: E402
import pandas as pd                                                       # noqa: E402
import statsmodels.formula.api as smf                                     # noqa: E402
import statsmodels.api as sm                                              # noqa: E402
from scipy.stats import mannwhitneyu                                      # noqa: E402

ROOT = str(RC.WORK_DIR)
PUP = str(RC.PUP_MNI_DIR)
DEP_CSF = os.path.join(PUP, "cohort_energy_deposition_csf_v18h3.csv")     # hero3, CSF/GM/WM split
DEP_MC = os.path.join(PUP, "cohort_energy_deposition_mc.csv")             # Monte Carlo, --full only
OUT = str(RC.RESULTS_DIR / "v18h3")
CL_POS = 26.0                                                             # Centiloid amyloid+ cutoff
SCALE = (0.85, 0.70)                                                      # canvas compression (w, h)

# Pale translucent blue, built to the same recipe as _pa_box's GREEN_BG / WARM_BG (RGBA, alpha
# 0.168), so the tint sits at the same weight as the sibling figure's rows.
BLUE_BG = (0.66, 0.78, 0.89, 0.168)

# Amyloid token first, matching how the four groups were specified. The 1x3 layout is wide enough to
# hold this near the inherited 12 pt; fit_xticklabels measures it rather than assuming.
XLAB = ["Aβ+ AD", "Aβ− AD", "Aβ+ CU", "Aβ− CU"]

PAN = [("scalp_skull", "Scalp + skull thickness (mm)", BLUE_BG),
       ("par_su", "GM+WM energy-deposition fraction (%)", BLUE_BG),
       ("csf", "CSF energy-deposition fraction (%)", BLUE_BG)]

# Analysed in the sidecar but deliberately NOT given panels: GM and WM separately. The drawn panel
# is their sum (parenchyma), and three boxes of the same quantity would invite reading the sum and
# its parts as independent results when GM-WM correlate at rho = 0.83 across heads.
EXTRA = [("gm", "GM energy-deposition fraction (%)"),
         ("wm", "WM energy-deposition fraction (%)"),
         ("scalp", "Scalp thickness (mm), for reference"),
         ("skull", "Skull thickness (mm), for reference")]

PAN_FULL = [("gm", "GM energy-deposition fraction (%)", M.GREEN_BG),
            ("wm", "WM energy-deposition fraction (%)", M.GREEN_BG),
            ("csf", "CSF energy-deposition fraction (%)", M.GREEN_BG),
            ("par_su", "Surrogate parenchyma fraction (%)", M.WARM_BG),
            ("par_mc", "Monte-Carlo parenchyma fraction (%)", M.WARM_BG),
            ("bpf", "Brain parenchymal fraction (BPF)", M.WARM_BG)]


def perhead(path, cols):
    """Mean over the 19 electrodes of a sum of deposition columns, per head."""
    d = defaultdict(list)
    for r in csv.DictReader(open(path)):
        d[r["subject"].lower()].append(sum(float(r[c]) for c in cols))
    return {h: float(np.mean(v)) for h, v in d.items()}


def load():
    """One row per head with both factors resolved. Heads without a Centiloid are dropped."""
    gm = perhead(DEP_CSF, ["gm_frac"])
    wm = perhead(DEP_CSF, ["wm_frac"])
    csf = perhead(DEP_CSF, ["csf_frac"])
    mc = perhead(DEP_MC, ["gm_frac", "wm_frac"])
    strat = {r["subject"].lower(): r for r in csv.DictReader(open(M.STRAT_CSV))}
    feat = {r["sid"].lower(): r for r in csv.DictReader(open(M.C.CSV))}

    rows, skipped = [], []
    for h in sorted(gm):
        st, f = strat.get(h), feat.get(h)
        if st is None or f is None or h not in mc:
            skipped.append((h, "no metadata")); continue
        try:
            cl = float(st["centiloid"])
        except (TypeError, ValueError, KeyError):
            skipped.append((h, "no Centiloid")); continue
        try:
            bpf, scalp, skull = float(f["bpf"]), float(f["scalp"]), float(f["skull"])
        except (TypeError, ValueError, KeyError):
            skipped.append((h, "no BPF/scalp/skull")); continue
        dx = "AD" if st["group"] == "AD" else "HC"          # "HC" is _pa_box's internal key; drawn CU
        rows.append(dict(head=h, dx=dx, sex="M" if cl >= CL_POS else "F", centiloid=cl,
                         gm=gm[h] * 100, wm=wm[h] * 100, csf=csf[h] * 100,
                         par_su=(gm[h] + wm[h]) * 100, par_mc=mc[h] * 100,
                         bpf=bpf, scalp=scalp, skull=skull, scalp_skull=scalp + skull))
    return rows, skipped


def cells(rows):
    n = {}
    for dx in ("AD", "HC"):
        for sx in ("M", "F"):
            n[(dx, sx)] = sum(1 for r in rows if r["dx"] == dx and r["sex"] == sx)
    return n


def twoway(rows, items, path):
    """Type II two-way ANOVA per panel: diagnosis x amyloid, with the interaction.

    Type II because the design is unbalanced (20/12/14/26 at the CL>=26 cutline); it is the right
    choice when the interaction is not assumed present, and the interaction term is reported
    alongside so that assumption can be judged rather than taken on trust. Partial eta squared
    accompanies every term because with cells this small an F test alone cannot separate "no effect"
    from "no power", and the four cell means plus the two amyloid simple effects are printed so the
    SHAPE of an interaction is visible rather than inferred from its p value. The box brackets are
    Mann-Whitney and do not depend on this model.
    """
    df = pd.DataFrame(rows)
    df["dxf"] = df["dx"].map({"AD": "AD", "HC": "CU"})
    df["amy"] = df["sex"].map({"M": "pos", "F": "neg"})
    n = cells(rows)
    lines = ["Two-way ANOVA (type II), diagnosis x amyloid -- V18 hero3 surrogate",
             "cells: " + "  ".join(f"{d}-A{'+' if s == 'M' else '-'} n={v}" for (d, s), v in n.items()),
             f"amyloid+ = Centiloid >= {CL_POS:.0f}", ""]
    for it in items:
        key, lab = it[0], it[1]
        m = smf.ols(f"{key} ~ C(dxf) * C(amy)", data=df).fit()
        tab = sm.stats.anova_lm(m, typ=2)
        ss_res = tab.loc["Residual", "sum_sq"]
        mu = {(d, s): df[(df.dxf == d) & (df.amy == s)][key].mean()
              for d in ("AD", "CU") for s in ("pos", "neg")}
        lines.append(lab)
        lines.append("    cell means   AD/A+ {:.4f}   AD/A− {:.4f}   CU/A+ {:.4f}   CU/A− {:.4f}"
                     .format(mu[("AD", "pos")], mu[("AD", "neg")],
                             mu[("CU", "pos")], mu[("CU", "neg")]))
        lines.append("    amyloid simple effect (A+ minus A−):   within AD {:+.4f}   within CU {:+.4f}"
                     .format(mu[("AD", "pos")] - mu[("AD", "neg")],
                             mu[("CU", "pos")] - mu[("CU", "neg")]))
        for term, short in (("C(dxf)", "diagnosis"), ("C(amy)", "amyloid"),
                            ("C(dxf):C(amy)", "interaction")):
            ss, F, p = tab.loc[term, "sum_sq"], tab.loc[term, "F"], tab.loc[term, "PR(>F)"]
            lines.append(f"    {short:12s} F={F:7.3f}  p={p:.4f}  partial eta2={ss / (ss + ss_res):.4f}"
                         f"{'  *' if p < 0.05 else ''}")
        # the same four comparisons the figure draws as brackets, so the sidecar documents the
        # panels exactly and the undrawn keys are reported on identical terms
        for nm, (d1, s1), (d2, s2) in (("A+ vs A- within AD", ("AD", "pos"), ("AD", "neg")),
                                       ("A+ vs A- within CU", ("CU", "pos"), ("CU", "neg")),
                                       ("AD vs CU within A+", ("AD", "pos"), ("CU", "pos")),
                                       ("AD vs CU within A-", ("AD", "neg"), ("CU", "neg"))):
            u = mannwhitneyu(df[(df.dxf == d1) & (df.amy == s1)][key],
                             df[(df.dxf == d2) & (df.amy == s2)][key], alternative="two-sided")[1]
            lines.append(f"    post-hoc {nm:20s} p={u:.4f}{'  *' if u < 0.05 else ''}")
        lines.append("")
    txt = "\n".join(lines)
    open(path, "w").write(txt + "\n")
    print(txt)


def caption(fig, rows, y, fs=9.0, dy=0.052):
    """Two lines, not one.

    A single line of this caption measures ~9.4 in at 9 pt, which is WIDER than the compressed
    canvas -- and because bbox_inches="tight" grows the saved image to contain every artist, that one
    string silently undid the horizontal compression (measured: width x0.994 instead of x0.85).
    Splitting it is what actually makes the figure narrower; shrinking the font instead would have
    hidden the constraint rather than removed it.
    """
    n = cells(rows)
    order = [("AD", "M"), ("AD", "F"), ("HC", "M"), ("HC", "F")]
    l1 = "  ·  ".join(f"{XLAB[i]} n={n[k]}" for i, k in enumerate(order)) \
        + f"   ·   Aβ+ / Aβ− = Centiloid ≥ / < {CL_POS:.0f}"
    l2 = "energy-deposition fractions are V18 hero3 surrogate output   ·   post-hoc: Mann–Whitney U"
    fig.text(0.5, y + dy, l1, ha="center", va="bottom", fontsize=fs, color=M.GREY)
    fig.text(0.5, y, l2, ha="center", va="bottom", fontsize=fs, color=M.GREY)


def legend(fig, xy):
    fig.legend(handles=[Line2D([], [], marker="D", ls="none", mfc="white", mec=M.INK, mew=1.5,
                               ms=9, label="Mean"),
                        Line2D([], [], color="#2a2a2a", lw=1.9, label="Median")],
               loc="center", bbox_to_anchor=xy, ncol=2, frameon=True, fontsize=10.5,
               handletextpad=0.4, columnspacing=1.2, borderpad=0.45, handlelength=1.6,
               edgecolor="#d6d6d6", facecolor="white").get_frame().set_linewidth(0.7)


def figure_row(rows, outdir, stem):
    """One row of three panels: the delivery-gating anatomy, then the two tissue compartments."""
    # Canvas compression preserves font sizes, so the
    # figure becomes denser rather than merely smaller; the two fitters below repair what that costs.
    fig, axs = plt.subplots(1, 3, figsize=(11.0 * SCALE[0], 5.6 * SCALE[1]))
    # The marker key goes ABOVE the panels: after the 30% vertical compression the band under the
    # tick labels is only wide enough for the two caption lines, and a key placed there lands on
    # top of them. The top strip was empty anyway.
    fig.subplots_adjust(left=0.072, right=0.99, top=0.905, bottom=0.185, wspace=0.30)
    with W.compact_group_axis(XLAB, len(PAN)):
        for ax, (key, ylab, bg) in zip(axs, PAN):
            M._pa_box(ax, rows, key, ylab, bg)
    legend(fig, (0.5, 0.958))
    caption(fig, rows, 0.012)
    W.fit_ylabels(fig, list(axs))
    W.fit_xticklabels(fig, list(axs))
    save(fig, outdir, stem)


def figure_full(rows, outdir, stem):
    """Draw the optional 2-by-3 panel including the Monte Carlo control."""
    fig, axs = plt.subplots(2, 3, figsize=(9.3 * SCALE[0], 8.6 * SCALE[1]))
    fig.subplots_adjust(left=0.082, right=0.988, top=0.975, bottom=0.125, wspace=0.34, hspace=0.43)
    with W.compact_group_axis(XLAB, len(PAN_FULL)):
        for ax, (key, ylab, bg) in zip(axs.flatten(), PAN_FULL):
            M._pa_box(ax, rows, key, ylab, bg)
    legend(fig, (0.535, 0.525))
    caption(fig, rows, 0.018)
    W.relayout_compact(fig, bottom=0.075, hspace=0.26, legend_xy=(0.535, 0.525))
    W.fit_ylabels(fig, list(axs.flatten()))
    W.fit_xticklabels(fig, list(axs.flatten()))
    save(fig, outdir, stem)


def save(fig, outdir, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(outdir, stem + ".png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--stem", default="parenchyma_amyloid_box_v18h3")
    ap.add_argument("--full", action="store_true", help="2x3 incl. GM/WM/BPF and the MC control")
    a = ap.parse_args()

    if not os.path.isfile(M.C.CSV):                     # renamed directory; see fig_update0803_v18h3
        M.C.CSV = M.C.CSV.replace(os.sep + "cohorts-features-for-statistic" + os.sep,
                                  os.sep + "V17-cohorts-features-for-statistic" + os.sep)
    for p in (DEP_CSF, DEP_MC, M.STRAT_CSV, M.C.CSV):
        if not os.path.isfile(p):
            raise SystemExit(f"[v18h3] missing {p}")
    os.makedirs(a.outdir, exist_ok=True)

    rows, skipped = load()
    n = cells(rows)
    print(f"[v18h3] {len(rows)} heads in the 2x2, {len(skipped)} excluded "
          f"({sum(1 for _, why in skipped if why == 'no Centiloid')} without Centiloid)")
    for k, v in n.items():
        print(f"    {k[0]:3s} amyloid{'+' if k[1] == 'M' else '-'}: n={v}")
    if min(n.values()) < 5:
        raise SystemExit("[v18h3] a cell has fewer than 5 heads -- the 2x2 is not viable")

    pan = PAN_FULL if a.full else PAN
    stem = a.stem + ("_full" if a.full else "")
    twoway(rows, [(k, l) for k, l, _ in pan] + EXTRA,
           os.path.join(a.outdir, stem + "_anova.txt"))
    (figure_full if a.full else figure_row)(rows, a.outdir, stem)


if __name__ == "__main__":
    sys.exit(main())
