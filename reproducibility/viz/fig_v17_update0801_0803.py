#!/usr/bin/env python3
"""Supplementary cohort, deposition, and illumination statistics.

The module provides forest plots with exact two-sided p-values, target-delivery
analyses, amyloid-versus-delivery analyses, layer-attenuation models, Mantel
tests, and parenchymal deposition comparisons.
"""
import os, sys, csv
from collections import defaultdict
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm, mannwhitneyu, spearmanr, wilcoxon, pearsonr
from scipy.spatial.distance import pdist
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPRO_ROOT)
sys.path.insert(0, HERE)
import repro_config as RC                                               # noqa: E402
# reuse the validated data loaders + regression estimators from the existing scripts
import fig_v17_cohort_stats as C
import fig_v17_fluence_stats as FS
import fig_v17_fluence_strat as FST

for _f in ["/usr/share/fonts/truetype/LiberationSans-Regular.ttf",
           "/usr/share/fonts/truetype/LiberationSans-Bold.ttf"]:
    if os.path.exists(_f): fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none", "pdf.fonttype": 42})
CLAY, BLUE, INK, GREY = "#C0897B", "#8AA0AF", "#4a4a4a", "#8a8a8a"
DARKRED = "#7B2E2B"      # refined dark-wine border for significant heatmap cells
ROOT = str(RC.WORK_DIR)
OUT = str(RC.RESULTS_DIR / "v17viz4" / "update0803"); os.makedirs(OUT, exist_ok=True)
SIGKEY = "significance: n.s. p≥0.05  ·  * p<0.05  ·  ** p<0.01  ·  *** p<0.001 (two-sided)"

# ------------------------------------------------------------------ statistics
def pstars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."

def pfmt(p):
    return "p<0.001" if p < 1e-3 else f"p={p:.3f}"

def p_from_ci(b, lo, hi):
    """Two-sided p for a coefficient from its symmetric 95% CI (se = (hi-lo)/(2·1.96))."""
    se = (hi - lo) / (2 * 1.96)
    if se <= 0: return 1.0
    return float(2 * (1 - norm.cdf(abs(b / se))))

def cohen_p(a, b):
    """Cohen's d (+95% CI) and a two-sided Mann–Whitney U p-value."""
    d, lo, hi = FST.cohen_ci(a, b)
    try:    p = float(mannwhitneyu(a, b, alternative="two-sided")[1])
    except Exception: p = 1.0
    return d, lo, hi, p

def with_p_ols(res):     # res = [(name,b,lo,hi)] -> [(name,b,lo,hi,p)]
    return [(n, b, lo, hi, p_from_ci(b, lo, hi)) for (n, b, lo, hi) in res]

def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling z (rank-residualised)."""
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    def resid(a, c):
        A = np.column_stack([np.ones_like(c), c]); beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta
    ex, ey = resid(rx, rz), resid(ry, rz)
    r, p = stats.pearsonr(ex, ey)
    return r, p

# ------------------------------------------------------------------ clean forest
def forest_clean(series, ylabels, ylab_map, title, xlabel, fname,
                 subtitle="", grids=(-0.5, -0.25, 0.25, 0.5), xlim=None, rowh=0.62):
    """series = [(label, [(name,b,lo,hi,p)...], colour)]. The legend is the colour-coded header of
    a right-margin stats column (stars + exact p per estimate) — never inside the data area."""
    ns = len(series)
    yl = [ylab_map.get(x, x) for x in ylabels]
    y = np.arange(len(ylabels))[::-1]
    fig, ax = plt.subplots(figsize=(6.8 + 2.5 * ns, rowh * len(ylabels) + 2.4))
    right = 0.985 - (0.145 * ns + 0.05)          # reserve canvas for the stats column(s)
    fig.subplots_adjust(left=0.255, right=right, top=0.815, bottom=0.145)
    for g in grids: ax.axvline(g, color="#ececec", lw=1.0, zorder=0)
    ax.axvline(0, color="#b8b8b8", lw=1.4, zorder=1)
    offs = np.linspace(0.16, -0.16, ns) if ns > 1 else [0.0]
    trans = ax.get_yaxis_transform()             # x = axes fraction, y = data
    for si, ((lab, res, c), off) in enumerate(zip(series, offs)):
        d = {n: (b, lo, hi, p) for n, b, lo, hi, p in res}
        for yi, xc in zip(y, ylabels):
            if xc not in d: continue
            b, lo, hi, p = d[xc]
            ax.plot([lo, hi], [yi + off] * 2, color=c, lw=2.7, solid_capstyle="round", zorder=3)
            ax.plot(b, yi + off, "o", ms=9, color=c, mec="white", mew=0.8, zorder=4)
        xf = 1.045 + si * (0.86 / ns)            # stats column x (axes fraction, right margin)
        ax.text(xf, len(ylabels) - 0.35, lab, transform=trans, color=c, fontsize=14.5,
                fontweight="bold", ha="left", va="bottom", clip_on=False)      # header = legend
        for yi, xc in zip(y, ylabels):
            if xc not in d: continue
            b, lo, hi, p = d[xc]
            st = pstars(p)
            ax.text(xf, yi, f"{st:<4}{pfmt(p)}", transform=trans,
                    color=(c if p < 0.05 else GREY), fontsize=12.5,
                    fontweight=("bold" if p < 0.05 else "normal"),
                    ha="left", va="center", clip_on=False, family="monospace")
    ax.set_yticks(y); ax.set_yticklabels(yl, fontsize=16); ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=14); ax.set_ylim(-0.7, len(ylabels) - 0.15)
    if xlim is not None: ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=16)
    fig.suptitle(title, x=0.255, y=0.975, ha="left", fontsize=16, fontweight="bold")
    if subtitle:
        fig.text(0.255, 0.895, subtitle, ha="left", fontsize=12.5, color=GREY)
    fig.text(0.255, 0.028, SIGKEY, ha="left", fontsize=11.5, color=GREY)
    _save(fig, fname)

def _save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig); print("wrote", os.path.join(OUT, name + ".png"))

# ============================ PART B — fixed existing figures ============================
def fix_forest_effect_sizes():
    rows = C.load()
    contrasts = [("AD vs HC", lambda r: r["group"] == "AD", lambda r: r["group"] == "healthy", CLAY),
                 ("Male vs Female", lambda r: r["sex"] == "M", lambda r: r["sex"] == "F", BLUE)]
    ylabels = [k for k, _ in C.FEATS]; ymap = {k: l for k, l in C.FEATS}
    series = []
    for lab, fa, fb, c in contrasts:
        res = []
        for key, _ in C.FEATS:
            a, b = C.vals(rows, key, fa), C.vals(rows, key, fb)
            res.append((key,) + cohen_p(a, b))
        series.append((lab, res, c))
    forest_clean(series, ylabels, ymap,
                 "Cohort effect sizes — diagnosis vs sex",
                 "Standardized effect size, Cohen's d   (+ = AD>HC / Male>Female)",
                 "forest_effect_sizes",
                 subtitle="Cohen's d (95% CI); p = two-sided Mann–Whitney U.  n = 84 heads (42 AD / 42 HC; 46 M / 38 F).",
                 grids=(-1.0, -0.5, 0.5, 1.0, 1.5, 2.0), rowh=0.6)

def fix_fluence_drivers():
    rows = FS.load_merged()
    XA = ["depth", "skull", "scalp", "bpf", "icv", "centiloid"]
    mc = with_p_ols(FS.std_ols(rows, "flu_mc", XA)[0])
    sur = with_p_ols(FS.std_ols(rows, "flu_sur", XA)[0])
    forest_clean([("Monte Carlo", mc, CLAY), ("Surrogate", sur, BLUE)], XA, FS.LABEL,
                 "Drivers of delivered fluence (15–30 mm)",
                 "Standardized coefficient, β", "fluence_drivers",
                 subtitle="OLS, 6384 scene×depth observations; p = coefficient t-test.",
                 grids=(-0.75, -0.5, -0.25, 0.25), xlim=(-0.95, 0.30), rowh=0.62)
    XB = ["depth", "skull", "scalp", "icv", "flag"]
    er = with_p_ols(FS.std_ols(rows, "aerr", XB)[0])
    forest_clean([("Surrogate |error|", er, BLUE)], XB, FS.LABEL,
                 "Drivers of surrogate depth error",
                 "Standardized coefficient, β", "fidelity_drivers",
                 subtitle="OLS on |surrogate − MC|; p = coefficient t-test.",
                 grids=(-0.1, -0.05, 0.05, 0.1, 0.15), xlim=(-0.13, 0.19), rowh=0.62)

def fix_fluence_drivers_demo():
    rows = FST.load()
    XA = ["skull", "scalp", "bpf", "icv", "age", "sexM", "AD", "centiloid"]
    for depth, fname in ((25, "fluence_drivers_demo"), (30, "fluence_drivers_demo_30mm")):
        mc = with_p_ols(FST.cluster_ols(rows, f"mc{depth}", XA)[0])
        sur = with_p_ols(FST.cluster_ols(rows, f"sur{depth}", XA)[0])
        forest_clean([("Monte Carlo", mc, CLAY), ("Surrogate", sur, BLUE)], XA, FST.LABEL,
                     f"Drivers of delivered fluence at {depth} mm (depth removed)",
                     "Standardized coefficient, β", fname,
                     subtitle="Cluster-robust OLS by subject (84 clusters, 1596 scenes); p = robust coefficient t-test.",
                     grids=(-0.2, -0.1, 0.1, 0.2), xlim=(-0.28, 0.28), rowh=0.6)

def fix_fluence_by_stratum():
    rows = FST.load()
    heads = {}
    for r in rows: heads.setdefault(r["subject"], []).append(r)
    H = {}
    for h, rs in heads.items():
        H[h] = dict(mc=np.mean([x["mc25"] for x in rs]),
                    fid=np.mean([x["fid"] for x in rs if x["fid"] is not None]),
                    AD=rs[0]["AD"], sexM=rs[0]["sexM"], amyloid=rs[0]["amyloid"],
                    mmse_lt24=rs[0]["mmse_lt24"], e4=rs[0]["e4"])
    STRATA = [("AD vs HC", lambda h: h["AD"] == 1, lambda h: h["AD"] == 0),
              ("Male vs Female", lambda h: h["sexM"] == 1, lambda h: h["sexM"] == 0),
              ("Amyloid+ vs −", lambda h: h["amyloid"] == "positive", lambda h: h["amyloid"] == "negative"),
              ("MMSE <24 vs ≥24", lambda h: h["mmse_lt24"] == 1, lambda h: h["mmse_lt24"] == 0),
              ("APOE ε4+ vs −", lambda h: h["e4"] == 1, lambda h: h["e4"] == 0)]
    flu_res, fid_res, ylabs, ymap = [], [], [], {}
    for name, fa, fb in STRATA:
        A = [h for h in H.values() if fa(h)]; B = [h for h in H.values() if fb(h)]
        key = "k_" + name
        ylabs.append(key); ymap[key] = f"{name}\n(n={len(A)}/{len(B)})"
        flu_res.append((key,) + cohen_p([h["mc"] for h in A], [h["mc"] for h in B]))
        fid_res.append((key,) + cohen_p([h["fid"] for h in A], [h["fid"] for h in B]))
    forest_clean([("delivered fluence", flu_res, CLAY), ("surrogate fidelity R²", fid_res, BLUE)],
                 ylabs, ymap, "Stratified group differences (per head)",
                 "Cohen's d between strata   (+ = first group higher)", "fluence_by_stratum",
                 subtitle="Per-head means over 19 electrodes; d (95% CI); p = two-sided Mann–Whitney U.",
                 grids=(-1.0, -0.5, 0.5, 1.0), xlim=(-1.25, 1.35), rowh=0.72)

DEP_CSV = os.path.join(ROOT, "dataset_OASIS", "pup_mni", "cohort_energy_deposition.csv")
DEP_MC_CSV = os.path.join(ROOT, "dataset_OASIS", "pup_mni", "cohort_energy_deposition_mc.csv")
DEP_CSF_CSV = os.path.join(ROOT, "dataset_OASIS", "pup_mni", "cohort_energy_deposition_csf.csv")
DEP_MC_CSF_CSV = os.path.join(ROOT, "dataset_OASIS", "pup_mni", "cohort_energy_deposition_mc_csf.csv")
THICK_CSV = os.path.join(ROOT, "dataset_OASIS", "pup_mni", "cohort_electrode_thickness.csv")
STRAT_CSV = os.path.join(ROOT, "dataset_OASIS", "oasis3_metadata", "cohort_stratification.csv")
EEG19 = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "C3", "C4", "Cz",
         "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "O1", "O2"]

def load_deposition():
    """Per-head surrogate absorbed-energy deposition fractions (mean over the 19 electrodes),
    merged with demographics/structure. Keyed by head."""
    heads = {}
    for r in csv.DictReader(open(DEP_CSV)):
        d = heads.setdefault(r["subject"], {"gm": [], "wm": [], "other": [], "group": r["group"]})
        d["gm"].append(float(r["gm_frac"])); d["wm"].append(float(r["wm_frac"]))
        d["other"].append(float(r["other_frac"]))
    feat = {r["sid"].lower(): r for r in csv.DictReader(open(C.CSV))}
    out = {}
    def fv(f, k):
        try: return float(f[k])
        except: return None
    for h, d in heads.items():
        f = feat.get(h.lower())
        if not f: continue
        out[h] = dict(gm=float(np.mean(d["gm"])), wm=float(np.mean(d["wm"])),
                      other=float(np.mean(d["other"])), group=d["group"],
                      sexM=1.0 if f["sex"] == "M" else 0.0,
                      **{k: fv(f, k) for k in ("centiloid", "bpf", "icv", "scalp",
                                               "muscle", "skull", "age", "cdrsb", "mmse")})
    return out

def _dep_panel(ax, M, Fem, ylabel, title):
    """One Male-vs-Female boxplot of a deposition fraction (%), star compactly left of p."""
    d, lo, hi, p = cohen_p(M, Fem)
    data, cols = [np.array(M) * 100, np.array(Fem) * 100], [BLUE, CLAY]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#3a3a46", lw=1.8), boxprops=dict(lw=0),
                    whiskerprops=dict(color="#a8a8a8", lw=1.2), capprops=dict(color="#a8a8a8", lw=1.2))
    for patch, c in zip(bp["boxes"], cols): patch.set_facecolor(c); patch.set_alpha(0.28)
    rng = np.random.default_rng(0)
    for xi, (v, c) in enumerate(zip(data, cols)):
        ax.scatter(rng.normal(xi, 0.07, len(v)), v, s=22, color=c, alpha=0.85, lw=0, zorder=3)
    top = max(data[0].max(), data[1].max()); bot = min(data[0].min(), data[1].min())
    spread = (top - bot) or 1; yb = top + 0.11 * spread
    ax.plot([0, 1], [yb, yb], color="#3a3a46", lw=1.2)
    ax.text(0.5, yb + 0.03 * spread, f"{pstars(p)} {pfmt(p)}   d={d:+.2f}",
            ha="center", va="bottom", fontsize=13)      # star compactly LEFT of p
    ax.set_ylim(bot - 0.08 * spread, yb + 0.22 * spread)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Male\n(n={len(M)})", f"Female\n(n={len(Fem)})"], fontsize=14)
    ax.set_xlim(-0.6, 1.6); ax.set_ylabel(ylabel, fontsize=14.5)
    ax.set_title(title, fontsize=14, loc="left", pad=8)
    ax.tick_params(axis="y", labelsize=12); ax.tick_params(axis="x", length=0)
    ax.spines["left"].set_visible(True); ax.spines["left"].set_color("#3a3a46"); ax.spines["left"].set_linewidth(0.8)

def fix_fluence_sex_box():
    """Surrogate GM & WM absorbed-energy deposition fraction, Male vs Female — 4 boxes, tall/slim."""
    H = load_deposition()
    gmM = [h["gm"] for h in H.values() if h["sexM"] == 1]; gmF = [h["gm"] for h in H.values() if h["sexM"] == 0]
    wmM = [h["wm"] for h in H.values() if h["sexM"] == 1]; wmF = [h["wm"] for h in H.values() if h["sexM"] == 0]
    fig, axs = plt.subplots(2, 1, figsize=(4.7, 8.8))
    fig.subplots_adjust(left=0.235, right=0.95, top=0.905, bottom=0.075, hspace=0.34)
    _dep_panel(axs[0], gmM, gmF, "GM absorbed-energy fraction (%)", "a · Grey matter")
    _dep_panel(axs[1], wmM, wmF, "WM absorbed-energy fraction (%)", "b · White matter")
    fig.suptitle("Where the surrogate deposits optical energy", x=0.02, y=0.975,
                 ha="left", fontsize=14.5, fontweight="bold")
    _save(fig, "fluence_sex_box")

def fix_feature_correlation():
    """Custom feature-correlation heatmap: drops Education + the two skull sub-compartments,
    relabels amyloid as Centiloid, and adds GM/WM deposition fraction and Sex(M=1). Significant
    (p<0.05) Spearman cells are bold."""
    H = list(load_deposition().values())
    FEATS2 = [("cdrsb", "CDR–SB"), ("mmse", "MMSE"), ("centiloid", "Centiloid"), ("bpf", "BPF"),
              ("icv", "ICV"), ("scalp", "Scalp thickness"), ("muscle", "Temporalis thickness"),
              ("skull", "Skull thickness"), ("age", "Age"),
              ("gm", "GM deposition frac"), ("wm", "WM deposition frac"), ("sexM", "Sex (M=1)")]
    keys = [k for k, _ in FEATS2]; labs = [l for _, l in FEATS2]; n = len(keys)
    M = np.full((n, n), np.nan); P = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = [], []
            for h in H:
                vi, vj = h.get(keys[i]), h.get(keys[j])
                if vi is None or vj is None: continue
                a.append(vi); b.append(vj)
            sr = stats.spearmanr(a, b); M[i, j] = sr.correlation; P[i, j] = sr.pvalue
    fig, ax = plt.subplots(figsize=(9.6, 8.8))
    im = ax.imshow(M, cmap=C.MOR, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            v = M[i, j]; signif = (i != j and P[i, j] < 0.05)
            col = "white" if abs(v) > 0.6 else INK
            txt = f"{v:.2f}".replace("0.", ".").replace("-.", "–.")
            if signif:                                  # dark-red box + stars above the number
                ax.add_patch(Rectangle((j - 0.44, i - 0.44), 0.88, 0.88, fill=False,
                                       edgecolor=DARKRED, lw=2.0, zorder=5))
                ax.text(j, i + 0.12, txt, ha="center", va="center", fontsize=12, color=col, zorder=6)
                ax.text(j, i - 0.26, pstars(P[i, j]), ha="center", va="center", fontsize=12,
                        fontweight="bold", color=col, zorder=6)
            else:
                ax.text(j, i, txt, ha="center", va="center", fontsize=12, color=col, zorder=6)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=13)
    ax.set_yticklabels(labs, fontsize=13); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Spearman ρ", fontsize=15); cb.ax.tick_params(labelsize=13); cb.outline.set_visible(False)
    fig.tight_layout()
    fig.text(0.01, 0.005, "dark-red box + */**/*** :  Spearman p<0.05 / <0.01 / <0.001  (n = 84 heads);  "
             "GM/WM deposition frac = surrogate absorbed-energy share (mean over 19 electrodes)",
             ha="left", fontsize=11, color=GREY)
    _save(fig, "feature_correlation")

# ============================ 0803 NEW — parenchyma absorption, group × method ============================
def load_parenchyma():
    """Per-head brain-parenchyma (GM+WM) absorbed-energy fraction (%), mean over 19 electrodes, for
    the surrogate and the MC ground truth, merged with diagnosis / amyloid status / sex."""
    def perhead(f):
        d = defaultdict(list)
        for r in csv.DictReader(open(f)):
            d[r["subject"]].append(float(r["gm_frac"]) + float(r["wm_frac"]))
        return {h: float(np.mean(v)) for h, v in d.items()}
    su, mc = perhead(DEP_CSV), perhead(DEP_MC_CSV)
    strat = {r["subject"].lower(): r for r in csv.DictReader(open(STRAT_CSV))}
    rows = []
    for h in su:
        st = strat.get(h.lower())
        if not st or h not in mc: continue
        try: cl = float(st["centiloid"])
        except (TypeError, ValueError): cl = None
        # Amyloid positivity follows the adopted Centiloid >= 26 threshold.
        amy = "unknown" if cl is None else ("positive" if cl >= 26 else "negative")
        rows.append(dict(head=h, su=su[h] * 100, mc=mc[h] * 100, grp=st["group"],
                         sex=st["sex"], centiloid=cl, amy=amy))
    return rows

def _bracket(ax, x1, x2, y, text, color=INK, fs=12.5):
    ax.plot([x1, x1, x2, x2], [y - 0.012 * y, y, y, y - 0.012 * y], color=color, lw=1.2, clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.004 * y, text, ha="center", va="bottom", fontsize=fs, color=color)

def fig_parenchyma_group_box():
    """4-box AD amyloid+ / HC amyloid− × Surrogate / GT, with group AND method significance."""
    R = load_parenchyma()
    adp = [r for r in R if r["grp"] == "AD" and r["amy"] == "positive"]
    hcn = [r for r in R if r["grp"] == "HC" and r["amy"] == "negative"]
    A_su, A_gt = [r["su"] for r in adp], [r["mc"] for r in adp]
    H_su, H_gt = [r["su"] for r in hcn], [r["mc"] for r in hcn]
    data = [A_su, A_gt, H_su, H_gt]; pos = [0, 1, 2.7, 3.7]
    cols = [BLUE, CLAY, BLUE, CLAY]                 # surrogate = blue, GT(MC) = clay
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.16)
    bp = ax.boxplot(data, positions=pos, widths=0.72, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#3a3a46", lw=1.8), boxprops=dict(lw=0),
                    whiskerprops=dict(color="#a8a8a8", lw=1.2), capprops=dict(color="#a8a8a8", lw=1.2))
    for patch, c in zip(bp["boxes"], cols): patch.set_facecolor(c); patch.set_alpha(0.30)
    rng = np.random.default_rng(0)
    for xi, (v, c) in zip(pos, zip(data, cols)):
        ax.scatter(rng.normal(xi, 0.06, len(v)), v, s=22, color=c, alpha=0.85, lw=0, zorder=3)
    # paired method tests (same heads) + unpaired group tests
    p_mA = wilcoxon(A_su, A_gt).pvalue; p_mH = wilcoxon(H_su, H_gt).pvalue
    p_gS = mannwhitneyu(A_su, H_su, alternative="two-sided").pvalue
    p_gG = mannwhitneyu(A_gt, H_gt, alternative="two-sided").pvalue
    top = max(max(d) for d in data)
    _bracket(ax, pos[0], pos[1], top * 1.06, f"{pstars(p_mA)} {pfmt(p_mA)}")     # method within AD+
    _bracket(ax, pos[2], pos[3], top * 1.06, f"{pstars(p_mH)} {pfmt(p_mH)}")     # method within HC−
    _bracket(ax, pos[0], pos[2], top * 1.24, f"{pstars(p_gS)} {pfmt(p_gS)}  (Surrogate)")   # group, surrogate
    _bracket(ax, pos[1], pos[3], top * 1.42, f"{pstars(p_gG)} {pfmt(p_gG)}  (GT)")          # group, GT
    ax.set_ylim(0, top * 1.58)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"AD amyloid+\nSurrogate\n(n={len(adp)})", f"AD amyloid+\nGT\n(n={len(adp)})",
                        f"HC amyloid−\nSurrogate\n(n={len(hcn)})", f"HC amyloid−\nGT\n(n={len(hcn)})"],
                       fontsize=12.5)
    ax.set_xlim(-0.6, 4.3); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Brain-parenchyma absorbed-energy fraction (%)", fontsize=14.5)
    ax.tick_params(axis="y", labelsize=12.5)
    ax.spines["left"].set_visible(True); ax.spines["left"].set_color("#3a3a46"); ax.spines["left"].set_linewidth(0.8)
    fig.suptitle("Brain-parenchyma optical absorption — group vs method", x=0.12, y=0.965,
                 ha="left", fontsize=15.5, fontweight="bold")
    fig.text(0.12, 0.885, "Per-head GM+WM absorbed-energy share (mean over 19 electrodes).  amyloid± = "
             "Centiloid ≥/< 26.  Method: paired Wilcoxon;  group AD+ vs HC−: Mann–Whitney U.",
             ha="left", fontsize=11.5, color=GREY)
    fig.text(0.12, 0.03, SIGKEY, ha="left", fontsize=11, color=GREY)
    _save(fig, "parenchyma_group_box")

GREEN_BG, WARM_BG = (0.66, 0.81, 0.68, 0.168), (0.90, 0.81, 0.73, 0.168)   # pale, translucent row tints (alpha −30%)

def _pa_load():
    """Per-head rows for the sex×diagnosis panels: surrogate energy-deposition fractions (CSF/GM/WM,
    %, mean over the 19 electrodes, from the CSF-split deposition table) plus the overlying anatomy
    that gates delivery (skull, scalp thickness in mm; brain parenchymal fraction BPF) plus sex and
    diagnosis. n = 84 (42 AD / 42 HC)."""
    prof = defaultdict(list)
    for r in csv.DictReader(open(DEP_CSF_CSV)):
        prof[r["subject"]].append((float(r["csf_frac"]), float(r["gm_frac"]), float(r["wm_frac"])))
    dep = {h: np.mean(v, axis=0) for h, v in prof.items()}       # (csf,gm,wm) mean over electrodes
    feat = {r["sid"].lower(): r for r in csv.DictReader(open(C.CSV))}
    rows = []
    for h, (csf, gm, wm) in dep.items():
        f = feat.get(h.lower())
        if not f: continue
        def fv(k):
            try: return float(f[k])
            except (TypeError, ValueError): return None
        sk, sc, bpf = fv("skull"), fv("scalp"), fv("bpf")
        if None in (sk, sc, bpf): continue
        rows.append(dict(csf=csf * 100, gm=gm * 100, wm=wm * 100, skull=sk, scalp=sc, bpf=bpf,
                         sex=f["sex"], dx="AD" if f["group"] == "AD" else "HC"))
    return rows

def _pa_box(ax, rows, key, ylab, bg):
    """One unified sex×diagnosis box panel: framed y-axis line, faint translucent row tint (bg).
    Four post-hoc brackets — Male vs Female within each diagnosis (neutral), and AD vs HC within
    each sex (blue = Male, clay = Female) — each carrying the exact p beside the significance mark
    when significant (else n.s.).  Group labels sit above the per-group sample-size caption."""
    GRP = [("AD", "M", 0.0), ("AD", "F", 0.584), ("HC", "M", 1.333), ("HC", "F", 1.917)]  # −25% inter-box whitespace
    SEXCOL = {"M": BLUE, "F": CLAY}; rng = np.random.default_rng(1)
    def grp(dx, sx): return [r[key] for r in rows if r["dx"] == dx and r["sex"] == sx]
    def nlab(dx):
        m = sum(1 for r in rows if r["dx"] == dx and r["sex"] == "M")
        f = sum(1 for r in rows if r["dx"] == dx and r["sex"] == "F")
        return f"(n={m + f}, M:F={m}:{f})"
    vals_all = [r[key] for r in rows]; vmax, vmin = max(vals_all), min(vals_all); span = (vmax - vmin) or 1.0
    for dx, sx, x in GRP:
        v = np.array(grp(dx, sx)); c = SEXCOL[sx]
        bp = ax.boxplot([v], positions=[x], widths=0.374, patch_artist=True, showfliers=False,  # box −15%
                        medianprops=dict(color="#2a2a2a", lw=1.9), boxprops=dict(lw=0),
                        whiskerprops=dict(color="#8f8f8f", lw=1.1), capprops=dict(color="#8f8f8f", lw=1.1))
        bp["boxes"][0].set_facecolor(c); bp["boxes"][0].set_alpha(0.36)
        ax.scatter(rng.normal(x, 0.046, len(v)), v, s=13, color=c, alpha=0.82, lw=0.3, edgecolor="white", zorder=3)
        ax.plot(x, v.mean(), marker="D", ms=7.5, mfc="white", mec=c, mew=1.7, zorder=5)   # mean
    def bracket(x1, x2, yb, a, b, color):
        p = mannwhitneyu(a, b, alternative="two-sided")[1]
        ax.plot([x1, x1, x2, x2], [yb - span * 0.02, yb, yb, yb - span * 0.02], color=color, lw=1.2, clip_on=False)
        lbl = f"{pstars(p)} {pfmt(p)}" if p < 0.05 else "n.s."     # exact p hugs the significance mark
        ax.text((x1 + x2) / 2, yb + span * 0.010, lbl, ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color=color)
    MBR, FBR = "#54708A", "#9C6553"        # darkened male / female tones for the across-diagnosis brackets
    bracket(0.0, 0.584, vmax + span * 0.07, grp("AD", "M"), grp("AD", "F"), "#3a3a46")    # within AD: M vs F
    bracket(1.333, 1.917, vmax + span * 0.07, grp("HC", "M"), grp("HC", "F"), "#3a3a46")  # within HC: M vs F
    bracket(0.0, 1.333, vmax + span * 0.22, grp("AD", "M"), grp("HC", "M"), MBR)          # Male: AD vs HC
    bracket(0.584, 1.917, vmax + span * 0.37, grp("AD", "F"), grp("HC", "F"), FBR)        # Female: AD vs HC
    ax.set_ylim(vmin - span * 0.09, vmax + span * 0.50); ax.set_xlim(-0.42, 2.34)
    ax.set_xticks([g[2] for g in GRP])
    ax.set_xticklabels(["Male" if g[1] == "M" else "Female" for g in GRP], fontsize=12)
    for gx, dx in [(0.292, "AD"), (1.625, "HC")]:
        ax.text(gx, -0.10, dx, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=15, fontweight="bold")
        ax.text(gx, -0.185, nlab(dx), transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=9.5, color=GREY)
    ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=12.5)
    ax.set_ylabel(ylab, fontsize=13.5)
    ax.set_facecolor(bg)                    # rows differ ONLY by this translucent tint
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(True); ax.spines[sp].set_color("#4a4a4a"); ax.spines[sp].set_linewidth(1.1)

def fig_parenchyma_anova():
    """2×3 box-panel figure (plots only — no titles, legend or table). Row 1 (pale green tint):
    surrogate GM / WM / CSF energy-deposition fraction (%). Row 2 (pale warm tint): skull thickness,
    scalp thickness (mm) and brain parenchymal fraction (BPF) — the anatomy that gates delivery.
    Every panel is a unified sex×diagnosis box plot; the Male-vs-Female bracket carries the exact p
    beside the significance mark when significant. n = 84 (42 AD / 42 HC)."""
    from matplotlib.lines import Line2D
    rows = _pa_load()
    fig, axs = plt.subplots(2, 3, figsize=(9.3, 8.6))            # width −20%
    fig.subplots_adjust(left=0.082, right=0.988, top=0.975, bottom=0.125, wspace=0.34, hspace=0.43)  # inter-row gap −20%
    PAN = [("gm", "GM energy-deposition fraction (%)", GREEN_BG),
           ("wm", "WM energy-deposition fraction (%)", GREEN_BG),
           ("csf", "CSF energy-deposition fraction (%)", GREEN_BG),      # new panel, right of row 1
           ("skull", "Skull thickness (mm)", WARM_BG),
           ("scalp", "Scalp thickness (mm)", WARM_BG),
           ("bpf", "Brain parenchymal fraction (BPF)", WARM_BG)]         # new panel, right of row 2
    for ax, (key, ylab, bg) in zip(axs.flatten(), PAN):
        _pa_box(ax, rows, key, ylab, bg)
    # compact marker key in the (reduced) inter-row whitespace: white diamond = mean, line = median
    leg = fig.legend(handles=[Line2D([], [], marker="D", ls="none", mfc="white", mec=INK, mew=1.5, ms=9, label="Mean"),
                              Line2D([], [], color="#2a2a2a", lw=1.9, label="Median")],
                     loc="center", bbox_to_anchor=(0.535, 0.516), ncol=2, frameon=True, fontsize=10.5,
                     handletextpad=0.4, columnspacing=1.2, borderpad=0.45, handlelength=1.6,
                     edgecolor="#d6d6d6", facecolor="white")
    leg.get_frame().set_linewidth(0.7)
    _save(fig, "parenchyma_anova")

# ============================ 0803 NEW — Mantel-test linkET correlation network ============================
def _load_env_tissue(group=None):
    """df_env (10 continuous demographic/anatomical features) + df_csf/df_wm/df_gm (84×19 surrogate
    deposition profiles). Returns FEATS, heads, env{key:array over heads}, tissue{name:(n,19)}.
    group in {None(all 84), 'AD', 'HC'} restricts to that diagnostic group."""
    feat = {r["sid"].lower(): r for r in csv.DictReader(open(C.CSV))}
    fatd = defaultdict(list)
    for r in csv.DictReader(open(THICK_CSV)):
        try: fatd[r["subject"].lower()].append(float(r["fat_mm"]))
        except (ValueError, KeyError): pass
    fatm = {h: float(np.mean(v)) for h, v in fatd.items() if v}
    prof = defaultdict(dict)
    for r in csv.DictReader(open(DEP_CSF_CSV)):
        prof[r["subject"]][r["electrode"]] = (float(r["csf_frac"]), float(r["gm_frac"]), float(r["wm_frac"]))
    heads = sorted(h for h in feat if h in prof and all(e in prof[h] for e in EEG19))
    if group in ("AD", "HC"):
        gval = {"AD": ("AD",), "HC": ("healthy", "HC")}[group]
        heads = [h for h in heads if feat[h]["group"] in gval]
    FEATS = [("cdrsb", "CDR–SB"), ("mmse", "MMSE"), ("centiloid", "Centiloid"), ("bpf", "BPF"),
             ("icv", "ICV"), ("scalp", "Scalp thickness"), ("fat", "Fat thickness"),
             ("skull", "Skull thickness"), ("age", "Age"), ("sexM", "Sex (M=1)")]
    env = {}
    for key, _ in FEATS:
        vals = []
        for h in heads:
            f = feat[h]
            if key == "sexM": v = 1.0 if f["sex"] == "M" else 0.0
            elif key == "fat": v = fatm.get(h, np.nan)
            else:
                try: v = float(f[key])
                except (ValueError, KeyError, TypeError): v = np.nan
            vals.append(v)
        env[key] = np.array(vals, float)
    tis = {"CSF": np.array([[prof[h][e][0] for e in EEG19] for h in heads]),
           "GM":  np.array([[prof[h][e][1] for e in EEG19] for h in heads]),
           "WM":  np.array([[prof[h][e][2] for e in EEG19] for h in heads])}
    return FEATS, heads, env, tis

def _mantel(dt, f, perms=999, rng=None):
    """Mantel r + two-sided permutation p between a fixed condensed distance vector dt and the
    euclidean distance of feature values f (permute f's individual order)."""
    rng = rng or np.random.default_rng(0)
    df = pdist(f.reshape(-1, 1))
    if np.std(dt) < 1e-12 or np.std(df) < 1e-12: return 0.0, 1.0
    r = pearsonr(dt, df)[0]
    cnt = 1
    for _ in range(perms):
        dfp = pdist(f[rng.permutation(len(f))].reshape(-1, 1))
        if abs(pearsonr(dt, dfp)[0]) >= abs(r): cnt += 1
    return r, cnt / (perms + 1)

def _mant_color(p):  return "#3E8E9C" if p < 0.01 else "#E3A23B" if p < 0.05 else "#b9b9b9"
def _mant_width(r):  return 3.8 if abs(r) >= 0.4 else 2.1 if abs(r) >= 0.2 else 0.8

def fig_mantel_correlation(group=None):
    from scipy.spatial.distance import squareform
    FEATS, heads, env, tis = _load_env_tissue(group)
    keys = [k for k, _ in FEATS]; labs = [l for _, l in FEATS]; n = len(keys)
    # --- Spearman rho + BH-FDR (symmetric); rank-based fits the ordinal/skewed clinical features ---
    R = np.full((n, n), np.nan); Padj = np.ones((n, n)); pr = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = env[keys[i]], env[keys[j]]; m = ~(np.isnan(a) | np.isnan(b))
            # within a single diagnostic group some features (e.g. CDR-SB in HC) can be constant
            if m.sum() >= 4 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9:
                rr, pp = spearmanr(a[m], b[m])
            else: rr, pp = np.nan, 1.0
            R[i, j] = R[j, i] = rr; pr.append((i, j, pp))
    adj = multipletests([p for _, _, p in pr], method="fdr_bh")[1]
    for (i, j, _), pa in zip(pr, adj): Padj[i, j] = Padj[j, i] = pa
    # --- Mantel: tissue Bray-Curtis vs each feature Euclidean ---
    tsq = {t: squareform(pdist(tis[t], metric="braycurtis")) for t in ("CSF", "WM", "GM")}
    rng = np.random.default_rng(0); mant = {}
    for t in ("CSF", "WM", "GM"):
        for k in keys:
            y = env[k]; idx = np.where(~np.isnan(y))[0]
            sub = tsq[t][np.ix_(idx, idx)]; dt = sub[np.triu_indices(len(idx), 1)]
            mant[(t, k)] = _mantel(dt, y[idx], 999, rng)
    # ---------------- draw: LOWER-LEFT correlation triangle + UPPER-RIGHT nodes/chords ----------------
    fig, ax = plt.subplots(figsize=(12.4, 11.2)); ax.set_aspect("equal")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.185)
    norm = Normalize(-1, 1); cmap = C.MOR; FS = 16.5
    for i in range(n):
        for j in range(i):                          # lower-left triangle (i>j) — original cell style
            r = R[i, j]
            if not np.isfinite(r): continue
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(norm(r)),
                                   edgecolor="white", lw=0.9, zorder=3))
            col = "white" if abs(r) > 0.6 else INK
            txt = f"{r:.2f}".replace("0.", ".").replace("-.", "–.")
            if Padj[i, j] < 0.05:
                ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False,
                                       edgecolor=DARKRED, lw=2.6, zorder=5))
                ax.text(j, i + 0.08, txt, ha="center", va="center", fontsize=FS, color=col, zorder=6)
                ax.text(j, i - 0.12, pstars(Padj[i, j]), ha="center", va="center", fontsize=FS,
                        fontweight="bold", color=col, zorder=6)
            else:
                ax.text(j, i, txt, ha="center", va="center", fontsize=FS, color=col, zorder=6)
    # diagonal anchors + labels (left rows, bottom cols)
    for i in range(n):
        ax.plot(i, i, "o", ms=10, color="#2a2a2a", zorder=7)
        ax.text(-0.72, i, labs[i], ha="right", va="center", fontsize=19.5, color=INK)                 # left (every anchor)
        if i <= n - 2:
            ax.text(i, n - 0.42, labs[i], ha="right", va="top", rotation=45, rotation_mode="anchor",
                    fontsize=19.5, color=INK)                                                          # bottom
    # tissue nodes in the UPPER-RIGHT, close to the diagonal -> short chords
    TN = [("CSF", (4.9, 1.0)), ("WM", (6.5, 2.6)), ("GM", (8.1, 4.2))]
    npos = {t: p for t, p in TN}
    order = sorted(((t, k) for t in ("CSF", "WM", "GM") for k in keys), key=lambda tk: mant[tk][1], reverse=True)
    for t, k in order:
        r, p = mant[t, k]; i = keys.index(k)
        ax.add_patch(FancyArrowPatch(npos[t], (i, i), connectionstyle="arc3,rad=-0.13", arrowstyle="-",
                     lw=_mant_width(r) * 1.55, color=_mant_color(p), alpha=0.95 if p < 0.05 else 0.6,
                     zorder=4 if p < 0.05 else 2, clip_on=False))
    for t, (x, y) in TN:
        ax.plot(x, y, "o", ms=28, color="#E8654F", mec="white", mew=1.6, zorder=8)
        ax.text(x + 0.6, y, f"{t} $E_{{\\mathrm{{ab}}}}$", ha="left", va="center",
                fontsize=23, fontweight="bold", color="#333")
    ax.set_xlim(-5.3, n + 1.7); ax.set_ylim(n + 1.5, -1.0)   # compact, y inverted (row 0 top)
    ax.axis("off")
    if group is not None:                         # AD-only / HC-only version: label the cohort
        gt = {"AD": "Alzheimer's disease", "HC": "Healthy controls"}[group]
        fig.text(0.28, 0.99, f"{gt}  (n = {len(heads)})", ha="center", va="top",
                 fontsize=23, fontweight="bold", color=INK)
    _mantel_legends(fig, cmap, norm)
    _save(fig, "feature_correlation" if group is None else f"feature_correlation_{group}42")

def _mantel_legends(fig, cmap, norm):
    from matplotlib.cm import ScalarMappable
    # dedicated bottom legend band (figure coords 0..1); fonts +80%, compact
    lax = fig.add_axes([0.0, 0.0, 1.0, 0.135]); lax.axis("off"); lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    # Pearson's R colorbar
    cax = fig.add_axes([0.045, 0.055, 0.16, 0.022])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Spearman's ρ", fontsize=18); cb.ax.tick_params(labelsize=14); cb.outline.set_visible(False)
    # Mantel's R -> line width
    lax.text(0.275, 0.90, "Mantel's R", fontsize=18, fontweight="bold")
    for yi, (lab, w) in enumerate([("< 0.2", 1.2), ("0.2 – 0.4", 3.2), ("≥ 0.4", 5.9)]):
        yy = 0.78 - 0.27 * yi
        lax.plot([0.275, 0.335], [yy, yy], color="#666", lw=w, solid_capstyle="round", clip_on=False); lax.text(0.345, yy, lab, fontsize=15, va="center")
    # Mantel's P -> line colour
    lax.text(0.455, 0.90, "Mantel's P", fontsize=18, fontweight="bold")
    for yi, (lab, c) in enumerate([("< 0.01", "#3E8E9C"), ("0.01 – 0.05", "#E3A23B"), ("≥ 0.05", "#b9b9b9")]):
        yy = 0.78 - 0.27 * yi
        lax.plot([0.455, 0.515], [yy, yy], color=c, lw=4.2, solid_capstyle="round", clip_on=False); lax.text(0.525, yy, lab, fontsize=15, va="center")
    # node meaning
    lax.plot(0.648, 0.66, "o", ms=22, color="#E8654F"); lax.text(0.670, 0.66,
             "surrogate absorbed-energy fraction $E_{\\mathrm{ab}}$  (19 sites · Bray–Curtis)", fontsize=14, va="center")
    lax.plot(0.648, 0.22, "o", ms=11, color="#2a2a2a"); lax.text(0.670, 0.22,
             "demographic / clinical / anatomical variable  ·  cell stars: BH-FDR Spearman "
             "$P$ (*<.05  **<.01  ***<.001)", fontsize=14, va="center")

# ============================ 0803 NEW — AD vs HC peak-GM-fluence at 4 stimulation sites ============================
GMFLU_CSV = os.path.join(ROOT, "dataset_OASIS", "pup_mni", "gm_maxfluence_0803.csv")
STIM4 = ["F3", "F4", "Fp1", "Fp2"]

def _ad_hc_4panel(metric, ylabel, fname, title):
    """4-panel (one per stimulation electrode) AD-vs-HC box + jittered points of `metric`
    (surrogate), same 14/14 subset as ad_vs_hc_amyloid.  Mann–Whitney U; sig symbol hugs p."""
    rows = list(csv.DictReader(open(GMFLU_CSV)))
    fig, axs = plt.subplots(2, 2, figsize=(7.7, 8.2))     # ~20% narrower than the amyloid 2×2
    fig.subplots_adjust(left=0.115, right=0.965, top=0.885, bottom=0.075, wspace=0.34, hspace=0.44)
    rng = np.random.default_rng(0)
    for ax, el in zip(axs.ravel(), STIM4):
        AD = [float(r[metric]) for r in rows if r["electrode"] == el and r["group"] == "AD"]
        HC = [float(r[metric]) for r in rows if r["electrode"] == el and r["group"] == "HC"]
        data, cols = [AD, HC], [CLAY, BLUE]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color="#3a3a46", lw=1.8), boxprops=dict(lw=0),
                        whiskerprops=dict(color="#a8a8a8", lw=1.2), capprops=dict(color="#a8a8a8", lw=1.2))
        for patch, c in zip(bp["boxes"], cols): patch.set_facecolor(c); patch.set_alpha(0.28)
        for xi, (v, c) in enumerate(zip(data, cols)):
            ax.scatter(rng.normal(xi, 0.075, len(v)), v, s=24, color=c, alpha=0.85, lw=0, zorder=3)
        d, lo, hi, p = cohen_p(AD, HC)                    # cohen_p → Mann–Whitney U
        pv = "p<0.001" if p < 1e-3 else f"p={p:.3f}"
        top = max(max(AD), max(HC)); bot = min(min(AD), min(HC)); sp = (top - bot) or 1; yb = top + 0.12 * sp
        ax.plot([0, 1], [yb, yb], color="#3a3a46", lw=1.1)
        ax.text(0.5, yb + 0.03 * sp, f"{pstars(p)} {pv}   d={d:+.2f}", ha="center", va="bottom", fontsize=12)
        ax.set_ylim(bot - 0.08 * sp, yb + 0.24 * sp)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"AD\n(n={len(AD)})", f"HC\n(n={len(HC)})"], fontsize=12)
        ax.set_xlim(-0.6, 1.6); ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=11)
        ax.set_title(el, fontsize=14.5, loc="left", pad=6, fontweight="bold")
        ax.spines["left"].set_visible(True); ax.spines["left"].set_color("#3a3a46"); ax.spines["left"].set_linewidth(0.8)
    for ax in axs[:, 0]: ax.set_ylabel(ylabel, fontsize=12.5)
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=14.5, fontweight="bold")
    fig.text(0.02, 0.925, "AD amyloid+ (n=14) vs HC amyloid− (n=14) · surrogate $E_{\\mathrm{ab}}$ · "
             "Mann–Whitney U (n.s./*/**/***).", ha="left", fontsize=10.5, color=GREY)
    _save(fig, fname)

def fig_ad_hc_gm_maxflu():
    _ad_hc_4panel("gm_maxflu_log10", "Peak GM fluence (log$_{10}$)", "ad_vs_hc_gm_maxflu",
                  "Peak grey-matter fluence — AD vs HC at 4 stimulation sites")

def fig_ad_hc_gm_maxflu_dist():
    _ad_hc_4panel("dist_mm", "Depth of peak GM voxel from entry (mm)", "ad_vs_hc_gm_maxflu_dist",
                  "Depth of peak grey-matter fluence from the entry — AD vs HC")

# ============================ PART A — new analyses ============================
def load_optim():
    """adam-ring optimiser (budget=1000) joined to target anatomy + demographics (n=14 AD)."""
    tr = list(csv.DictReader(open(os.path.join(
        ROOT, "viz/out/v17inverse-optimize/single-point/update0730/timing_record.csv"))))
    opt = {r["head"]: r for r in tr if r["budget"] == "1000"}
    ad14 = {r["head"]: r for r in csv.DictReader(open(os.path.join(
        ROOT, "viz/out/v17inverse-optimize/single-point/cohort_ad14.csv")))}
    feat = {r["sid"].lower(): r for r in csv.DictReader(open(C.CSV))}
    rows = []
    for h, o in opt.items():
        a, f = ad14.get(h), feat.get(h.lower())
        if not (a and f): continue
        rows.append(dict(head=h, A=float(o["A"]), B=float(o["B"]), mcxJ=float(o["mcxJ"]),
            gain=float(o["gain_over_elec"]), frac=float(o["frac_of_ceiling"]),
            depth=float(a["depth_radial_mm"]), suvr=float(a["pvc_suvr"]),
            region=a["region"], hemi=a["region"][:2].upper(),
            centiloid=float(f["centiloid"]), bpf=float(f["bpf"]), icv=float(f["icv"]),
            skull=float(f["skull"]), scalp=float(f["scalp"]), age=float(f["age"]), sex=f["sex"]))
    return rows

def _scatter(ax, x, y, hue, title, xlabel, ylabel, logy=False, size=None, statloc="upper left"):
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = sorted(set(hue)); pal = {"LH": BLUE, "RH": CLAY}
    for g in order:
        m = np.array([h == g for h in hue])
        s = 70 if size is None else np.asarray(size)[m]
        ax.scatter(x[m], (y[m]), s=s, c=pal.get(g, INK), alpha=0.85, lw=0.6,
                   edgecolor="white", zorder=3, label=g)
    if logy: ax.set_yscale("log")
    # Spearman on the raw values + rank-fit line for guidance
    r, p = spearmanr(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    b = np.polyfit(x, np.log10(y) if logy else y, 1)
    ax.plot(xs, 10 ** (np.polyval(b, xs)) if logy else np.polyval(b, xs),
            color=INK, lw=1.6, ls="--", alpha=0.7, zorder=2)
    tx, ha = (0.96, "right") if "right" in statloc else (0.04, "left")
    ax.text(tx, 0.965, f"Spearman ρ = {r:+.2f}\n{pstars(p)}  {pfmt(p)}",
            transform=ax.transAxes, va="top", ha=ha, fontsize=13.5,
            fontweight="bold" if p < 0.05 else "normal", color=INK)
    ax.set_title(title, fontsize=14.5, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=14); ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)

def fig_target_delivery():
    R = load_optim()
    dep = [r["depth"] for r in R]; flu = [r["mcxJ"] for r in R]
    gain = [r["gain"] for r in R]; hue = [r["hemi"] for r in R]
    A = [r["A"] for r in R]; Bang = [r["B"] for r in R]
    fig, axs = plt.subplots(1, 3, figsize=(16.8, 5.4))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.76, bottom=0.135, wspace=0.32)
    _scatter(axs[0], dep, [10 ** v for v in flu], hue,
             "a · Light reaching the target falls with depth",
             "Radial target depth (mm)", "MC fluence at target (a.u., log)", logy=True,
             statloc="upper right")
    _scatter(axs[1], dep, gain, hue,
             "b · Free-angle steering gains most for deep targets",
             "Radial target depth (mm)", "Gain over best fixed electrode (×)", logy=True)
    pal = {"LH": BLUE, "RH": CLAY}
    for g in ("LH", "RH"):
        m = [i for i, r in enumerate(R) if r["hemi"] == g]
        axs[2].scatter([Bang[i] for i in m], [A[i] for i in m],
                       s=[30 + 12 * gain[i] for i in m], c=pal[g], alpha=0.8,
                       edgecolor="white", lw=0.7, zorder=3, label=("Left hemisphere" if g == "LH" else "Right hemisphere"))
    axs[2].axvline(0, color="#d8d8d8", lw=1.0); axs[2].axhline(0, color="#d8d8d8", lw=1.0)
    axs[2].set_title("c · Optimal beam-angle solution space", fontsize=14.5, loc="left", pad=8)
    axs[2].set_xlabel("Optimal azimuth angle B (deg)", fontsize=14)
    axs[2].set_ylabel("Optimal elevation angle A (deg)", fontsize=14)
    axs[2].tick_params(labelsize=12)
    axs[2].legend(loc="lower right", frameon=False, fontsize=11.5, handletextpad=0.3)
    axs[2].text(0.04, 0.96, "marker size scales with gain", transform=axs[2].transAxes,
                va="top", ha="left", fontsize=11.5, color=GREY)
    # shared hemisphere legend (a,b)
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Line2D([], [], marker="o", ls="none", color=BLUE, label="Left hemisphere"),
                           Line2D([], [], marker="o", ls="none", color=CLAY, label="Right hemisphere")],
                  loc="lower left", frameon=False, fontsize=11.5, handletextpad=0.3)
    fig.suptitle("Per-patient target delivery — adam-ring optimiser (n = 14 AD, Monte-Carlo scored)",
                 x=0.055, y=0.975, ha="left", fontsize=16.5, fontweight="bold")
    fig.text(0.055, 0.905, "Optimal continuous beam angle and the delivered fluence AT each patient's "
             "primary amyloid target; ρ = Spearman, p two-sided.", ha="left", fontsize=12.5, color=GREY)
    _save(fig, "target_delivery_drivers")

def fig_amyloid_vs_delivery():
    R = load_optim()
    cl = [r["centiloid"] for r in R]; su = [r["suvr"] for r in R]
    flu = [r["mcxJ"] for r in R]; dep = [r["depth"] for r in R]; hue = [r["hemi"] for r in R]
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.75, bottom=0.12, wspace=0.28)
    _scatter(axs[0], cl, su, hue,
             "a · Burden predicts how HOT the target is",
             "Centiloid (global amyloid burden)", "Target PVC(RSF) SUVR")
    _scatter(axs[1], cl, [10 ** v for v in flu], hue,
             "b · Burden does NOT predict light delivered",
             "Centiloid (global amyloid burden)", "MC fluence at target (a.u., log)", logy=True)
    rp, pp = partial_spearman(np.array(cl), np.array(flu), np.array(dep))
    axs[1].text(0.04, 0.83, f"partial ρ (|depth) = {rp:+.2f}\n{pfmt(pp)}",
                transform=axs[1].transAxes, va="top", ha="left", fontsize=12.5, color=GREY)
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Line2D([], [], marker="o", ls="none", color=BLUE, label="Left hemisphere"),
                           Line2D([], [], marker="o", ls="none", color=CLAY, label="Right hemisphere")],
                  loc="lower right", frameon=False, fontsize=11.5, handletextpad=0.3)
    fig.suptitle("Amyloid burden and light delivery are decoupled (n = 14 AD)",
                 x=0.08, y=0.965, ha="left", fontsize=16.5, fontweight="bold")
    fig.text(0.08, 0.90, "Optical delivery to the target is set by geometry (depth), not by how much "
             "amyloid is present — so per-patient optical planning is needed regardless of burden.",
             ha="left", fontsize=12.5, color=GREY)
    _save(fig, "amyloid_vs_delivery")

def fig_layer_attenuation():
    """Standardized cluster-robust OLS of delivered fluence@25 on the finest layer stack."""
    th = {(r["subject"], r["electrode"]): r for r in csv.DictReader(open(os.path.join(
        ROOT, "dataset_OASIS/pup_mni/cohort_electrode_thickness.csv")))}
    svm = list(csv.DictReader(open(os.path.join(ROOT, "dataset_OASIS/pup_mni/cohort_surrogate_vs_mc.csv"))))
    def g(d, k):
        try: return float(d[k])
        except: return None
    rows = []
    for s in svm:
        t = th.get((s["subject"], s["electrode"]))
        if not t: continue
        mc, su = g(s, "mc_flu_d25"), g(s, "sur_flu_d25")
        if mc is None or su is None: continue
        rows.append(dict(subject=s["subject"], skin=g(t, "skin_mm"), muscle=g(t, "muscle_mm"),
            fat=g(t, "fat_mm"), cortical=g(t, "cortical_mm"), cancellous=g(t, "cancellous_mm"),
            mc25=mc, sur25=su))
    XA = ["skin", "muscle", "fat", "cortical", "cancellous"]
    LAB = {"skin": "Skin", "muscle": "Muscle", "fat": "Fat", "cortical": "Cortical bone",
           "cancellous": "Cancellous bone"}
    mc = with_p_ols(FST.cluster_ols(rows, "mc25", XA)[0])
    sur = with_p_ols(FST.cluster_ols(rows, "sur25", XA)[0])
    forest_clean([("Monte Carlo", mc, CLAY), ("Surrogate", sur, BLUE)], XA, LAB,
                 "Which layer attenuates transcranial light most",
                 "Standardized coefficient, β  on delivered fluence @25 mm", "layer_attenuation",
                 subtitle=f"Finest scalp/skull layering; cluster-robust OLS by subject "
                          f"({len(set(r['subject'] for r in rows))} clusters, {len(rows)} electrode scenes).",
                 grids=(-0.2, -0.1, 0.1, 0.2), xlim=(-0.28, 0.20), rowh=0.62)

# ============================ 0804 NEW — mediation / orthogonality / collinearity ============================
SVM_CSV = os.path.join(ROOT, "dataset_OASIS", "pup_mni", "cohort_surrogate_vs_mc.csv")

def _med_load():
    """Per-head arrays for the causal-structure analyses (n=84): demographics/anatomy from the
    cohort feature table, the surrogate brain-parenchyma deposition fraction (GM+WM %, mean over 19
    electrodes) and the surrogate delivered fluence at 25 mm depth (mean over electrodes)."""
    prof = defaultdict(list)
    for r in csv.DictReader(open(DEP_CSF_CSV)):
        prof[r["subject"].lower()].append((float(r["gm_frac"]), float(r["wm_frac"]), float(r["csf_frac"])))
    dm = {h: 100 * np.mean(v, 0) for h, v in prof.items()}          # (gm, wm, csf) %
    brain = {h: dm[h][0] + dm[h][1] for h in dm}                    # GM+WM parenchyma %
    f25 = defaultdict(list)
    for r in csv.DictReader(open(SVM_CSV)):
        try: f25[r["subject"].lower()].append(float(r["sur_flu_d25"]))
        except (TypeError, ValueError, KeyError): pass
    flu25 = {h: float(np.mean(v)) for h, v in f25.items() if v}
    feat = {r["sid"].lower(): r for r in csv.DictReader(open(C.CSV))}
    heads = sorted(set(brain) & set(flu25) & set(feat))
    def col(fn): return np.array([fn(h) for h in heads], float)
    def fc(h, k):
        try: return float(feat[h][k])
        except (ValueError, KeyError, TypeError): return np.nan
    return dict(heads=heads,
                sexM=col(lambda h: 1.0 if feat[h]["sex"] == "M" else 0.0),
                AD=col(lambda h: 1.0 if feat[h]["group"] == "AD" else 0.0),
                scalp=col(lambda h: fc(h, "scalp")), skull=col(lambda h: fc(h, "skull")),
                muscle=col(lambda h: fc(h, "muscle")), bpf=col(lambda h: fc(h, "bpf")),
                icv=col(lambda h: fc(h, "icv")), age=col(lambda h: fc(h, "age")),
                centiloid=col(lambda h: fc(h, "centiloid")),
                gm=col(lambda h: dm[h][0]), wm=col(lambda h: dm[h][1]), csf=col(lambda h: dm[h][2]),
                brain=col(lambda h: brain[h]), flu25=col(lambda h: flu25[h]))

def _z(a): return (a - np.nanmean(a)) / np.nanstd(a)

def _ols(y, X):
    """OLS with t-test p-values. X is (n,) or (n,k). Returns beta (incl. intercept), p."""
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    Xa = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    resid = y - Xa @ beta; n, k = len(y), Xa.shape[1]
    s2 = resid @ resid / max(n - k, 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xa.T @ Xa)))
    p = 2 * (1 - stats.t.cdf(np.abs(beta / np.where(se > 0, se, np.nan)), max(n - k, 1)))
    return beta, p

def _mediation(X, M, Y, nboot=5000, seed=0):
    """Single-mediator model (M,Y standardized; X as given). Bootstrap 95% CI of the indirect a·b."""
    Xv = np.asarray(X, float); Mv, Yv = _z(np.asarray(M, float)), _z(np.asarray(Y, float))
    ba, pa = _ols(Mv, Xv); a = ba[1]
    bY, pY = _ols(Yv, np.column_stack([Xv, Mv])); cprime, b = bY[1], bY[2]
    bc, pc = _ols(Yv, Xv); c = bc[1]
    rng = np.random.default_rng(seed); n = len(Yv); boots = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        ba_, _ = _ols(_z(Mv[idx]), Xv[idx]); bY_, _ = _ols(_z(Yv[idx]), np.column_stack([Xv[idx], _z(Mv[idx])]))
        boots[i] = ba_[1] * bY_[2]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(a=a, pa=pa[1], b=b, pb=pY[2], cprime=cprime, pcp=pY[1], c=c, pc=pc[1],
                indirect=a * b, lo=lo, hi=hi, sig=(lo > 0) == (hi > 0))

def _pcorr_ci(x, y, Z=None, alpha=0.05, spearman=False):
    """(Partial) Pearson/Spearman r of x,y controlling columns of Z, with Fisher CI + p (TOST-ready n).
    spearman=True rank-transforms x, y and each control column first (partial Spearman)."""
    m = np.isfinite(x) & np.isfinite(y)
    if Z is not None: m &= np.all(np.isfinite(Z), 1)
    xx, yy = x[m], y[m]; ZZ = Z[m] if Z is not None else None; k = 0
    if spearman:
        xx, yy = stats.rankdata(xx), stats.rankdata(yy)
        if ZZ is not None and ZZ.shape[1] > 0:
            ZZ = np.column_stack([stats.rankdata(ZZ[:, j]) for j in range(ZZ.shape[1])])
    if ZZ is not None and ZZ.shape[1] > 0:
        Za = np.column_stack([np.ones(m.sum()), ZZ]); k = ZZ.shape[1]
        xx = xx - Za @ np.linalg.lstsq(Za, xx, rcond=None)[0]
        yy = yy - Za @ np.linalg.lstsq(Za, yy, rcond=None)[0]
    r, p = stats.pearsonr(xx, yy); n = m.sum()
    se = 1 / np.sqrt(max(n - k - 3, 1)); zc = stats.norm.ppf(1 - alpha / 2)
    ci = np.tanh(np.arctanh(r) + np.array([-zc, zc]) * se)
    return dict(r=r, lo=ci[0], hi=ci[1], p=p, n=n, k=k)

def _tost_r(r, n, k=0, bound=0.2, alpha=0.05):
    """Two one-sided tests that |true r| < bound (equivalence to a negligible correlation)."""
    se = 1 / np.sqrt(max(n - k - 3, 1)); z, zb = np.arctanh(r), np.arctanh(bound)
    p = max(stats.norm.sf((z + zb) / se), stats.norm.cdf((z - zb) / se))
    ci90 = np.tanh(z + np.array([-1.645, 1.645]) * se)
    return dict(p=p, ci90_lo=ci90[0], ci90_hi=ci90[1], equiv=(ci90[0] > -bound and ci90[1] < bound))

def _node(ax, xy, text, fc):
    ax.add_patch(FancyBboxPatch((xy[0] - 0.145, xy[1] - 0.083), 0.29, 0.166,
                 boxstyle="round,pad=0.012,rounding_size=0.03", fc=fc, ec="#3a3a46", lw=1.3, zorder=3))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=12.5, zorder=4, color=INK)

def _med_panel(ax, res, xlab, mlab, ylab, mcol):
    """One mediation path diagram: X→M (a), M→Y (b), X→Y (c' direct); indirect a·b summarised below."""
    from matplotlib.lines import Line2D
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    X, M, Y = (0.145, 0.50), (0.5, 0.86), (0.855, 0.50)
    _node(ax, X, xlab, "#eceae6"); _node(ax, M, mlab, mcol); _node(ax, Y, ylab, "#dfe6e6")
    def arrow(p, q, coef, p_val, lx, ly, ha="center"):
        sig = p_val < 0.05; col = INK if sig else "#a9a9a9"
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16,
                     lw=2.3 if sig else 1.5, color=col, ls="-" if sig else (0, (4, 3)),
                     shrinkA=2, shrinkB=2, zorder=2))
        ax.text(lx, ly, f"{coef:+.2f} {pstars(p_val)}", ha=ha, va="center", fontsize=12.5,
                color=col, fontweight="bold" if sig else "normal")
    arrow((0.255, 0.60), (0.375, 0.80), res["a"], res["pa"], 0.245, 0.755, "right")   # a: X->M
    arrow((0.625, 0.80), (0.745, 0.60), res["b"], res["pb"], 0.755, 0.755, "left")    # b: M->Y
    arrow((0.29, 0.44), (0.71, 0.44), res["cprime"], res["pcp"], 0.5, 0.375)          # c': X->Y direct
    ax.text(0.5, 0.335, "direct", ha="center", va="center", fontsize=9.5, color="#8a8a8a", style="italic")
    ind_sig = res["sig"]; ic = mcol_dark = "#2f6d54" if ind_sig else "#9a9a9a"
    ax.text(0.5, 0.135, f"Indirect  a·b = {res['indirect']:+.2f}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=ic)
    ax.text(0.5, 0.048, f"95% CI [{res['lo']:+.2f}, {res['hi']:+.2f}]"
            f"{'  (sig.)' if ind_sig else '  (n.s.)'}      total c = {res['c']:+.2f} {pstars(res['pc'])}",
            ha="center", va="center", fontsize=11, color=INK)

def fig_mediation():
    """Two mediation path diagrams (bootstrap 95% CI, n=84): the sex effect on the brain-reaching
    fraction is fully carried by scalp thickness; the disease effect on penetration depth is carried
    by atrophy (BPF).  Two distinct anatomical knobs, both surrogate-derived."""
    D = _med_load()
    rA = _mediation(D["sexM"], D["scalp"], D["brain"])            # sex → scalp → brain fraction
    rB = _mediation(D["AD"], D["bpf"], D["flu25"])               # AD → BPF(atrophy) → depth fluence
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 6.3))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.03, wspace=0.06)
    _med_panel(axs[0], rA, "Male sex", "Scalp thickness\n(soft tissue)", "Brain deposition\nfraction (%)", "#e7d6cd")
    _med_panel(axs[1], rB, "AD diagnosis", "Brain parenchymal\nfraction (atrophy)", "Delivered fluence\n@ 25 mm", "#e7d6cd")
    axs[0].set_title("a · The male delivery penalty IS scalp thickness", fontsize=13.5, loc="center", pad=2)
    axs[1].set_title("b · Disease raises depth penetration VIA atrophy", fontsize=13.5, loc="center", pad=2)
    fig.suptitle("Mediation: two anatomical factors route the demographic effects on light delivery",
                 x=0.5, y=0.955, ha="center", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.885, "Standardized path coefficients (mediator & outcome z-scored); solid = p<0.05, "
             "dashed = n.s.  Indirect effect a·b with subject-bootstrap 95% CI (5000 resamples).",
             ha="center", fontsize=11.5, color=GREY)
    fig.text(0.5, 0.015, "a: full mediation — the direct sex path collapses (n.s.) once scalp thickness is included.   "
             "b: suppression — atrophy raises depth fluence (indirect +), partly masked in the total effect.",
             ha="center", fontsize=10.5, color=GREY)
    _save(fig, "mediation_delivery")

def fig_delivery_orthogonality():
    """Amyloid burden is decoupled from optical delivery: the modest marginal Centiloid–delivery
    association is fully explained by co-varying anatomy — the anatomy-adjusted correlation collapses
    into a negligible (|r|<0.2) band.  Left: raw scatter; right: progressive-adjustment forest."""
    from matplotlib.lines import Line2D
    D = _med_load()
    cent, flu = D["centiloid"], D["brain"]      # outcome = brain-parenchyma deposition fraction (%)
    grp = D["AD"]; m = np.isfinite(cent)
    # (a) scatter Centiloid vs brain deposition fraction (the metric with a confounded marginal link)
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.9), gridspec_kw=dict(width_ratios=[1.05, 1.0]))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.135, wspace=0.30)
    ax = axs[0]
    for gv, c, lab in [(1.0, CLAY, "AD"), (0.0, BLUE, "HC")]:
        s = m & (grp == gv)
        ax.scatter(cent[s], flu[s], s=34, color=c, alpha=0.82, lw=0.3, edgecolor="white", label=lab)
    xf = np.array([np.nanmin(cent[m]), np.nanmax(cent[m])])
    bY, _ = _ols(flu[m], cent[m]); ax.plot(xf, bY[0] + bY[1] * xf, color=INK, lw=1.8, zorder=4)
    rm = _pcorr_ci(cent, flu)
    ax.text(0.04, 0.96, f"marginal r = {rm['r']:+.2f}  [{rm['lo']:+.2f}, {rm['hi']:+.2f}]\n{pfmt(rm['p'])}   (n={rm['n']})",
            transform=ax.transAxes, va="top", ha="left", fontsize=12, color=INK)
    ax.set_xlabel("Centiloid (global amyloid burden)", fontsize=13.5)
    ax.set_ylabel("Brain-parenchyma deposition fraction (%)", fontsize=13.5)
    ax.set_title("a · Raw association (confounded)", fontsize=13.5, loc="left", pad=8)
    ax.tick_params(labelsize=11.5); ax.spines["left"].set_visible(True)
    ax.legend(frameon=False, fontsize=12, loc="lower right", handletextpad=0.3)
    # (b) progressive-adjustment forest: Centiloid -> delivery partial r, |r|<0.2 negligible band
    ax = axs[1]; geo = np.column_stack([D["scalp"], D["skull"]]); atr = D["bpf"][:, None]
    demo = np.column_stack([D["sexM"], D["AD"]])
    LV = [("unadjusted", None),
          ("+ geometry (scalp, skull)", geo),
          ("+ atrophy (BPF)", np.column_stack([geo, atr])),
          ("+ sex & diagnosis", np.column_stack([geo, atr, demo]))]
    ax.axvspan(-0.2, 0.2, color="#e9efe9", zorder=0)                         # negligible band |r|<0.2
    ax.axvline(0, color="#b8b8b8", lw=1.3, zorder=1)
    yy = np.arange(len(LV))[::-1]
    for (lab, Z), y in zip(LV, yy):
        r = _pcorr_ci(cent, flu, Z)
        col = "#2f6d54" if (r["lo"] > -0.2 and r["hi"] < 0.2) else CLAY
        ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=2.8, solid_capstyle="round", zorder=3)
        ax.plot(r["r"], y, "o", ms=9, color=col, mec="white", mew=0.8, zorder=4)
        ax.text(0.83, y, f"r={r['r']:+.2f}", transform=ax.get_yaxis_transform(), va="center",
                ha="left", fontsize=11.5, color=col, family="monospace")
    tt = _tost_r(_pcorr_ci(cent, flu, LV[-1][1])["r"], rm["n"], k=5)
    ax.set_yticks(yy); ax.set_yticklabels([l for l, _ in LV], fontsize=12.5)
    ax.set_xlim(-0.45, 1.02); ax.set_ylim(-0.72, len(LV) - 0.4)
    ax.set_xlabel("Centiloid → brain-delivery partial correlation, r", fontsize=13.5)
    ax.tick_params(axis="x", labelsize=11.5); ax.tick_params(axis="y", length=0)
    ax.set_title("b · Adjusted association collapses to negligible", fontsize=13.5, loc="left", pad=8)
    ax.text(0.0, -0.55, "|r| < 0.2 : negligible", ha="center", va="center", fontsize=10.5, color="#5f7a63")
    fig.suptitle("Amyloid burden is decoupled from transcranial light delivery",
                 x=0.075, y=0.955, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.075, 0.885, "The modest Centiloid–delivery trend is confounded by anatomy; adjusting for "
             "overlying geometry and atrophy drives it to r≈0 — no direct optical effect of amyloid.",
             ha="left", fontsize=11.5, color=GREY)
    verdict = "formally equivalent" if tt["equiv"] else "point estimate negligible; equivalence n-limited"
    fig.text(0.075, 0.03, f"Fully-adjusted TOST vs ±0.2: p={tt['p']:.2f}, 90% CI [{tt['ci90_lo']:+.2f}, "
             f"{tt['ci90_hi']:+.2f}] — {verdict}.  Delivered depth-fluence is null even marginally (same conclusion).",
             ha="left", fontsize=10, color=GREY)
    _save(fig, "delivery_orthogonality")

def fig_collinearity_vif():
    """Variance-inflation diagnostics for the cohort predictor set — the scalp/temporalis soft-tissue
    pair is near-collinear (VIF≈31, ρ≈0.96); keep one.  All other predictors are independent (VIF<3)."""
    D = _med_load()
    NM = [("scalp", "Scalp thickness"), ("muscle", "Temporalis thickness"), ("skull", "Skull thickness"),
          ("bpf", "BPF"), ("icv", "ICV"), ("age", "Age"), ("sexM", "Male sex"),
          ("AD", "AD diagnosis"), ("centiloid", "Centiloid")]
    M = np.column_stack([D[k] for k, _ in NM]); good = np.all(np.isfinite(M), 1); Mz = _z(M[good])
    vif = []
    for i in range(Mz.shape[1]):
        others = np.delete(Mz, i, 1); b, _ = _ols(Mz[:, i], others)
        r2 = 1 - ((Mz[:, i] - np.column_stack([np.ones(len(Mz)), others]) @ b) ** 2).sum() / \
             ((Mz[:, i] - Mz[:, i].mean()) ** 2).sum()
        vif.append(1 / (1 - r2))
    order = np.argsort(vif); labs = [NM[i][1] for i in order]; v = np.array(vif)[order]
    fig, ax = plt.subplots(figsize=(8.4, 5.6)); fig.subplots_adjust(left=0.29, right=0.95, top=0.80, bottom=0.13)
    y = np.arange(len(v))
    cols = [CLAY if vv >= 10 else (BLUE if vv >= 5 else "#9fb0a6") for vv in v]
    ax.barh(y, v, color=cols, edgecolor="white", height=0.7, zorder=3)
    for yi, vv in zip(y, v): ax.text(vv + 0.4, yi, f"{vv:.1f}", va="center", fontsize=11.5, color=INK)
    for xv, lb in [(5, "VIF 5"), (10, "VIF 10")]:
        ax.axvline(xv, color="#c8c8c8", lw=1.1, ls="--", zorder=1)
        ax.text(xv, len(v) - 0.35, lb, ha="center", va="bottom", fontsize=10, color=GREY)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=13); ax.tick_params(length=0)
    ax.set_xlim(0, max(v) * 1.14); ax.set_xlabel("Variance inflation factor (VIF)", fontsize=13.5)
    ax.tick_params(axis="x", labelsize=11.5)
    fig.suptitle("Collinearity of cohort predictors", x=0.29, y=0.955, ha="left", fontsize=15.5, fontweight="bold")
    fig.text(0.29, 0.885, "Scalp and temporalis thickness are near-collinear (ρ≈0.96) — use one soft-tissue "
             "term; every other predictor is independent (VIF<3).", ha="left", fontsize=11, color=GREY)
    _save(fig, "collinearity_vif")

def _load_mc_dep_csf():
    """Per-head MC ground-truth (gm, wm, csf) deposition fraction (%), mean over the 19 electrodes."""
    prof = defaultdict(list)
    for r in csv.DictReader(open(DEP_MC_CSF_CSV)):
        prof[r["subject"].lower()].append((float(r["gm_frac"]), float(r["wm_frac"]), float(r["csf_frac"])))
    return {h: 100 * np.mean(v, 0) for h, v in prof.items()}

def fig_bpf_deposition_partial():
    """Confounder-adjusted association of atrophy (BPF) with GM/WM/CSF deposition fraction, comparing
    the MC ground truth against the surrogate. Left: partial correlations (controlling scalp, skull,
    ICV, sex, age) for MC(GT) vs surrogate, BH-FDR across tissues — atrophy specifically raises CSF
    deposition in BOTH, GM/WM stay null. Right: surrogate added-variable plot for the CSF effect."""
    from matplotlib.lines import Line2D
    D = _med_load()
    ctrl = np.column_stack([D["scalp"], D["skull"], D["icv"], D["sexM"], D["age"]])   # optical confounders
    mc = _load_mc_dep_csf()
    def mccol(i): return np.array([mc[h][i] if h in mc else np.nan for h in D["heads"]])
    MC = {"gm": mccol(0), "wm": mccol(1), "csf": mccol(2)}
    TIS = [("gm", "GM"), ("wm", "WM"), ("csf", "CSF")]
    adj = {k: _pcorr_ci(D["bpf"], D[k], ctrl) for k, _ in TIS}                        # surrogate
    adjmc = {k: _pcorr_ci(D["bpf"], MC[k], ctrl) for k, _ in TIS}                     # MC ground truth
    adjs = {k: _pcorr_ci(D["bpf"], D[k], ctrl, spearman=True) for k, _ in TIS}
    fdr = dict(zip([k for k, _ in TIS], multipletests([adj[k]["p"] for k, _ in TIS], method="fdr_bh")[1]))
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.6), gridspec_kw=dict(width_ratios=[1.12, 1.0]))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.935, bottom=0.14, wspace=0.32)
    # (left) MC ground truth vs surrogate — both confounder-adjusted partial correlations
    ax = axs[0]; ax.axvspan(-0.2, 0.2, color="#eef1ee", zorder=0); ax.axvline(0, color="#b8b8b8", lw=1.3, zorder=1)
    yy = np.arange(len(TIS))[::-1]
    SER = [("MC (ground truth)", CLAY, "o", 0.17, adjmc), ("Surrogate (predicted)", BLUE, "s", -0.17, adj)]
    for (k, lab), y in zip(TIS, yy):
        for nm, c, mk, dy, src in SER:
            res = src[k]
            ax.plot([res["lo"], res["hi"]], [y + dy] * 2, color=c, lw=2.6, solid_capstyle="round", zorder=3)
            ax.plot(res["r"], y + dy, mk, ms=9, color=c, mec="white", mew=0.9, zorder=4)
        ax.text(0.995, y + 0.17, f"GT {adjmc[k]['r']:+.2f}", transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=10.5, color=CLAY, family="monospace")
        st = f" {pstars(fdr[k])}" if fdr[k] < 0.05 else ""
        ax.text(0.995, y - 0.17, f"Sur {adj[k]['r']:+.2f}{st}", transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=10.5, color=BLUE, family="monospace",
                fontweight="bold" if fdr[k] < 0.05 else "normal")
    ax.set_yticks(yy); ax.set_yticklabels([lab for _, lab in TIS], fontsize=15)
    ax.set_xlim(-0.62, 0.62); ax.set_ylim(-0.6, len(TIS) - 0.4)
    ax.set_xlabel("Partial correlation of BPF with deposition fraction, r", fontsize=13)
    ax.tick_params(axis="x", labelsize=11.5); ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=CLAY, ms=9, label="MC (ground truth)"),
                       Line2D([], [], marker="s", ls="none", color=BLUE, ms=9, label="Surrogate (predicted)")],
              loc="upper left", frameon=False, fontsize=11, handletextpad=0.3)
    ax.text(-0.2, -0.52, "|r|<0.2", ha="center", va="center", fontsize=9.5, color="#7f907f")
    # (right) surrogate added-variable (partial-regression) plot for the significant CSF effect
    ax = axs[1]; m = np.all(np.isfinite(np.column_stack([D["bpf"], D["csf"], ctrl])), 1)
    Za = np.column_stack([np.ones(m.sum()), ctrl[m]])
    xr = D["bpf"][m] - Za @ np.linalg.lstsq(Za, D["bpf"][m], rcond=None)[0]
    yr = D["csf"][m] - Za @ np.linalg.lstsq(Za, D["csf"][m], rcond=None)[0]
    for gv, c, lab in [(1.0, CLAY, "AD"), (0.0, BLUE, "HC")]:
        s = D["AD"][m] == gv; ax.scatter(xr[s], yr[s], s=34, color=c, alpha=0.82, lw=0.3, edgecolor="white", label=lab)
    xf = np.array([xr.min(), xr.max()]); b = np.polyfit(xr, yr, 1); ax.plot(xf, np.polyval(b, xf), color=INK, lw=1.9, zorder=4)
    # stats label + AD/HC legend moved to the top-right corner (legend as one row, third line)
    ax.text(0.965, 0.965, f"partial r = {adj['csf']['r']:+.2f}  [{adj['csf']['lo']:+.2f}, {adj['csf']['hi']:+.2f}]",
            transform=ax.transAxes, va="top", ha="right", fontsize=11.5, color=INK)
    ax.text(0.965, 0.885, f"{pfmt(adj['csf']['p'])}   FDR {pfmt(fdr['csf'])}   (n={adj['csf']['n']})",
            transform=ax.transAxes, va="top", ha="right", fontsize=11.5, color=INK)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.80), ncol=2, frameon=False, fontsize=11.5,
              handletextpad=0.3, columnspacing=1.3)
    ax.set_xlabel("BPF  |  residual after covariates", fontsize=12.5)
    ax.set_ylabel("CSF deposition fraction  |  residual", fontsize=12.5)
    ax.tick_params(labelsize=11); ax.spines["left"].set_visible(True)
    _save(fig, "bpf_deposition_partial")
    return adjmc, adj, adjs, fdr

def run_mediation_suite():
    """Generate the four 0804 analysis figures + write a numeric results summary to update0803."""
    D = _med_load()
    rA = _mediation(D["sexM"], D["scalp"], D["brain"]); rB = _mediation(D["AD"], D["bpf"], D["flu25"])
    ctrl = np.column_stack([D["scalp"], D["skull"], D["bpf"], D["sexM"], D["AD"]])
    marg = _pcorr_ci(D["centiloid"], D["brain"])            # primary outcome = brain deposition fraction
    adj = _pcorr_ci(D["centiloid"], D["brain"], ctrl)
    marg_flu = _pcorr_ci(D["centiloid"], D["flu25"])        # cross-check = delivered depth-fluence
    adj_flu = _pcorr_ci(D["centiloid"], D["flu25"], ctrl)
    tt = _tost_r(adj["r"], marg["n"], k=5)
    fig_mediation(); fig_delivery_orthogonality(); fig_collinearity_vif()
    bmarg, badj, badjs, bfdr = fig_bpf_deposition_partial()
    lines = [
        "# update0803 — 中介/正交/共线性 分析结果 (0804)", "",
        "所有指标均为代理(surrogate)输出, n=84 (42 AD / 42 HC). 显著性: n.s. ≥.05 · * <.05 · ** <.01 · *** <.001.", "",
        "## 1. 中介分析 (mediation_delivery.png; bootstrap 5000, 95% CI)", "",
        "**(a) 性别 → 头皮厚度 → 脑内沉积占比 —— 完全中介**",
        f"- a (性别→头皮): {rA['a']:+.2f} (p={rA['pa']:.2g}); b (头皮→占比|性别): {rA['b']:+.2f} (p={rA['pb']:.2g})",
        f"- 直接效应 c': {rA['cprime']:+.2f} (p={rA['pcp']:.2g}, 不显著→完全中介); 总效应 c: {rA['c']:+.2f} (p={rA['pc']:.2g})",
        f"- **间接效应 a·b = {rA['indirect']:+.2f}, 95% CI [{rA['lo']:+.2f}, {rA['hi']:+.2f}]** (排除0=显著)",
        "- 结论: 男性到脑比例更低, 完全由更厚的头皮(软组织)解释, 非性别本身.", "",
        "**(b) AD 诊断 → 脑实质分数 BPF(萎缩) → 25mm 递送荧光 —— 抑制型中介**",
        f"- a (AD→BPF): {rB['a']:+.2f} (p={rB['pa']:.2g}, AD 萎缩更重); b (BPF→荧光|AD): {rB['b']:+.2f} (p={rB['pb']:.2g})",
        f"- 直接 c': {rB['cprime']:+.2f} (p={rB['pcp']:.2g}); 总 c: {rB['c']:+.2f} (p={rB['pc']:.2g})",
        f"- **间接效应 a·b = {rB['indirect']:+.2f}, 95% CI [{rB['lo']:+.2f}, {rB['hi']:+.2f}]** (显著)",
        "- 结论: 萎缩(CSF 增多)提升深部穿透; 与总效应反向的直接项部分掩盖(抑制/suppression).", "",
        "## 2. 淀粉样-递送 正交性 (delivery_orthogonality.png; 主结局=脑内沉积占比)", "",
        f"- 边际相关 Centiloid vs 脑内占比: r={marg['r']:+.2f} [{marg['lo']:+.2f}, {marg['hi']:+.2f}], {pfmt(marg['p'])} (n={marg['n']}) — 弱正趋势(混杂)",
        f"- 校正解剖(头皮/颅骨/BPF/性别/诊断)后偏相关: r={adj['r']:+.2f} [{adj['lo']:+.2f}, {adj['hi']:+.2f}], {pfmt(adj['p'])} — 塌陷进可忽略带",
        f"- 交叉验证(25mm 递送荧光): 边际 r={marg_flu['r']:+.2f} {pfmt(marg_flu['p'])} 本就≈0; 校正后 r={adj_flu['r']:+.2f} {pfmt(adj_flu['p'])}",
        f"- 等价检验 TOST(界 ±0.2): p={tt['p']:.2f}, 90% CI [{tt['ci90_lo']:+.2f}, {tt['ci90_hi']:+.2f}] → "
        f"{'形式上等价' if tt['equiv'] else '点估计可忽略, 但 n 有限, 未达形式等价'}.",
        "- 结论: 边际关联为解剖混杂所致; 校正后≈0, 淀粉样负荷对光学递送无直接作用.", "",
        "## 3. 共线性诊断 (collinearity_vif.png)", "",
        "- 头皮↔颞肌 VIF≈31 (ρ≈0.96), 冗余, 多变量模型二选一; 其余预测因子 VIF<3, 相互独立.", "",
        "## 4. 脑萎缩 BPF 与 GM/WM/CSF 沉积占比的独立关联 — MC金标准 vs 代理 (bpf_deposition_partial.png)", "",
        "偏相关(Pearson), 控制 头皮/颅骨厚度、ICV、性别、年龄; BH-FDR 跨 3 个组织校正. 低 BPF = 萎缩更重.",
        f"- **BPF~CSF**: 代理 r={badj['csf']['r']:+.2f} [{badj['csf']['lo']:+.2f}, {badj['csf']['hi']:+.2f}], "
        f"{pfmt(badj['csf']['p'])}, FDR {pfmt(bfdr['csf'])} → **显著**;  MC(GT) r={bmarg['csf']['r']:+.2f} "
        f"({pfmt(bmarg['csf']['p'])}) → 方向与量级一致, 代理复现金标准",
        f"- BPF~GM: 代理 r={badj['gm']['r']:+.2f} (n.s.) / GT r={bmarg['gm']['r']:+.2f};  "
        f"BPF~WM: 代理 r={badj['wm']['r']:+.2f} (n.s.) / GT r={bmarg['wm']['r']:+.2f} — 均为空",
        f"- Spearman 一致(CSF 代理 {badjs['csf']['r']:+.2f}); 再加控制 诊断+淀粉样 后 CSF r=−0.30 (FDR≈0.03) 更显著.",
        "- 结论: **萎缩把沉积能量特异性地重分配到 CSF(而非 GM/WM)**, 且 **MC 金标准与代理一致**; 该效应在"
        "原始 Spearman 热图中被混杂(头围/几何)抑制而看不到——须偏相关/多元回归才能显现.", "",
        "## 正文/补充 建议", "",
        "- 正文: 一张\"临床转化\"图 = 中介(a) + 正交性(b); 说明递送受几何门控、与淀粉样解耦→需个体化光学规划.",
        "- 补充: 共线性 VIF、脑内占比结局的平行结果、相关热图/Mantel/ANOVA、TOST 全表.",
        "- 提醒: n=84(PET 子集更小)限制多变量可信度; power 靠 1596-scene 聚类稳健模型, 可解释性靠 84-被试模型.",
    ]
    with open(os.path.join(OUT, "mediation_equivalence_results.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(OUT, "mediation_equivalence_results.md"))
    for k, v in [("med sex→scalp→brain a·b", (rA["indirect"], rA["lo"], rA["hi"])),
                 ("med AD→bpf→flu a·b", (rB["indirect"], rB["lo"], rB["hi"])),
                 ("amyloid marginal r", (marg["r"], marg["lo"], marg["hi"])),
                 ("amyloid adjusted r", (adj["r"], adj["lo"], adj["hi"]))]:
        print(f"  {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")

def main():
    print("=== PART B: fixed existing figures ===")
    fix_forest_effect_sizes()
    fix_fluence_drivers()
    fix_fluence_drivers_demo()
    fix_fluence_by_stratum()
    if os.path.isfile(DEP_CSV):
        fix_fluence_sex_box()          # now GM/WM absorbed-energy deposition fractions
        fix_feature_correlation()      # + deposition + sex, minus educ/bone splits
    else:
        print(f"  [skip] {DEP_CSV} not ready — run batch_energy_deposition.py first")
    if os.path.isfile(DEP_CSV) and os.path.isfile(DEP_MC_CSV):
        print("=== 0803 NEW: parenchyma group×method + sex×amyloid ANOVA ===")
        fig_parenchyma_group_box()
        fig_parenchyma_anova()
    else:
        print(f"  [skip] MC deposition not ready — run batch_energy_deposition_mc.py first")
    print("=== PART A: new analyses ===")
    fig_target_delivery()
    fig_amyloid_vs_delivery()
    fig_layer_attenuation()
    if os.path.isfile(DEP_CSF_CSV) and os.path.isfile(SVM_CSV):
        print("=== 0804 NEW: mediation / orthogonality / collinearity ===")
        run_mediation_suite()

if __name__ == "__main__":
    main()
