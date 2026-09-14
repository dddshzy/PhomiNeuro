"""Relate absorbed-energy fractions to head anatomy and cohort variables.

Per-subject deposition is averaged over 19 electrodes. Scatter panels show
method-specific OLS fits; forest panels compare standardized unadjusted and
multivariable estimates. Centiloid-dependent models exclude missing values.
"""
import argparse, csv, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import matplotlib.font_manager as fm                                       # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import repro_config as RC                                               # noqa: E402
os.environ.setdefault("MPLCONFIGDIR", str(RC.RESULTS_DIR / "v18h3" / "_mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

for _f in ["/usr/share/fonts/truetype/LiberationSans-Regular.ttf",
           "/usr/share/fonts/truetype/LiberationSans-Bold.ttf"]:
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Liberation Sans", "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white",
                     "svg.fonttype": "none", "pdf.fonttype": 42})

PUP = str(RC.PUP_MNI_DIR)
FEAT = os.environ.get(
    "PHOMINEURO_COHORT_FEATURES",
    str(RC.OASIS_METADATA_DIR / "cohort_sex_features.csv"),
)
OUT = str(RC.RESULTS_DIR / "v18h3")
CL_POS = 26.0

# Deeper members of the project's clay/blue pair: same hue families as the group figures, but
# clearly separated, because here the two series are METHODS rather than cohort groups.
SER = [("su", "Ours", "#2B6CA3"), ("mc", "MCX", "#B4553C")]
INK, GREY = "#3a3a3a", "#7a7a7a"

TIS = [("gm", "GM"), ("wm", "WM"), ("csf", "CSF")]
XVAR = [("skull", "Skull thickness (mm)"),
        ("scalp", "Scalp thickness (mm)"),
        ("bpf", "Brain parenchymal fraction")]
# Physically motivated plus the demographics the table stratifies on. Education, MMSE and APOE are
# left out on purpose: they have no path to photon transport that is not already carried by BPF,
# atrophy and amyloid, and each extra column costs precision at n = 72.
PRED = [("centiloid", "Centiloid"), ("scalp", "Scalp thickness"), ("skull", "Skull thickness"),
        ("bpf", "BPF"), ("age", "Age"), ("sexM", "Sex (male)"), ("dxAD", "Diagnosis (AD)")]


def perhead(path, col):
    d = defaultdict(list)
    for r in csv.DictReader(open(path)):
        d[r["subject"].lower()].append(float(r[col]))
    return {h: float(np.mean(v)) * 100.0 for h, v in d.items()}


def load():
    dep = {}
    for tag, path in (("su", "cohort_energy_deposition_csf_v18h3.csv"),
                      ("mc", "cohort_energy_deposition_mc_csf.csv")):
        for k, _ in TIS:
            dep[(tag, k)] = perhead(os.path.join(PUP, path), f"{k}_frac")
    rows = []
    for r in csv.DictReader(open(FEAT)):
        h = r["sid"].lower()
        if h not in dep[("su", "gm")] or h not in dep[("mc", "gm")]:
            continue
        try:
            rec = dict(sid=h, sex=r["sex"], group=r["group"], age=float(r["age"]),
                       bpf=float(r["bpf"]), skull=float(r["skull"]), scalp=float(r["scalp"]))
        except (TypeError, ValueError):
            continue
        try:
            rec["centiloid"] = float(r["centiloid"])
        except (TypeError, ValueError):
            rec["centiloid"] = np.nan
        rec["sexM"] = 1.0 if r["sex"] == "M" else 0.0
        rec["dxAD"] = 1.0 if r["group"] == "AD" else 0.0
        for tag, _, _ in SER:
            for k, _ in TIS:
                rec[f"{k}_{tag}"] = dep[(tag, k)][h]
            rec[f"par_{tag}"] = rec[f"gm_{tag}"] + rec[f"wm_{tag}"]
        rows.append(rec)
    return pd.DataFrame(rows)


def fit_line(x, y, xs):
    """OLS fit plus the 95% confidence band of the MEAN response (not a prediction interval)."""
    m = sm.OLS(y, sm.add_constant(x)).fit()
    pr = m.get_prediction(sm.add_constant(xs))
    ci = pr.conf_int()
    return m, pr.predicted_mean, ci[:, 0], ci[:, 1]


def scatter(df, outdir, stem, size, dpi):
    from scipy.stats import pearsonr
    fig, axs = plt.subplots(len(TIS), len(XVAR), figsize=tuple(size))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.925, bottom=0.075, wspace=0.16, hspace=0.14)
    for i, (tk, tl) in enumerate(TIS):
        for j, (xk, xl) in enumerate(XVAR):
            ax = axs[i, j]
            xs = np.linspace(df[xk].min(), df[xk].max(), 100)
            for row, (tag, lab, col) in enumerate(SER):
                x, y = df[xk].values, df[f"{tk}_{tag}"].values
                ax.scatter(x, y, s=11, color=col, alpha=0.55, lw=0, zorder=2)
                m, yh, lo, hi = fit_line(x, y, xs)
                ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0, zorder=1)
                ax.plot(xs, yh, color=col, lw=1.9, zorder=3)
                r, p = pearsonr(x, y)
                # top-RIGHT, the two lines 60% closer together and lifted clear of the cloud
                ax.text(0.975, 0.992 - 0.046 * row, f"{lab}  r={r:+.2f}, {pfmt(p)}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
                        color=col, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{tl} absorbed-energy\nfraction (%)", fontsize=10.5)
            if i == len(TIS) - 1:
                ax.set_xlabel(xl, fontsize=10.5)
            ax.tick_params(labelsize=9.0)
            if i != len(TIS) - 1:
                ax.set_xticklabels([])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#9a9a9a"); ax.spines[s].set_linewidth(0.8)
    fig.legend(handles=[Line2D([], [], color=c, lw=2.2, marker="o", ms=5, label=l)
                        for _, l, c in SER],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=11.5,
               handlelength=1.9, columnspacing=2.2)
    fig.text(0.5, 0.012, f"n = {len(df)} subjects   ·   absorbed-energy fraction is the per-subject "
             "mean over the 19 electrodes   ·   scalp = skin + muscle   ·   "
             "line = OLS fit, band = 95% CI of the mean",
             ha="center", va="bottom", fontsize=8.6, color=GREY)
    save(fig, outdir, stem, dpi)


def pfmt(p):
    return "p<.001" if p < 1e-3 else f"p={p:.3f}".replace("0.", ".")


def zscore(s):
    return (s - s.mean()) / s.std(ddof=1)


def zebra(ax, ypos, lo="#F1F4F7", hi="#FFFFFF"):
    """Alternating row bands, so the panel reads as a table rather than as a plot with floating rows.
    Drawn at zorder 0 with the axis line and the zero reference above them."""
    for i, y in enumerate(ypos):
        ax.axhspan(y - 0.5, y + 0.5, color=lo if i % 2 == 0 else hi, lw=0, zorder=0)


def pstar(p):
    return "*" if p >= 0.01 else ("**" if p >= 1e-3 else "***")


def forest_models(df, ycol):
    """(unadjusted, adjusted) standardised betas with 95% CI, plus VIF and adjusted R2."""
    d = df.dropna(subset=["centiloid"]).copy()
    y = zscore(d[ycol])
    Z = pd.DataFrame({k: (zscore(d[k]) if k not in ("sexM", "dxAD") else d[k]) for k, _ in PRED})
    uni = {}
    for k, _ in PRED:
        m = sm.OLS(y, sm.add_constant(Z[[k]])).fit()
        uni[k] = (m.params[k], *m.conf_int().loc[k].values, m.pvalues[k])
    M = sm.OLS(y, sm.add_constant(Z)).fit()
    adj = {k: (M.params[k], *M.conf_int().loc[k].values, M.pvalues[k]) for k, _ in PRED}
    X = sm.add_constant(Z).values
    vif = {k: float(1.0 / (1.0 - sm.OLS(Z[k], sm.add_constant(Z.drop(columns=[k]))).fit().rsquared))
           for k, _ in PRED}
    return uni, adj, vif, M.rsquared_adj, len(d)


def forest(df, outdir, stem, size, dpi):
    res = {tag: forest_models(df, f"par_{tag}") for tag, _, _ in SER}
    n = res["su"][4]
    print(f"[v18h3] forest: n={n}  adj R2  Ours {res['su'][3]:.3f}  MCX {res['mc'][3]:.3f}")
    print("        VIF: " + "  ".join(f"{lab} {res['su'][2][k]:.2f}" for k, lab in PRED))

    fig, axs = plt.subplots(1, 2, figsize=tuple(size), sharey=True)
    fig.subplots_adjust(left=0.225, right=0.985, top=0.87, bottom=0.135, wspace=0.09)
    ypos = np.arange(len(PRED))[::-1]
    # "mutually adjusted" is the standard term for a multivariable model in which every estimate is
    # adjusted for all the other covariates simultaneously -- which is exactly what the panel shows.
    for ax, which, title in ((axs[0], 0, "Unadjusted (single-predictor models)"),
                             (axs[1], 1, "Mutually adjusted (7-covariate multivariable model)")):
        zebra(ax, ypos)
        ax.axvline(0, color="#b0b0b0", lw=1.0, zorder=1)
        for si, (tag, lab, col) in enumerate(SER):
            est = res[tag][which]
            off = 0.17 * (1 - 2 * si)
            for yi, (k, _) in zip(ypos, PRED):
                b, lo, hi, p = est[k]
                ax.plot([lo, hi], [yi + off] * 2, color=col, lw=1.7, solid_capstyle="round",
                        zorder=2)
                ax.plot(b, yi + off, "o", ms=6.2, color=col, mec="white", mew=1.0, zorder=3)
                if p < 0.05:
                    ax.text(hi + 0.03, yi + off, f"{pstar(p)} {pfmt(p)}", va="center", ha="left",
                            fontsize=9.0, color=col, fontweight="bold")
        ax.set_yticks(ypos); ax.set_ylim(-0.5, len(PRED) - 0.5)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=12.0, fontweight="bold", pad=6)
        ax.set_xlabel("Standardised β  (SD of deposition per SD of predictor)", fontsize=10.0)
        ax.tick_params(labelsize=10.5)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#9a9a9a"); ax.spines["bottom"].set_linewidth(0.8)
    axs[0].set_yticklabels([lab for _, lab in PRED], fontsize=11.5)
    fig.legend(handles=[Line2D([], [], color=c, lw=2.2, marker="o", ms=6, label=l)
                        for _, l, c in SER],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=11.5,
               handlelength=1.9, columnspacing=2.2)
    save(fig, outdir, stem, dpi)
    return res


def forest_tissue(df, outdir, stem, size, dpi):
    fig, axs = plt.subplots(1, 3, figsize=tuple(size), sharey=True)
    fig.subplots_adjust(left=0.165, right=0.99, top=0.87, bottom=0.135, wspace=0.09)
    ypos = np.arange(len(PRED))[::-1]
    for ax, (tk, tl) in zip(axs, TIS):
        zebra(ax, ypos)
        ax.axvline(0, color="#b0b0b0", lw=1.0, zorder=1)
        for si, (tag, lab, col) in enumerate(SER):
            _, adj, _, r2, n = forest_models(df, f"{tk}_{tag}")
            off = 0.17 * (1 - 2 * si)
            for yi, (k, _) in zip(ypos, PRED):
                b, lo, hi, p = adj[k]
                ax.plot([lo, hi], [yi + off] * 2, color=col, lw=1.7, solid_capstyle="round", zorder=2)
                ax.plot(b, yi + off, "o", ms=6.0, color=col, mec="white", mew=1.0, zorder=3)
                if p < 0.05:
                    ax.text(hi + 0.03, yi + off, f"{pstar(p)} {pfmt(p)}", va="center", ha="left",
                            fontsize=8.6, color=col, fontweight="bold")
        ax.set_yticks(ypos); ax.set_ylim(-0.5, len(PRED) - 0.5)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{tl} deposition", fontsize=12.0, fontweight="bold", pad=6)
        ax.set_xlabel("Standardised β", fontsize=10.0)
        ax.tick_params(labelsize=10.0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#9a9a9a"); ax.spines["bottom"].set_linewidth(0.8)
    axs[0].set_yticklabels([lab for _, lab in PRED], fontsize=11.0)
    fig.legend(handles=[Line2D([], [], color=c, lw=2.2, marker="o", ms=6, label=l)
                        for _, l, c in SER],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=11.5,
               handlelength=1.9, columnspacing=2.2)
    save(fig, outdir, stem, dpi)


def save(fig, outdir, stem, dpi):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(outdir, stem + ".png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["scatter", "forest", "tissue", "all"], default="all")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()

    df = load()
    print(f"[v18h3] loaded {len(df)} subjects; {df['centiloid'].notna().sum()} with a Centiloid")
    for tag, lab, _ in SER:
        s = df[f"par_{tag}"]
        print(f"        parenchyma {lab:4s} mean {s.mean():.3f}  sd {s.std(ddof=1):.3f}  "
              f"skew {s.skew():+.2f}")

    if a.which in ("scatter", "all"):
        scatter(df, a.outdir, "deposition_scatter3x3_v18h3", (10.0, 8.4), a.dpi)
    if a.which in ("forest", "all"):
        forest(df, a.outdir, "deposition_forest_v18h3", (11.0, 4.6), a.dpi)
    if a.which in ("tissue", "all"):
        forest_tissue(df, a.outdir, "deposition_forest_tissue_v18h3", (12.4, 4.6), a.dpi)


if __name__ == "__main__":
    sys.exit(main())
