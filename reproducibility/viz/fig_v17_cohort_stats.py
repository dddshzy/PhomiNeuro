#!/usr/bin/env python3
"""Two Nature-style single figures for the held-out AD/HC cohort statistical analysis:
  (1) effect-size forest plot  — Cohen's d (+95% CI) for AD-vs-HC and Male-vs-Female across all
      cohort features (the diagnosis-vs-sex double dissociation);
  (2) feature-correlation heatmap (Spearman) — collinearity structure for stratified modelling.
Style: Arial (metric-identical Liberation Sans), large type, Morandi palette, borderless
elements, white ground, compact. Saved to viz/out/v17viz4/ as png/pdf/svg.
"""
import os, csv, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# ---- Arial via metric-identical Liberation Sans ----
for _f in ["/usr/share/fonts/truetype/LiberationSans-Regular.ttf",
           "/usr/share/fonts/truetype/LiberationSans-Bold.ttf"]:
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none", "pdf.fonttype": 42,
})
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import repro_config as RC  # noqa: E402
CSV = os.environ.get(
    "PHOMINEURO_COHORT_FEATURES",
    str(RC.OASIS_METADATA_DIR / "cohort_sex_features.csv"),
)
OUT = os.environ.get("V17VIZ4_OUT", str(RC.RESULTS_DIR / "v17viz4"))
os.makedirs(OUT, exist_ok=True)

# Morandi palette
CLAY = "#C0897B"      # AD vs HC / positive
BLUE = "#8AA0AF"      # Male vs Female / negative
INK  = "#4a4a4a"
MOR = LinearSegmentedColormap.from_list("morandi_div", [BLUE, "#c9d2d6", "#f3efe8", "#e2cabf", CLAY])

# feature (column, display label), grouped: clinical/pathology → geometry → demographic
FEATS = [("cdrsb", "CDR–SB"), ("mmse", "MMSE"), ("centiloid", "Amyloid"),
         ("bpf", "BPF"), ("icv", "ICV"), ("scalp", "Scalp thickness"),
         ("muscle", "Temporalis thickness"), ("skull", "Skull thickness"),
         ("cortical", "Cortical bone"), ("cancellous", "Cancellous bone"),
         ("age", "Age"), ("educ", "Education")]

def load():
    rows = list(csv.DictReader(open(CSV)))
    def col(rs, k):
        return np.array([float(r[k]) for r in rs if r[k] not in ("", "None")])
    data = {}
    for key, _ in FEATS:
        data[key] = rows  # placeholder
    return rows

def vals(rows, key, filt):
    return np.array([float(r[key]) for r in rows if filt(r) and r[key] not in ("", "None")])

def cohen_ci(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2)/(na+nb-2))
    d = (a.mean()-b.mean())/sp if sp > 0 else 0.0
    se = np.sqrt((na+nb)/(na*nb) + d*d/(2*(na+nb)))
    return d, d-1.96*se, d+1.96*se

# ============================ Figure 1: forest plot ============================
def forest(rows):
    y = np.arange(len(FEATS))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    for g in (-0.2, -0.5, -0.8, 0.2, 0.5, 0.8):
        ax.axvline(g, color="#ececec", lw=1.0, zorder=0)
    ax.axvline(0, color="#b8b8b8", lw=1.4, zorder=1)
    series = [("AD vs HC", lambda r: r["group"] == "AD", lambda r: r["group"] == "healthy", CLAY, +0.17),
              ("Male vs Female", lambda r: r["sex"] == "M", lambda r: r["sex"] == "F", BLUE, -0.17)]
    for label, fa, fb, c, off in series:
        for yi, (key, _) in zip(y, FEATS):
            d, lo, hi = cohen_ci(vals(rows, key, fa), vals(rows, key, fb))
            ax.plot([lo, hi], [yi+off, yi+off], color=c, lw=2.6, solid_capstyle="round", zorder=3)
            ax.plot(d, yi+off, "o", ms=9, color=c, zorder=4)
        ax.plot([], [], "o", color=c, ms=10, label=label, lw=0)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in FEATS], fontsize=16)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(FEATS)-0.3)
    ax.set_xlabel("Standardized effect size, d", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.legend(loc="lower right", frameon=False, fontsize=15, handletextpad=0.3, borderaxespad=0.4)
    ax.text(ax.get_xlim()[1], len(FEATS)-0.5, "+ = AD>HC / Male>Female",
            ha="right", va="bottom", fontsize=12.5, color="#8a8a8a")
    fig.tight_layout()
    save(fig, "forest_effect_sizes")

# ============================ Figure 2: correlation heatmap ============================
def heatmap(rows):
    keys = [k for k, _ in FEATS]; labs = [l for _, l in FEATS]
    n = len(keys)
    M = np.full((n, n), np.nan); P = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = [], []
            for r in rows:
                if r[keys[i]] in ("", "None") or r[keys[j]] in ("", "None"): continue
                a.append(float(r[keys[i]])); b.append(float(r[keys[j]]))
            sr = stats.spearmanr(a, b); M[i, j] = sr.correlation; P[i, j] = sr.pvalue
    fig, ax = plt.subplots(figsize=(9.2, 8.4))
    im = ax.imshow(M, cmap=MOR, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            v = M[i, j]; signif = (i != j and P[i, j] < 0.05)
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "–."),
                    ha="center", va="center", fontsize=12.5,
                    fontweight="bold" if signif else "normal",
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(labs, fontsize=14)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Spearman ρ", fontsize=15); cb.ax.tick_params(labelsize=13)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.text(0.01, 0.005, "bold ρ: p<0.05 (n = 84 heads)", ha="left", fontsize=11.5, color="#8a8a8a")
    save(fig, "feature_correlation")

def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig); print("wrote", os.path.join(OUT, name + ".png"))

if __name__ == "__main__":
    rows = load()
    forest(rows)
    heatmap(rows)
