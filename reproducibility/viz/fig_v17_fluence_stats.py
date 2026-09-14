#!/usr/bin/env python3
"""Explanatory analysis: what anatomical factors drive the delivered fluence (MC truth AND V17
surrogate), and what degrades the surrogate's fidelity. Two Nature-style single figures matching
v17viz4 (Arial/Liberation, Morandi, borderless, white, compact).

  Fig 1 (forest): standardized OLS beta (±95% CI) of anatomy predictors on log10 fluence at 25 mm
     depth, for MC (clay) vs surrogate (blue). Overlap = the surrogate reproduces the physics.
  Fig 2 (forest): standardized beta of anatomy/geometry on the surrogate's whole-field fidelity R2
     and on the |surrogate-MC| depth error. Where the surrogate is less accurate.

Inputs: cohort_surrogate_vs_mc.csv (this run) + cohort_electrode_thickness.csv + cohort_bpf.csv +
cohort_stratification.csv.  Run after the sbatch job completes.
"""
import os, csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _f in ["/usr/share/fonts/truetype/LiberationSans-Regular.ttf",
           "/usr/share/fonts/truetype/LiberationSans-Bold.ttf"]:
    if os.path.exists(_f): fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none", "pdf.fonttype": 42})
CLAY, BLUE = "#C0897B", "#8AA0AF"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import repro_config as RC  # noqa: E402
PUP = str(RC.PUP_MNI_DIR)
MET = str(RC.OASIS_METADATA_DIR)
OUT = str(RC.RESULTS_DIR / "v17viz4"); os.makedirs(OUT, exist_ok=True)

def load_merged():
    """LONG format: one row per (scene, target depth). Columns depth/flu_mc/flu_sur/aerr + anatomy."""
    svm = {(r["subject"], r["electrode"]): r for r in
           csv.DictReader(open(os.path.join(PUP, "cohort_surrogate_vs_mc.csv")))}
    th = {(r["subject"], r["electrode"]): r for r in
          csv.DictReader(open(os.path.join(PUP, "cohort_electrode_thickness.csv")))}
    bpf = {r["subject"].lower(): r for r in csv.DictReader(open(os.path.join(MET, "cohort_bpf.csv")))}
    strat = {r["subject"].lower(): r for r in
             csv.DictReader(open(os.path.join(MET, "cohort_stratification.csv")))}
    def g(d, k):
        try: return float(d[k])
        except: return None
    rows = []
    for key, s in svm.items():
        t = th.get(key)
        if t is None or key[0] not in bpf: continue
        anat = dict(subject=key[0], electrode=key[1], flag=1 if t["flag"] else 0,
                    skull=g(t, "skull_mm"), scalp=(g(t,"skin_mm") or 0)+(g(t,"muscle_mm") or 0),
                    bpf=g(bpf[key[0]], "bpf"), icv=g(bpf[key[0]], "icv_ml"),
                    centiloid=g(strat.get(key[0], {}), "centiloid"))
        for d in (15, 20, 25, 30):
            mc, su = g(s, f"mc_flu_d{d}"), g(s, f"sur_flu_d{d}")
            if mc is None or su is None: continue
            rows.append(dict(depth=float(d), flu_mc=mc, flu_sur=su, err=su-mc, aerr=abs(su-mc), **anat))
    return rows

def std_ols(rows, ycol, xcols):
    """standardized OLS: returns [(name, beta, lo, hi)] with 95% CI."""
    good = [r for r in rows if r[ycol] is not None and all(r[c] is not None for c in xcols)]
    y = np.array([r[ycol] for r in good], float)
    X = np.array([[r[c] for c in xcols] for r in good], float)
    y = (y - y.mean()) / y.std()
    X = (X - X.mean(0)) / X.std(0)
    Xa = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    resid = y - Xa @ beta
    dof = len(y) - Xa.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(Xa.T @ Xa)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = []
    for i, name in enumerate(xcols):
        b, s = beta[i+1], se[i+1]
        out.append((name, b, b-1.96*s, b+1.96*s))
    return out, len(good)

LABEL = {"depth": "Target depth", "skull": "Skull thickness", "scalp": "Scalp thickness",
         "muscle": "Temporalis", "cortical": "Cortical bone", "cancellous": "Cancellous bone",
         "bpf": "BPF", "icv": "ICV", "flag": "Sinus/sphenoid", "centiloid": "Amyloid burden"}

def forest(series, xcols, title, xlabel, fname, note=""):
    y = np.arange(len(xcols))[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 0.62*len(xcols)+1.9))
    allv = [v for (_, res, _) in series for (_, b, lo, hi) in res for v in (lo, hi)]
    lo_, hi_ = min(allv), max(allv); span = (hi_ - lo_) or 1.0
    ax.set_xlim(min(lo_-0.10*span, -0.04*span), max(hi_+0.10*span, 0.04*span))
    ax.set_axisbelow(True); ax.grid(axis="x", color="#ececec", lw=1.0, zorder=0)
    ax.axvline(0, color="#b8b8b8", lw=1.4, zorder=1)
    offs = np.linspace(0.17, -0.17, len(series)) if len(series) > 1 else [0]
    for (lab, res, c), off in zip(series, offs):
        d = {n: (b, lo, hi) for n, b, lo, hi in res}
        for yi, xc in zip(y, xcols):
            b, lo, hi = d[xc]
            ax.plot([lo, hi], [yi+off]*2, color=c, lw=2.6, solid_capstyle="round", zorder=3)
            ax.plot(b, yi+off, "o", ms=9, color=c, zorder=4)
        ax.plot([], [], "o", color=c, ms=10, label=lab)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[x] for x in xcols], fontsize=16)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=14)
    ax.set_ylim(-0.7, len(xcols)-0.3); ax.set_xlabel(xlabel, fontsize=16)
    if len(series) > 1:
        ax.legend(loc="center left", frameon=False, fontsize=14.5, handletextpad=0.3)
    ax.set_title(title, fontsize=15, loc="left", pad=12)
    if note: ax.text(0.015, 0.03, note, transform=ax.transAxes, fontsize=12,
                     color="#8a8a8a", va="bottom", ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig); print("wrote", os.path.join(OUT, fname+".png"))

def main():
    rows = load_merged()
    print(f"observations (scene x depth): {len(rows)}")
    XA = ["depth", "skull", "scalp", "bpf", "icv", "centiloid"]
    mc, n1 = std_ols(rows, "flu_mc", XA)
    sur, n2 = std_ols(rows, "flu_sur", XA)
    forest([("Monte Carlo", mc, CLAY), ("Surrogate", sur, BLUE)], XA,
           "Drivers of delivered fluence (15–30 mm)",
           "Standardized coefficient, β", "fluence_drivers")
    XB = ["depth", "skull", "scalp", "icv", "flag"]
    er, n3 = std_ols(rows, "aerr", XB)
    forest([("Surrogate error", er, BLUE)], XB,
           "Drivers of surrogate error",
           "Standardized coefficient, β", "fidelity_drivers")
    for nm, res in [("MC", mc), ("SUR", sur)]:
        print(nm, {n: f"{b:+.2f}" for n, b, lo, hi in res})
    print("|error|", {n: f"{b:+.2f}" for n, b, lo, hi in er})

if __name__ == "__main__":
    main()
