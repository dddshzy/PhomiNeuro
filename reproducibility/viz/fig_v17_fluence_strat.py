#!/usr/bin/env python3
"""Comprehensive fluence factor + stratified analysis over the FULL held-out cohort (84 heads x
19 electrodes = 1596 scenes), delivered fluence fixed at 25 mm (depth removed).

Fig A `fluence_drivers_demo`: standardized regression of log10 fluence@25 on
  [skull, scalp, BPF, ICV, age, sex(M), AD] — MC (clay) vs V17 surrogate (blue),
  CLUSTER-ROBUST SE by subject (sex/AD/age/BPF/ICV are head-level, repeated over 19 electrodes).
Fig B `fluence_by_stratum`: per-head Cohen's d (+95% CI) of mean delivered MC fluence@25 (clay)
  and mean surrogate fidelity R² (blue) across the cohort strata from V17-cohort-stratification.md
  (AD-HC, sex, amyloid, MMSE, APOE) — who receives more light, and is the surrogate equally
  accurate across subgroups.
"""
import os, csv, sys
import numpy as np
from scipy import stats
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

def num(d, k):
    try: return float(d[k])
    except: return None

def load():
    svm = {(r["subject"], r["electrode"]): r for r in
           csv.DictReader(open(os.path.join(PUP, "cohort_surrogate_vs_mc.csv")))}
    th = {(r["subject"], r["electrode"]): r for r in
          csv.DictReader(open(os.path.join(PUP, "cohort_electrode_thickness.csv")))}
    strat = {r["subject"].lower(): r for r in csv.DictReader(open(os.path.join(MET, "cohort_stratification.csv")))}
    bpf = {r["subject"].lower(): r for r in csv.DictReader(open(os.path.join(MET, "cohort_bpf.csv")))}
    rows = []
    for key, s in svm.items():
        t = th.get(key); st = strat.get(key[0]); bp = bpf.get(key[0])
        if not (t and st and bp): continue
        mc, su = num(s, "mc_flu_d25"), num(s, "sur_flu_d25")
        if mc is None or su is None: continue
        rows.append(dict(subject=key[0], electrode=key[1],
            skull=num(t, "skull_mm"), scalp=(num(t, "skin_mm") or 0)+(num(t, "muscle_mm") or 0),
            bpf=num(bp, "bpf"), icv=num(bp, "icv_ml"), age=num(st, "age"),
            centiloid=num(st, "centiloid"),
            sexM=1.0 if st["sex"] == "M" else 0.0, AD=1.0 if st["group"] == "AD" else 0.0,
            amyloid=st["amyloid_status"], mmse_lt24=0.0 if st["mmse_ge24"] == "1" else 1.0,
            e4=1.0 if int(st["e4_dose"]) > 0 else 0.0,
            mc25=mc, sur25=su, mc30=num(s, "mc_flu_d30"), sur30=num(s, "sur_flu_d30"),
            fid=num(s, "fid_r2")))
    return rows

def cluster_ols(rows, ycol, xcols):
    good = [r for r in rows if r[ycol] is not None and all(r[c] is not None for c in xcols)]
    y = np.array([r[ycol] for r in good]); X = np.array([[r[c] for c in xcols] for r in good])
    cl = np.array([r["subject"] for r in good])
    y = (y - y.mean()) / y.std(); X = (X - X.mean(0)) / X.std(0)
    Xa = np.column_stack([np.ones(len(y)), X])
    XtXi = np.linalg.pinv(Xa.T @ Xa); beta = XtXi @ Xa.T @ y; u = y - Xa @ beta
    meat = np.zeros((Xa.shape[1],) * 2)
    for c in np.unique(cl):
        Xg = Xa[cl == c]; s = Xg.T @ u[cl == c]; meat += np.outer(s, s)
    G, n, k = len(np.unique(cl)), len(y), Xa.shape[1]
    cov = (G/(G-1))*((n-1)/(n-k)) * XtXi @ meat @ XtXi
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return [(xc, beta[i+1], beta[i+1]-1.96*se[i+1], beta[i+1]+1.96*se[i+1]) for i, xc in enumerate(xcols)], G

def cohen_ci(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.std(ddof=1)**2+(nb-1)*b.std(ddof=1)**2)/(na+nb-2))
    d = (a.mean()-b.mean())/sp if sp > 0 else 0.0
    se = np.sqrt((na+nb)/(na*nb)+d*d/(2*(na+nb)))
    return d, d-1.96*se, d+1.96*se

LABEL = {"skull": "Skull thickness", "scalp": "Scalp thickness", "bpf": "BPF",
         "icv": "ICV", "age": "Age", "sexM": "Male sex", "AD": "AD diagnosis",
         "centiloid": "Amyloid burden"}

def forest(series, ylabels, title, xlabel, fname, note="", ylab_map=None, legend_loc="lower right",
           xlim=None, grids=(-0.5, -0.25, 0.25, 0.5)):
    yl = [ (ylab_map or LABEL).get(x, x) for x in ylabels ]
    y = np.arange(len(ylabels))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.6*len(ylabels)+2.0))
    for g in grids: ax.axvline(g, color="#ececec", lw=1.0, zorder=0)
    ax.axvline(0, color="#b8b8b8", lw=1.4, zorder=1)
    offs = np.linspace(0.17, -0.17, len(series)) if len(series) > 1 else [0]
    for (lab, res, c), off in zip(series, offs):
        d = {n: (b, lo, hi) for n, b, lo, hi in res}
        for yi, xc in zip(y, ylabels):
            if xc not in d: continue
            b, lo, hi = d[xc]
            ax.plot([lo, hi], [yi+off]*2, color=c, lw=2.6, solid_capstyle="round", zorder=3)
            ax.plot(b, yi+off, "o", ms=9, color=c, zorder=4)
        ax.plot([], [], "o", color=c, ms=10, label=lab)
    ax.set_yticks(y); ax.set_yticklabels(yl, fontsize=16); ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=14); ax.set_ylim(-0.7, len(ylabels)-0.3)
    if xlim is not None: ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=16)
    if len(series) > 1: ax.legend(loc=legend_loc, frameon=False, fontsize=14.5, handletextpad=0.3)
    ax.set_title(title, fontsize=15, loc="left", pad=12)
    if note: ax.text(0.015, 0.965, note, transform=ax.transAxes, fontsize=11.5,
                     color="#8a8a8a", va="top", ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig); print("wrote", fname)

def sex_box(H):
    M = [h["mc"] for h in H.values() if h["sexM"] == 1]
    F = [h["mc"] for h in H.values() if h["sexM"] == 0]
    d, lo, hi = cohen_ci(M, F); _, p = stats.mannwhitneyu(M, F, alternative="two-sided")
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    data, cols = [M, F], [BLUE, CLAY]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.52, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#3a3a46", lw=1.8), boxprops=dict(lw=0),
                    whiskerprops=dict(color="#a8a8a8", lw=1.2), capprops=dict(color="#a8a8a8", lw=1.2))
    for patch, c in zip(bp["boxes"], cols): patch.set_facecolor(c); patch.set_alpha(0.28)
    rng = np.random.default_rng(0)
    for xi, (v, c) in enumerate(zip(data, cols)):
        ax.scatter(rng.normal(xi, 0.075, len(v)), v, s=26, color=c, alpha=0.85, lw=0, zorder=3)
    top = max(max(M), max(F)); spread = top - min(min(M), min(F)); yb = top + 0.09*spread
    ax.plot([0, 1], [yb, yb], color="#3a3a46", lw=1.2)
    star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
    ax.text(0.5, yb + 0.015*spread, f"{star}    d = {d:+.2f}", ha="center", fontsize=15.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Male\n(n={len(M)})", f"Female\n(n={len(F)})"], fontsize=16)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Delivered fluence at 25 mm (log$_{10}$)", fontsize=16)
    ax.set_title("Males receive more transcranial light", fontsize=15.5, loc="left", pad=10)
    ax.tick_params(axis="y", labelsize=14); ax.tick_params(axis="x", length=0)
    ax.spines["left"].set_visible(True); ax.spines["left"].set_color("#3a3a46")
    ax.spines["left"].set_linewidth(0.8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"fluence_sex_box.{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig); print(f"wrote fluence_sex_box  (M {np.mean(M):.2f} vs F {np.mean(F):.2f}, d={d:+.2f}, p={p:.1e})")

def main():
    rows = load()
    print(f"scenes: {len(rows)}  heads: {len(set(r['subject'] for r in rows))}")

    # ---- Fig A: regression (depth removed, demographics added), cluster-robust ----
    XA = ["skull", "scalp", "bpf", "icv", "age", "sexM", "AD", "centiloid"]
    def drivers(depth, fname):
        mc, _ = cluster_ols(rows, f"mc{depth}", XA)
        sur, _ = cluster_ols(rows, f"sur{depth}", XA)
        forest([("Monte Carlo", mc, CLAY), ("Surrogate", sur, BLUE)], XA,
               f"Drivers of delivered fluence at {depth} mm",
               "Standardized coefficient, β", fname,
               xlim=(-0.27, 0.27), grids=(-0.2, -0.1, 0.1, 0.2))
        print(f"@{depth}  MC :", {n: f"{b:+.2f}" for n, b, lo, hi in mc})
        print(f"@{depth}  SUR:", {n: f"{b:+.2f}" for n, b, lo, hi in sur})
    drivers(25, "fluence_drivers_demo")
    drivers(30, "fluence_drivers_demo_30mm")

    # ---- Fig B: stratified per-head Cohen's d (delivered fluence & fidelity) ----
    heads = {}
    for r in rows:
        heads.setdefault(r["subject"], []).append(r)
    H = {}   # per head: mean fluence@25, mean fidelity, + strat labels
    for h, rs in heads.items():
        H[h] = dict(mc=np.mean([x["mc25"] for x in rs]),
                    fid=np.mean([x["fid"] for x in rs if x["fid"] is not None]),
                    AD=rs[0]["AD"], sexM=rs[0]["sexM"], amyloid=rs[0]["amyloid"],
                    mmse_lt24=rs[0]["mmse_lt24"], e4=rs[0]["e4"])
    STRATA = [("AD vs HC",       lambda h: h["AD"] == 1,          lambda h: h["AD"] == 0),
              ("Male vs Female", lambda h: h["sexM"] == 1,        lambda h: h["sexM"] == 0),
              ("Amyloid+ vs −",  lambda h: h["amyloid"] == "positive", lambda h: h["amyloid"] == "negative"),
              ("MMSE <24 vs ≥24", lambda h: h["mmse_lt24"] == 1,  lambda h: h["mmse_lt24"] == 0),
              ("APOE ε4+ vs −",  lambda h: h["e4"] == 1,          lambda h: h["e4"] == 0)]
    flu_res, fid_res, ylabs = [], [], []
    for name, fa, fb in STRATA:
        A = [h for h in H.values() if fa(h)]; B = [h for h in H.values() if fb(h)]
        key = name.split(" ")[0] + name  # unique key
        ylabs.append(key)
        flu_res.append((key,) + cohen_ci([h["mc"] for h in A], [h["mc"] for h in B]))
        fid_res.append((key,) + cohen_ci([h["fid"] for h in A], [h["fid"] for h in B]))
        print(f"{name}: n={len(A)}/{len(B)}  fluence d={flu_res[-1][1]:+.2f}  fidelity d={fid_res[-1][1]:+.2f}")
    ymap = {k: n for (n, _, _), k in zip(STRATA, ylabs)}
    forest([("delivered fluence", flu_res, CLAY), ("surrogate fidelity R²", fid_res, BLUE)], ylabs,
           "Stratified group differences  (per head, Cohen's d)",
           "Cohen's d between strata  (+ = first group higher)",
           "fluence_by_stratum", ylab_map=ymap, legend_loc="upper right",
           note="per-head means (19 electrodes);\nMMSE<24 n=7, underpowered")
    sex_box(H)

if __name__ == "__main__":
    main()
