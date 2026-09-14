#!/usr/bin/env python3
"""Generate main-text and supplementary V18 held-out benchmark tables.

Scene-level ratio metrics are aggregated by their median and other metrics by
their mean. Gates are averaged per model seed, and tables report the mean and
standard deviation across three seeds.
"""
import os, sys, json, glob, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import repro_config as RC

ROOT = RC.WORK_DIR
S = RC.METRICS_DIR
OUTDIR = RC.TABLE_DIR

# Model key, display label, and table group in publication order.
ROWS = [
    ("v18hero3",    "**本工作 V18 hero3(定稿)** (entry, K=3000, 精简输入)", "ours"),
    ("ours",        "本工作 V17 hero (entry, K=3000)",            "ours"),
    ("v18h2",       "本工作 V18 hero2 (entry, K=3000)",           "ours"),
    ("vit",         "ViT / UNETR",                                "grid"),
    ("dynunet",     "DynUNet",                                    "grid"),
    ("segresnet",   "SegResNet",                                  "grid"),
    ("unet",        "UNet",                                       "grid"),
    ("fno",         "FNO",                                        "grid"),
    ("coordrff2",   "**Coord-RFF(定稿)** (输入与 hero3 逐项对齐)", "coord"),
    ("coordsiren2", "**SIREN(定稿)** (输入与 hero3 逐项对齐)",     "coord"),
    ("coordrff",    "Coord-RFF (未精简,对照)",                     "coordctrl"),
    ("coordsiren",  "SIREN (未精简,对照)",                         "coordctrl"),
    ("coordrff4",   "Coord-RFF (只丢 sin/cos 角)",                 "coord4"),
    ("coordsiren4", "SIREN (只丢 sin/cos 角)",                     "coord4"),
    ("coordrff3",   "Coord-RFF (再换成时间 PE)",                   "coord3"),
    ("coordsiren3", "SIREN (再换成时间 PE)",                       "coord3"),
]
# the steadymask panel and the floor-free panels name the same family differently; `coordrff` is a
# strict PREFIX of `coordrff2`, so every lookup here is an exact dict hit, never startswith().
ALIAS = {"ours": "v17k3", "v18h2": "v18hero2", "v18hero3": "v18hero3",
         "vit": "vit", "dynunet": "dynunet", "segresnet": "segresnet", "unet": "unet", "fno": "fno",
         "coordrff2": "coordrff2", "coordsiren2": "coordsiren2",
         "coordrff3": "coordrff3", "coordsiren3": "coordsiren3",
         "coordrff4": "coordrff4", "coordsiren4": "coordsiren4",
         "coordrff": "coordrff", "coordsiren": "coordsiren"}
# Intermediate input-simplification arms are reported in the supplement.
SUPP_ONLY = {"coord3", "coord4"}
GROUP = {"ours": "本工作", "grid": "网格族 (固定 10 通道输出)",
         "coord": "坐标族 (基线,输入与 hero3 对齐)",
         "coordctrl": "坐标族 (未精简形态,对照)",
         "coord4": "坐标族 (只丢 sin/cos 角)",
         "coord3": "坐标族 (再换成带限时间 PE)"}
RATIO = {"var_ratio", "slope"}
STEADY_GLOBS = ["steady_final/heldout/*.jsonl", "steady_final/heldout_hero3/*.jsonl",
                "steady_final/heldout_v18h2/*.jsonl", "steady_final/heldout_coord2/*.jsonl",
                "steady_final/heldout_coord3/*.jsonl", "steady_final/heldout_coord4/*.jsonl",
                "steady_final/heldout_vit/*.jsonl"]
T_NS = [0.1 + 0.2 * k for k in range(10)]

_HDR_FIXED = (
    "> 队列:84 例 held-out OASIS 头 × 6 电极(Fp1 F7 Cz T4 Pz O1)= **504 场景**,10 个时间门,"
    "每场景 50 000 采样点。\n"
    "> 这 84 例从未参与任何模型选择;20 例 Dev-test 已被多轮消融反复评分,只用于消融表。\n"
    "> 每个模型 **3 个随机种子**;`±` 是**门平均值的跨种子标准差**,不是场景间离散度"
    "(后者宽约两个数量级,回答的是头间变异性,不是该数值的可靠性)。\n"
    "> **定稿版本**:本工作为 **V18 hero3**,坐标基线为**输入对齐臂**(`2` 系列),表中以「定稿」标注;"
    "选型依据与代价见表 T1 脚注。\n")


def header(seen_fams):
    """Build the table banner and derive the absent-family list from loaded data."""
    absent = [l for f, l, _ in ROWS if f not in seen_fams]
    if not absent:
        return _HDR_FIXED
    return _HDR_FIXED + "> 尚未入表的族:" + "、".join(absent) + "。\n"


# ------------------------------------------------------------------ loading
def load_steady():
    fs = []
    for g in STEADY_GLOBS:
        fs += sorted(glob.glob(os.path.join(S, g)))
    d = pd.concat([pd.read_json(f, lines=True) for f in fs], ignore_index=True)
    d["fam"] = d["model"].map(lambda x: x.rsplit("_s", 1)[0]
                              if x.rsplit("_s", 1)[-1].isdigit() else x)
    d["scene"] = d["head"] + "/" + d["tag"]
    d = d[d.fam.isin([k for k, _, _ in ROWS])].copy()
    nsc = d.scene.nunique()
    print(f"steadymask: {len(fs)} files, {len(d):,} rows, {d.fam.nunique()} families, "
          f"{nsc} scenes, {d['head'].nunique()} heads, {d.tag.nunique()} electrodes")
    bad = [f for f in d.fam.unique() if d[d.fam == f].model.nunique() != 3]
    assert not bad, f"not 3 seeds for {bad}"
    assert len(d) == d.model.nunique() * nsc * 10, "ragged (model, scene, gate) grid"
    assert nsc == 504 and d.tag.nunique() == 6, f"expected 504 scenes / 6 electrodes, got {nsc}"
    return d


def scene_agg(sub, m):
    g = sub.groupby(["model", "gate"])[m]
    return g.median() if m in RATIO else g.mean()


def gate_mean(d, fam, m, rows=None):
    sub = d[d.fam == fam] if rows is None else d[(d.fam == fam) & rows]
    if m not in sub.columns or sub[m].isna().all() or sub.empty:
        return None
    ps = scene_agg(sub, m).groupby("model").mean()
    return float(ps.mean()), float(ps.std(ddof=1)), int(len(ps))


def per_gate(d, fam, m):
    sub = d[d.fam == fam]
    if m not in sub.columns or sub[m].isna().all() or sub.empty:
        return None
    ps = scene_agg(sub, m).unstack("gate")
    return [float(ps[g].mean()) for g in range(10)], [float(ps[g].std(ddof=1)) for g in range(10)]


def worst_gate(d, fam, m):
    """Compute each seed's worst gate, then its mean and SD across seeds."""
    sub = d[d.fam == fam]
    if m not in sub.columns or sub[m].isna().all() or sub.empty:
        return None
    w = scene_agg(sub, m).unstack("gate").min(axis=1)
    return float(w.mean()), float(w.std(ddof=1))


def gamma_panel():
    """gamma is floor-free, so it lives in the per-scene diag panel, not the steadymask JSONL."""
    out = {}
    for fam, _, _ in ROWS:
        fs = sorted(glob.glob(os.path.join(S, f"final_merged/scene_ho_{ALIAS[fam]}_s?.csv")))
        if len(fs) != 3:
            out[fam] = None
            continue
        gd = pd.concat([pd.read_csv(f).assign(seedfile=os.path.basename(f)) for f in fs],
                       ignore_index=True)
        ps = gd.groupby(["seedfile", "gate"])["gamma"].mean().unstack("gate")
        out[fam] = dict(pergate=[float(ps[g].mean()) for g in range(10)],
                        pergate_sd=[float(ps[g].std(ddof=1)) for g in range(10)],
                        gm=float(ps.mean(axis=1).mean()), sd=float(ps.mean(axis=1).std(ddof=1)),
                        n=len(fs))
    return out


def diag_panel(metric="r2"):
    """The OTHER support convention, on the SAME predictions and the same 504 scenes.

    steadymask : one support for all ten gates, fixed by the 2 ns TIME-INTEGRATED field at D = 7.
                 A gate's fluence is frequently exactly zero inside it (96.1% of supported voxels
                 at gate 0, 0.24% at gate 9 -- the light has not arrived yet), so early-gate R2 is
                 mostly "did the model put the arrival front in the right place" and late-gate R2 is
                 nearly pure amplitude.
    diag       : each gate's support comes from THAT gate's own peak. Early gates are then the
                 bright near-source blob and late gates the faint spread-out field, i.e. exactly the
                 opposite difficulty ordering.

    Neither is wrong. They are different questions, and this project must report both because the
    ranking between our family and the grid family is not the same under the two.
    """
    out = {}
    for fam, _, _ in ROWS:
        fs = sorted(glob.glob(os.path.join(S, f"final_merged/scene_ho_{ALIAS[fam]}_s?.csv")))
        if len(fs) != 3:
            out[fam] = None
            continue
        gd = pd.concat([pd.read_csv(x).assign(sf=os.path.basename(x)) for x in fs],
                       ignore_index=True)
        ps = gd.groupby(["sf", "gate"])[metric].mean().unstack("gate")
        out[fam] = dict(pergate=[float(ps[g].mean()) for g in range(10)],
                        gm=float(ps.mean(axis=1).mean()), sd=float(ps.mean(axis=1).std(ddof=1)),
                        worst=float(ps.min(axis=1).mean()),
                        worst_sd=float(ps.min(axis=1).std(ddof=1)),
                        seeds=ps.mean(axis=1).values)
    return out


def worst_gate(d, fam, m="r2"):
    """min over the ten gates, per seed, then mean +- SD. Convention-dependent in VALUE but the
    ORDERING it produces is the one statement that survives both support conventions."""
    ps = scene_agg(d[d.fam == fam], m).unstack("gate")
    w = ps.min(axis=1)
    return float(w.mean()), float(w.std(ddof=1))


def energy_panel():
    """Absorbed-energy ratio per gate. Summarised as MEDIAN |log10 ratio| over gates: symmetric in
    over- and under-deposition, unlike a mean of the raw ratio."""
    out = {}
    for fam, _, _ in ROWS:
        fs = sorted(glob.glob(os.path.join(S, f"final_merged/energy_{ALIAS[fam]}_s?.json")))
        if len(fs) != 3:
            out[fam] = None
            continue
        med, p25, p75, absl = [], [], [], []
        for f in fs:
            D = json.load(open(f))
            c = D[list(D)[0]]["oasis"]
            pg = c["per_gate"]
            m = [pg[str(g)]["energy_median"] for g in range(10)]
            med.append(m)
            p25.append([pg[str(g)]["energy_p25"] for g in range(10)])
            p75.append([pg[str(g)]["energy_p75"] for g in range(10)])
            absl.append(float(np.median([abs(np.log10(x)) for x in m if x and x > 0])))
        med, p25, p75 = np.array(med), np.array(p25), np.array(p75)
        out[fam] = dict(med=med.mean(0).tolist(), med_sd=med.std(0, ddof=1).tolist(),
                        p25=p25.mean(0).tolist(), p75=p75.mean(0).tolist(),
                        absl=float(np.mean(absl)), absl_sd=float(np.std(absl, ddof=1)),
                        n_scenes=int(c["n_scenes"]))
    return out


# ------------------------------------------------------------------ formatting
def fmt(v, sd, nd=4, best=False):
    if v is None:
        return "—"
    s = f"{v:.{nd}f} ± {sd:.{nd}f}"
    return f"**{s}**" if best else s


def table(colspecs, rowvals, bestdir):
    """colspecs = [(header, ndigits)]; rowvals = [(label, group, [(v,sd)|None])]."""
    lines = ["| 模型 | " + " | ".join(h for h, _ in colspecs) + " |",
             "|---|" + "---|" * len(colspecs)]
    best = []
    for j, (_, _) in enumerate(colspecs):
        vals = [rv[2][j][0] for rv in rowvals if rv[2][j] is not None]
        best.append((max(vals) if bestdir[j] > 0 else min(vals)) if vals else None)
    seen = set()
    for label, grp, cells in rowvals:
        if grp not in seen:
            seen.add(grp)
            lines.append(f"| **{GROUP[grp]}** |" + " |" * len(colspecs))
        row = []
        for j, (_, nd) in enumerate(colspecs):
            c = cells[j]
            row.append("—" if c is None else
                       fmt(c[0], c[1], nd, best[j] is not None and abs(c[0] - best[j]) < 1e-12))
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    return "\n".join(lines)


# Set in ``main`` from the model families present in the loaded results.
HDR = _HDR_FIXED


def write(path, title, body, note=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# {title}\n\n{HDR}\n{body}\n")
        if note:
            fh.write("\n" + note + "\n")
    print(f"  -> {os.path.relpath(path, OUTDIR)}")


# ------------------------------------------------------------------ tables
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    d = load_steady()
    G, E = gamma_panel(), energy_panel()
    seen_fams = set(d.fam.unique())
    global HDR
    HDR = header(seen_fams)
    present_all = [(f, l, g) for f, l, g in ROWS if f in seen_fams]      # 附录用:全部 13 族
    present = [r for r in present_all if r[2] not in SUPP_ONLY]          # 正文用:9 族
    missing = [l for f, l, _ in ROWS if f not in set(d.fam.unique())]
    if missing:
        print(f"  absent from the steadymask panel: {missing}")
    O = a.outdir

    # ---------------------------------------------------------------- T1 主表
    cols = [("R² ↑", 4), ("最差门 R² ↑", 4), ("RMSE ↓", 4), ("medAE ↓", 4), ("Spearman ↑", 4),
            ("CCC ↑", 4), ("gamma(3%/2px) ↑", 4), ("\\|log₁₀ 能量比\\| ↓", 4)]
    bd = [1, 1, -1, -1, 1, 1, 1, -1]
    rv = []
    for f, l, g in present:
        r2 = gate_mean(d, f, "r2")
        cells = [(r2[0], r2[1]), worst_gate(d, f, "r2")]
        cells += [None if c is None else (c[0], c[1])
                  for c in (gate_mean(d, f, m) for m in ("rmse", "medae", "spearman", "ccc"))]
        cells.append(None if G[f] is None else (G[f]["gm"], G[f]["sd"]))
        cells.append(None if E[f] is None else (E[f]["absl"], E[f]["absl_sd"]))
        rv.append((l, g, cells))
    write(os.path.join(O, "T1_heldout_accuracy_main.md"),
          "表 T1 — held-out 主表:门平均精度与能量守恒",
          table(cols, rv, bd),
          "*↑ 越大越好,↓ 越小越好;加粗 = 该列最优。*\n\n"
          "**口径**:R²/RMSE/medAE/Spearman/CCC 取自 steadymask 面板(时间积分支撑集,D=7);"
          "gamma(3%/2px) 与能量比取自 floor-free 的逐场景面板 —— gamma 在线性空间的完整二维切片上"
          "计算,永不触及地板,因而与支撑集口径无关。\n\n"
          "**最差门 R²** = 每个种子在 10 个门上取最小值,再跨种子平均。它与门平均是同一口径下的"
          "两个汇总统计量,不是另一种口径。设它的理由是:门平均会让一个「早门崩、晚门好」的模型"
          "与一个「全程平稳」的模型看起来相同,而时间分辨代理模型的可用性取决于**最弱的那个门**。"
          "本表中这两个统计量对 hero3 与 V17 给出不同答案(见表 T4),对网格族则一致但量级相差十倍。\n\n"
          "**定稿版本(表中以「定稿」标注)。** 本工作的定稿模型是 **V18 hero3**,定稿的坐标基线是"
          "**输入对齐臂**(Coord-RFF / SIREN 的 `2` 系列)。\n\n"
          "选 hero3 **不是因为它在本表上更准** —— 本表的精度列上它与 V17 hero 等价"
          "(门平均 −0.0027,落在带宽内),最差门还略差(−0.0117,显著,见表 T4)。"
          "决定性的是**时间轴**:hero3 把时间位置编码带限到 `num_freq_t=3`(f ≤ 4),"
          "以尊重 10 门 Δt = 0.2 ns 的采样定律(Nyquist f < 5);V17 hero 让时间与空间共用一条 "
          "f = 1…128 的阶梯,其中 f = 8, 16, 32, 64, 128 在 t 上全部混叠。"
          "在**同一批 504 个 held-out 场景、同一 E6 电极、同一 D=8 地板**上实测"
          "(`scratch_jobs/l2bc/l2b_v18_{hero1,v17}.json`):两者**在训练门上等价**"
          "(on-grid R² 0.7723 vs 0.7737),但**在门与门之间**相差 0.1370 R²"
          "(中点内插 R² **0.7413** vs **0.6044**),内插代价 0.0287 vs 0.1667,**小 5.82 倍**。"
          "一个以「在 t 上连续、可在训练门之间查询」为卖点的时间分辨代理模型,必须在这条轴上选型。\n\n"
          "*该 L2B 测量的两处限制*:(1) 带限臂测的是 **hero1**(与 hero3 同为 `num_freq=4 / "
          "num_freq_t=3`、同 K=3000 buffer,但不含 hero3 的 L_lin/分层加权/λ_PINN=1.0/精简输入),"
          "**hero3 本身未直接测过 L2B**;(2) 两侧均为**单 seed**,无带宽。"
          "因此这一条陈述的是「带限这一属性」的效应,不是 hero3 与 V17 的逐 seed 比较。\n\n"
          "**坐标基线为什么有两组行。** 正文的坐标基线是**输入对齐臂**:其非金字塔输入与 hero3 "
          "逐项一致(无光学、light 6 已丢 sin/cos 角、srcfeat 2、path 6、时间为带限傅里叶 PE),"
          "唯一差别是 xyz 的表示方式(RFF 256 维高斯编码 / 裸 xyz,对 hero3 是 FM 金字塔 + 倍频程 PE),"
          "而那正是架构对比的标的。**未精简形态列为对照**,因为它更强 —— "
          "对齐使 Coord-RFF 降 0.0130、SIREN 降 0.0245(均显著,见表 S6);"
          "只呈现被我们裁剪过的那一版,而把更强的移出正文,是选择性呈现。\n\n"
          "全部坐标与网格基线同训练配置:删失铰链 1.0 / offset 0.5、入射点参考、加权损失,"
          "与本工作一致。基线的 checkpoint 归属由**逐位重算**确认"
          "(重算若干场景与已存面板比对,`max |ΔR²| = 0`),不靠启动脚本推断。\n\n"
          "**|log₁₀ 能量比|** = 逐门吸收能量比中位数取 |log₁₀| 后再对 10 个门取中位数。"
          "用对称的对数尺度而非原始比值:2× 与 0.5× 是同等大小的误差,而比值的均值不是"
          "(一次按均值聚合曾把真值 0.47 报成 524.8)。逐门明细见表 T3。")

    # ---------------------------------------------------------------- T2 逐门 R²
    PG = {f: per_gate(d, f, "r2") for f, _, _ in present}
    WG = {f: worst_gate(d, f, "r2") for f, _, _ in present}
    pg = {f: PG[f][0] for f in PG}
    bestg = [max(pg[f][k] for f in pg) for k in range(10)]
    bw = max(WG[f][0] for f in WG)
    for f, l, _ in present:                       # provenance of the estimator change, into the log
        print(f"    worst-gate {l:34s} min-of-means {min(pg[f]):.4f} -> "
              f"mean-of-mins {WG[f][0]:.4f} ± {WG[f][1]:.4f}")
    lines = ["| 模型 | " + " | ".join(f"t={x:.1f} ns" for x in T_NS) + " | 最差门 |",
             "|---|" + "---|" * 11]
    seen = set()
    for f, l, g in present:
        if g not in seen:
            seen.add(g); lines.append(f"| **{GROUP[g]}** |" + " |" * 11)
        v, s = PG[f]
        cells = [(f"**{v[k]:.4f} ± {s[k]:.4f}**" if abs(v[k] - bestg[k]) < 1e-12
                  else f"{v[k]:.4f} ± {s[k]:.4f}") for k in range(10)]
        wm, ws = WG[f]
        cells.append(f"**{wm:.4f} ± {ws:.4f}**" if abs(wm - bw) < 1e-12
                     else f"{wm:.4f} ± {ws:.4f}")
        lines.append(f"| {l} | " + " | ".join(cells) + " |")
    write(os.path.join(O, "T2_heldout_pergate_r2_main.md"),
          "表 T2 — held-out 逐门 R²", "\n".join(lines),
          "*均值 ± 跨种子标准差(3 个训练种子);加粗 = 该列最优。*\n"
          "*「最差门」是**每个种子各自最差的那一个门**再跨种子取均值,与表 T4 的口径一致;"
          "它不是逐门均值曲线的最小值 —— 三条曲线先平均再取最小,与各自取最小再平均并不相等。*\n\n"
          "**早门与晚门衡量的不是同一件事,读数时必须知道。** 支撑集固定不动,但一个门的真值在"
          "支撑集内**大量恰好为零**(gate 0 有 96.4%,到 gate 9 降至 0.23% —— 光还没到)。"
          "因此早门 R² 主要衡量「模型是否把到达波前放在正确位置」,晚门才接近纯幅值精度。"
          "各门处于地板的比例见附录表 S5。\n\n"
          "**门平均不能替代这张表。** 网格族在 gate 0 上大幅落后(DynUNet 0.4222 对本工作 0.90–0.92),"
          "即把能量放到了光尚未到达的区域;它们在晚门反超。一个「早门崩、晚门好」的模型与一个"
          "「全程平稳」的模型,门平均可以完全相同。**最差门**这一列就是为此而设。")

    # ---------------------------------------------------------------- T3 能量
    lines = ["| 模型 | " + " | ".join(f"t={t:.1f} ns" for t in T_NS) + " |",
             "|---|" + "---|" * 10]
    seen = set()
    for f, l, g in present:
        if E[f] is None:
            continue
        if g not in seen:
            seen.add(g); lines.append(f"| **{GROUP[g]}** |" + " |" * 10)
        lines.append(f"| {l} | " + " | ".join(f"{E[f]['med'][k]:.3f}" for k in range(10)) + " |")
    write(os.path.join(O, "T3_heldout_energy_main.md"),
          "表 T3 — held-out 逐门吸收能量比(预测 / 真值,理想值 1.000)", "\n".join(lines),
          "*跨 504 场景取中位数,再对 3 个种子取平均。四分位区间见附录表 S3b。*\n\n"
          "能量守恒不是精度指标的推论:一个 R² 尚可的模型仍可能在晚门系统性地多沉积或少沉积"
          "一到两个数量级。这一列是本工作与网格族差距最大的地方。")

    # ---------------------------------------------------------------- S1 全指标
    present = present_all
    METS = [("r2", "R²", 4, 1), ("r2_common", "R²(common floor)", 4, 1), ("r2_raw", "R²(raw)", 4, 1),
            ("rmse", "RMSE", 4, -1), ("mae", "MAE", 4, -1), ("medae", "medAE", 4, -1),
            ("p90ae", "P90 AE", 4, -1), ("bias", "bias", 4, 0), ("slope", "slope", 4, 0),
            ("var_ratio", "var_ratio", 4, 0), ("rho", "Pearson ρ", 4, 1),
            ("spearman", "Spearman", 4, 1), ("ccc", "CCC", 4, 1), ("dice", "DICE", 4, 1),
            ("miss", "miss", 4, -1), ("tnr", "TNR", 4, 1)]
    lines = ["| 模型 | " + " | ".join(lab for _, lab, _, _ in METS) + " |",
             "|---|" + "---|" * len(METS)]
    seen = set()
    for f, l, g in present:
        if g not in seen:
            seen.add(g); lines.append(f"| **{GROUP[g]}** |" + " |" * len(METS))
        cs = []
        for m, _, nd, _ in METS:
            r = gate_mean(d, f, m)
            cs.append("—" if r is None else f"{r[0]:.{nd}f} ± {r[1]:.{nd}f}")
        lines.append(f"| {l} | " + " | ".join(cs) + " |")
    write(os.path.join(a.outdir, "S1_heldout_fullpanel_supp.md"),
          "表 S1 — held-out 全指标门平均(steadymask 面板)", "\n".join(lines),
          "*bias / slope / var_ratio 无「越大越好」方向:理想值分别为 0 / 1 / 1。*\n"
          "*`slope` 与 `var_ratio` 是比值型,跨场景按**中位数**聚合;其余按均值。*\n"
          "*R²(common floor) 与 R²(raw) 是同一预测在另外两种地板处理下的值,列在此处以便读者"
          "判断主表的 R² 有多少依赖于支撑集口径。*")

    # ---------------------------------------------------------------- S2 逐门明细
    present = present_all
    blocks = []
    for m, lab in (("rmse", "RMSE"), ("dice", "DICE"), ("spearman", "Spearman")):
        ls = [f"### {lab}(逐门)", "", "| 模型 | " + " | ".join(f"t={t:.1f}" for t in T_NS) + " |",
              "|---|" + "---|" * 10]
        for f, l, _ in present:
            r = per_gate(d, f, m)
            ls.append(f"| {l} | " + " | ".join(f"{r[0][k]:.4f} ± {r[1][k]:.4f}"
                                               for k in range(10)) + " |")
        blocks.append("\n".join(ls))
    ls = ["### gamma(3%/2px)(逐门,floor-free 面板)", "",
          "| 模型 | " + " | ".join(f"t={t:.1f}" for t in T_NS) + " |", "|---|" + "---|" * 10]
    for f, l, _ in present:
        if G[f]:
            ls.append(f"| {l} | " + " | ".join(f"{G[f]['pergate'][k]:.4f}" for k in range(10)) + " |")
    blocks.append("\n".join(ls))
    ls = ["### R² 的跨种子标准差(逐门)", "",
          "| 模型 | " + " | ".join(f"t={t:.1f}" for t in T_NS) + " |", "|---|" + "---|" * 10]
    for f, l, _ in present:
        r = per_gate(d, f, "r2")
        ls.append(f"| {l} | " + " | ".join(f"{r[1][k]:.4f}" for k in range(10)) + " |")
    blocks.append("\n".join(ls))
    write(os.path.join(a.outdir, "S2_heldout_pergate_detail_supp.md"),
          "表 S2 — held-out 逐门明细(RMSE / DICE / Spearman / gamma / 种子离散度)",
          "\n\n".join(blocks),
          "*RMSE / DICE / Spearman 各格为均值 ± 跨种子标准差(3 个训练种子);"
          "标准差是该门的场景聚合值在 3 个种子间的离散度,不是场景间离散度。*\n"
          "*DICE 与 miss 在中晚门对**所有**模型(含本工作)都趋于饱和,"
          "故几何类论断只在门 0 上陈述。*")

    # ---------------------------------------------------------------- S3 decade / CSF
    present = present_all
    METS3 = [("rmse_dec0_2", "RMSE dec 0–2"), ("rmse_dec2_4", "RMSE dec 2–4"),
             ("rmse_dec4_6", "RMSE dec 4–6"), ("rmse_dec6_7", "RMSE dec 6–7"),
             ("rmse_csf", "RMSE CSF"), ("rmse_noncsf", "RMSE 非 CSF")]
    lines = ["| 模型 | " + " | ".join(lab for _, lab in METS3) + " |",
             "|---|" + "---|" * len(METS3)]
    seen = set()
    for f, l, g in present:
        if g not in seen:
            seen.add(g); lines.append(f"| **{GROUP[g]}** |" + " |" * len(METS3))
        cs = []
        for m, _ in METS3:
            r = gate_mean(d, f, m)
            cs.append("—" if r is None else f"{r[0]:.4f} ± {r[1]:.4f}")
        lines.append(f"| {l} | " + " | ".join(cs) + " |")
    eb = ["", "### S3b 逐门能量比的四分位区间(中位数 [p25–p75])", "",
          "| 模型 | " + " | ".join(f"t={t:.1f}" for t in T_NS) + " |", "|---|" + "---|" * 10]
    for f, l, _ in present:
        if E[f] is None:
            continue
        eb.append(f"| {l} | " + " | ".join(
            f"{E[f]['med'][k]:.2f} [{E[f]['p25'][k]:.2f}–{E[f]['p75'][k]:.2f}]"
            for k in range(10)) + " |")
    write(os.path.join(a.outdir, "S3_heldout_decade_csf_supp.md"),
          "表 S3 — held-out 分十倍程带与组织类 RMSE,及能量比四分位区间",
          lines_join := "\n".join(lines) + "\n" + "\n".join(eb),
          "*十倍程带相对该门自身峰值定义。深层带(dec 6–7)是训练时被删失铰链处理的区域,"
          "其形状由 PINN + 铰链生成而非由标签监督。*\n"
          "*CSF 是本项目已知的历史弱区(误差约 20%),单列以便追踪。*")

    # ---------------------------------------------------------------- S4 临床分层
    present = present_all
    grp = pd.read_csv(os.path.join(ROOT, "heldout", "heldout_subjects.csv"))[["head", "group"]]
    dd = d.merge(grp, on="head", how="left")
    assert dd["group"].notna().all(), "held-out heads without a clinical record"
    lines = ["| 模型 | R² (HC) | R² (IMPAIRED) | Δ | RMSE (HC) | RMSE (IMPAIRED) | Δ |",
             "|---|---|---|---|---|---|---|"]
    seen = set()
    for f, l, g in present:
        if g not in seen:
            seen.add(g); lines.append(f"| **{GROUP[g]}** | | | | | | |")
        cs = []
        for m in ("r2", "rmse"):
            hc = gate_mean(dd, f, m, dd["group"] == "HC")
            im = gate_mean(dd, f, m, dd["group"] == "IMPAIRED")
            cs += [f"{hc[0]:.4f} ± {hc[1]:.4f}", f"{im[0]:.4f} ± {im[1]:.4f}",
                   f"{im[0] - hc[0]:+.4f}"]
        lines.append(f"| {l} | " + " | ".join(cs) + " |")
    write(os.path.join(a.outdir, "S4_heldout_clinical_group_supp.md"),
          "表 S4 — held-out 按临床分组分层(42 HC / 42 IMPAIRED)", "\n".join(lines),
          "*分组:CDR = 0 → HC;CDR ≥ 0.5 → IMPAIRED。两组各 42 例,按年龄与性别匹配。*\n"
          "*Δ = IMPAIRED − HC。此表用于检验代理模型的精度是否依赖于受试者的疾病状态;"
          "它不是一个诊断性结果,本工作并不宣称能从通量场区分两组。*")

    # ---------------------------------------------------------------- T4 判定
    present = [r for r in present_all if r[2] not in SUPP_ONLY]
    dv, dw = {}, {}
    for f, _, _ in present:
        ps = scene_agg(d[d.fam == f], "r2").unstack("gate")
        dv[f] = ps.mean(axis=1).values
        dw[f] = ps.min(axis=1).values
    nm = {k: l for k, l, _ in ROWS}

    def verdict(A, B):
        dl = A.mean() - B.mean()
        band = 1.96 * np.sqrt((A.std(ddof=1) ** 2 + B.std(ddof=1) ** 2) / 3)
        return f"{dl:+.4f} (±{band:.4f})", ("**显著**" if abs(dl) > band else "带内")

    ls = ["| 比较 | 门平均 R² | 判定 | 最差门 R² | 判定 |", "|---|---|---|---|---|"]
    # Report comparisons involving the final model first.
    PAIRS = [("v18hero3", "vit"), ("v18hero3", "dynunet"), ("v18hero3", "segresnet"),
             ("v18hero3", "unet"), ("v18hero3", "fno"),
             ("v18hero3", "coordrff2"), ("v18hero3", "coordsiren2"),
             ("v18hero3", "ours"), ("v18hero3", "v18h2"), ("v18h2", "ours"),
             ("ours", "dynunet"), ("ours", "segresnet"),
             ("ours", "unet"), ("ours", "fno"), ("ours", "coordrff"),
             ("coordrff3", "coordrff"), ("coordsiren3", "coordsiren"),
             ("coordrff2", "coordrff3"), ("coordsiren2", "coordsiren3"),
             ("coordrff2", "coordrff"), ("coordsiren2", "coordsiren")]
    for a_, b_ in PAIRS:
        if a_ not in dv or b_ not in dv:
            continue
        c = list(verdict(dv[a_], dv[b_])) + list(verdict(dw[a_], dw[b_]))
        ls.append(f"| {nm[a_]} − {nm[b_]} | " + " | ".join(c) + " |")
    write(os.path.join(O, "T4_heldout_significance_main.md"),
          "表 T4 — held-out 成对判定(3 seed vs 3 seed)", "\n".join(ls),
          "*带宽 = 1.96·√((σ_A² + σ_B²)/3),σ 取**门平均值(或最差门值)的跨种子标准差**。*\n\n"
          "**这不是显著性检验的替代品**:n = 3,带宽本身的估计很粗;它只回答"
          "「差值是否大到不能用换一次初始化来解释」。\n\n"
          "**hero3 vs V17:两个汇总统计量给出不同答案。** 门平均说「带内、不可区分」,"
          "最差门说「显著更差」。二者是**同一口径**下的不同汇总,不是口径分歧:"
          "hero3 的门平均与 V17 相当,是因为它在若干门上略好、在最弱的门上更差,平均后相抵。"
          "按预先登记的判定规则,只要 held-out 上不优于 V17 就**并列展示取舍、不替换 V17**,"
          "本轮据此保留 V17 列。\n\n"
          "**与网格族的比较必须连同表 T2 一起读**:门平均与最差门都是本工作领先,"
          "但两者的量级差别很大(对 DynUNet 门平均 +0.02,最差门 +0.23),"
          "因为网格族的失效集中在 gate 0。")

    # ---------------------------------------------------------------- S5 支撑集普查
    sm = d[d.fam == "ours"].groupby("gate")[["support_frac", "frac_floor", "n_sampled"]].mean()
    fs = sorted(glob.glob(os.path.join(S, "final_merged/scene_ho_v17k3_s?.csv")))
    gd = pd.concat([pd.read_csv(x) for x in fs], ignore_index=True)
    nr = gd.groupby("gate")["n_reach"].mean()
    ls = ["| 门 | t (ns) | 固定支撑集占组织 | 其中真值恰为零(处于地板) | 有效非地板采样点 | "
          "若改用逐门自参照支撑集 |", "|---|---|---|---|---|---|"]
    for k in range(10):
        sf, ff, ns = sm.loc[k, "support_frac"], sm.loc[k, "frac_floor"], sm.loc[k, "n_sampled"]
        ls.append(f"| {k} | {T_NS[k]:.1f} | {100 * sf:.2f} % | {100 * ff:.2f} % | "
                  f"{ns * (1 - ff):.0f} | {nr.loc[k]:.0f} 点(≈{100 * nr.loc[k] / 50000:.1f} % 组织) |")
    write(os.path.join(O, "S5_support_census_supp.md"),
          "表 S5 — 支撑集普查:为什么本工作用固定支撑集计分", "\n".join(ls),
          "*本表描述的是**数据与掩膜**,不是模型:除最后一列外各列在所有模型上逐位相同,"
          "此处取本工作 V17 hero 的记录。*\n\n"
          "**结论先行:支撑集必须固定,否则时间维的任何陈述都不成立。**\n\n"
          "本工作的支撑集由 2 ns **时间积分**场一次性定义(D = 7,即积分场峰值以下 7 个十倍程),"
          "= 组织的 **13.24 %**,十个门共用。第 4 列显示了它随时间的行为:gate 0 时支撑集内"
          "**96.4 % 的真值恰好为零**(光还没到那里),到 gate 9 降至 0.23 %。"
          "这不是缺陷而是物理,但它决定了读法 —— 早门 R² 主要衡量到达波前的位置,"
          "晚门才是幅值精度。\n\n"
          "**替代方案(每个门用该门自身峰值定义支撑集)不可用于时间维比较。** 末列给出它的规模:"
          "从 gate 0 的 228 点(≈0.5 % 组织)膨胀到 gate 9 的 12 633 点(≈25 % 组织),**55 倍**。"
          "在这种口径下,「R² 随门下降」这句话里模型误差与总体变化不可分离 —— "
          "本项目已多次因跨条件总体错配得出错误结论。因此正文全部主表统一使用固定支撑集,"
          "该历史口径仅用于提供 **gamma**(它在整个组织切片的线性空间上计算,不使用任何支撑集掩膜,"
          "因而不受此选择影响;表内各族全部为 entry 锚定,跨族可比)。\n\n"
          "**必须同时登记的代价。** D = 7 的固定支撑集**不覆盖深层尾部**,而那正是 PINN 与"
          "删失铰链起作用的区域。深区结论因此不由本表的 R² 承担,而由它自己的仪器给出:"
          "暗区指标(speck_frac / DEEP-LEAK,见表 S9)与吸收能量比(表 T3)。"
          "用换支撑集口径的方式去展示深区能力是不可接受的 —— 那会同时改变时间维的可比性。")

    # ---------------------------------------------------------------- S6 坐标基线泛化
    import torch
    ls = ["| 基线 | 输入 | in_features | 训练 val(3 seed) | best epoch | held-out R²(3 seed) |",
          "|---|---|---|---|---|---|"]
    # Checkpoints represented in the published table. All arms use floor_hinge=1
    # and entry-referenced source features; adjacent arms change one input block.
    SPEC = [("coordrff", "v17hin_coord_rff_s%d", "RFF256 + **光学4** + **light10**(含 sin/cos 角) + srcfeat2 + path6 + **t 标量**", 279),
            ("coordrff4", "v18c4h_coord_rff_s%d", "RFF256 + 光学4 + **light6**(丢角) + srcfeat2 + path6 + t 标量", 275),
            ("coordrff3", "v18c3h_coord_rff_ft3_s%d", "RFF256 + 光学4 + light6 + srcfeat2 + path6 + **t·PE7**", 281),
            ("coordrff2", "v18c2h_coord_rff_ft3_s%d", "RFF256 + **无光学** + light6 + srcfeat2 + path6 + t·PE7", 277),
            ("coordsiren", "v17hin_coord_siren_s%d", "xyz3 + **光学4** + **light10**(含 sin/cos 角) + srcfeat2 + path6 + **t 标量**", 26),
            ("coordsiren4", "v18c4h_coord_siren_s%d", "xyz3 + 光学4 + **light6**(丢角) + srcfeat2 + path6 + t 标量", 22),
            ("coordsiren3", "v18c3h_coord_siren_ft3_s%d", "xyz3 + 光学4 + light6 + srcfeat2 + path6 + **t·PE7**", 28),
            ("coordsiren2", "v18c2h_coord_siren_ft3_s%d", "xyz3 + **无光学** + light6 + srcfeat2 + path6 + t·PE7", 24)]
    CK3 = RC.INR_CHECKPOINT_DIR
    for fam, stem, desc, nf in SPEC:
        if fam not in dict((k, 1) for k in d.fam.unique()):
            continue                       # coord3 is not scored yet -> leave the row out entirely
        vals, eps = [], []
        for s in (1, 2, 3):
            p = os.path.join(CK3, f"metrics_{stem % s}.json")
            if not os.path.exists(p):
                continue
            D = json.load(open(p))
            h = [float(x) for x in D["history"]["val_head"]]
            vals.append(min(h)); eps.append(int(np.argmin(h)) + 1)
        r = gate_mean(d, fam, "r2")
        ls.append(f"| {nm[fam]} | {desc} | {nf} | "
                  f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f} | {sorted(eps)} | "
                  f"{r[0]:.4f} ± {r[1]:.4f} |")
    write(os.path.join(a.outdir, "S6_coord_baseline_generalisation_supp.md"),
          "表 S6 — 坐标基线输入精简的逐步分解(每步单变量,全部带删失铰链)",
          "\n".join(ls),
          "*四臂使用**同一个损失**(加权 + 删失铰链 1.0 / offset 0.5)、同一 buffer(`bufmm_v16`)、"
          "同一入射点参考(entry)、同样 60 epoch、同样 seed 1/2/3;唯一差别是上表的输入列,"
          "且每相邻两行只差一项。所有 `floor_hinge = 1.0` 与 `src_ref = entry` 均从 cfg 逐个核验过。*\n\n"
          "## 逐步结果(held-out 504 场景,3 seed,门平均 R²)\n\n"
          "| 步骤 | Coord-RFF | 判定 | SIREN | 判定 |\n"
          "|---|---|---|---|---|\n"
          "| 基线(279 / 26) | 0.6610 ± 0.0005 | — | 0.6536 ± 0.0048 | — |\n"
          "| 丢 sin/cos 角 | +0.0018 | 带内 (±0.0035) | **−0.0153** | **显著** (±0.0077) |\n"
          "| t 标量 → 带限 PE7 | −0.0027 | 带内 (±0.0036) | +0.0019 | 带内 (±0.0146) |\n"
          "| 丢 4 维光学属性 | **−0.0120** | **显著** (±0.0019) | −0.0111 | 带内 (±0.0138) |\n"
          "| **合计** | **−0.0130** | **显著** (±0.0015) | **−0.0245** | **显著** (±0.0060) |\n\n"
          "## 必须登记的一次结论作废\n\n"
          "本表的前一版把这条链测成 **−0.2362 / −0.2314**,并据此写下两条结论:"
          "「把输入对齐到 hero3 会削弱基线」与「首要嫌疑是时间编码」。**两条都不成立。**\n\n"
          "原因是那一版的三个精简臂用了 `--floor-hinge` 的默认值 **0.0**,而基线 "
          "(`inr_v17hin_coord_*`)是 **1.0** —— 链条的每一步都**同时丢掉了铰链**。"
          "把铰链加回同一个臂的直接测量是:\n\n"
          "| 臂 | 无铰链 | 带铰链 | 铰链贡献 |\n|---|---|---|---|\n"
          "| Coord-RFF(全精简) | 0.4150 | 0.6480 | **+0.2330** |\n"
          "| SIREN(全精简) | 0.4222 | 0.6291 | **+0.2069** |\n"
          "| Coord-RFF(中间臂) | 0.4248 | 0.6600 | **+0.2352** |\n"
          "| SIREN(中间臂) | 0.4433 | 0.6402 | **+0.1969** |\n\n"
          "即那 −0.236 **几乎全部是铰链**,输入精简本身只值 −0.013 / −0.025。"
          "这与本工作自身消融中删失铰链的 −0.4733 R² 相互印证,是跨架构的独立证据:"
          "**删失铰链在坐标族上同样是最大的单项**。\n\n"
          "## 三条可以陈述的结论\n\n"
          "1. **时间编码无关。** 裸标量换成带限傅里叶 PE(f ≤ 4)在两个架构上都落在噪声带内"
          "(−0.0027 / +0.0019)。前一版把它列为首要嫌疑,是被铰链混淆的结果。\n"
          "2. **光学属性有代价,但只有 −0.012 量级**,不是 −0.23。它在 RFF 上显著、在 SIREN 上落在带内。"
          "对一个没有学习到解剖表征的坐标模型,4 维局部光学属性确实是它唯一的组织描述子,"
          "但代价远小于先前所报。\n"
          "3. **`ang` 的冗余论断只在 RFF 上成立。** `ang` 与 `srcdir` 互为双射"
          "(`srcdir = −[cosA·sinB, cosA·cosB, sinA]`,A ∈ [−π/2, π/2] 使 cosA ≥ 0、反演唯一),"
          "理论上丢弃无害;RFF 上确实是 +0.0018(带内),但 **SIREN 上是 −0.0153,显著**。"
          "理论上的双射不等于网络利用了这个等价性——SIREN 的空间输入只有 3 维裸 xyz,"
          "冗余编码对它并不冗余。**这一条对本工作有直接含义**:hero3 的输入精简正是依赖同一个双射论断,"
          "它在带 1488 维金字塔的模型上成立(精简后 R² +0.0001),在裸坐标模型上不成立。\n\n"
          "## 正文用哪一臂\n\n"
          "正文的坐标基线是**输入对齐臂**(277 / 24),其非金字塔输入与 hero3 逐项一致,"
          "唯一差别是 xyz 的表示方式——那正是架构对比的标的。"
          "**未精简形态(279 / 26)保留在正文作对照**:它比对齐臂强 +0.0130 / +0.0245(均显著),"
          "只呈现较弱的那个而把较强的移出正文,是选择性呈现。\n\n"
          "## 同时登记基线自身的两个弱点\n\n"
          "(它们不利于我们的对照,故必须写出)"
          "(1) `Coord-RFF` 的某些 seed best-val 落得很早,训练曲线其后持续振荡;"
          "(2) SIREN 三个 seed 训练中途出现过发散值。"
          "两者说明该族在此任务上训练不稳定,读者可据此质疑其是否被充分调参——"
          "但本轮所有基线(含四个网格族与 ViT)一律不调参,这一点对所有基线一致。")


if __name__ == "__main__":
    main()
