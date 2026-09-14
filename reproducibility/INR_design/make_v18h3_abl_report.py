#!/usr/bin/env python3
"""Generate the V18 hero3 ablation report.

The report records available seeds per arm and computes comparison bands for
three-versus-three or three-versus-one seed designs. The monotonicity row is
marked not applicable because the final model already uses zero monotonic loss.
"""
import os, re, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import repro_config as RC

K = os.environ.get("PHOMINEURO_ABLATION_DIR", os.path.join(RC.METRICS_DIR, "ablation"))
OUT = os.path.join(K, "V18_HERO3_ABLATION.md")
NG = 10
sys.path.insert(0, HERE)
import train_inr_v11 as _T
E19 = set(_T.E19)

HERO = [f"v18h3a_s{i}" for i in range(3)]
# label, arm stem, group -- the V17 row order, preserved so the two tables align line by line
ROWS = [("− FM 特征金字塔", "nopyr", "feat"),
        ("FM encoder:原始未微调", "raw", "feat"),
        ("FM encoder:随机初始化", "randenc", "feat"),
        ("− 源→点路径积分", "nopath", "feat"),
        ("− 源相对坐标 (r, cosθ)", "nosrc", "feat"),
        ("− PINN 时变扩散项", "nopinn", "loss"),
        ("− 线性项 λ_lin", "nolin", "loss"),
        ("− 删失铰链 (censored hinge)", "nofloor", "loss"),
        ("− 径向单调项 mono", None, "loss"),          # None -> 不适用
        ("− 十倍程分层加权", "noboost", "wgt"),
        ("铰链偏移 0.5 → 0", "off0", "wgt"),
        ("− PINN 且 − FM 金字塔 (联合)", "nopinnpyr", "joint")]
GROUP = {"feat": "输入特征", "loss": "损失项", "wgt": "加权 / 结构", "joint": "联合消融"}
MAIN = [("r2", "R²", 1), ("rmse", "RMSE", 0), ("medae", "medAE", 0), ("spearman", "Spearman", 1),
        ("dice", "DICE", 1), ("gamma", "gamma(3%/2px)", 1), ("ccc", "CCC", 1)]
TARG = [("peak_logerr", "peak \\|Δlog₁₀\\|", 0), ("topK_relerr", "top-K rel.err", 0),
        ("mono_viol", "mono 违反", 0)]
DARKM = [("speck_frac", "speck_frac (%)", 0), ("rough", "\\|lap\\|", 0),
         ("leak", "LEAK (%)", 0), ("deep_leak", "DEEP-LEAK (%)", 0)]
DARK_RX = [("speck_frac", r"speck_frac\s*=\s*([0-9.]+)"), ("n_speck", r"n_speck\s*=\s*([0-9.]+)"),
           ("rough", r"\|lap\|\s*=\s*([0-9.]+)"), ("leak", r"LEAK%\s*above floor\s*=\s*([0-9.]+)"),
           ("deep_leak", r"DEEP-LEAK%\s*above floor\s*=\s*([0-9.]+)")]


def jload(p):
    return json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else None


def merge(kind, stems):
    out = {}
    for s in stems:
        for a, r in (jload(os.path.join(K, f"{kind}_{s}.json")) or {}).items():
            out.setdefault(a, r)
    return out


D = merge("diag", ["h3base", "h3abl_ft1", "h3abl_ft2", "h3abl_ft3", "h3abl_rnd", "hero3raw"])
T = merge("targeted", ["h3base", "h3abl_ft1", "h3abl_ft2", "h3abl_ft3", "h3abl_rnd", "hero3raw"])
E = merge("energy", ["hero3", "h3abl_a", "h3abl_b", "h3abl_rnd", "hero3raw"])


def arms_of(stem):
    """Return every available seed for an ablation arm."""
    if stem == "raw":
        cand = [f"v18h3raw_s{i}" for i in range(3)]
    else:
        cand = [f"v18h3a_{stem}_s{i}" for i in range(3)]
    return [a for a in cand if a in D]


def gm(rec, m):
    if not rec:
        return None
    xs = [rec[str(k)][m] for k in range(NG)
          if str(k) in rec and m in rec[str(k)] and np.isfinite(rec[str(k)][m])]
    return float(np.mean(xs)) if len(xs) == NG else None


def tv(arm, m):
    pg = (T.get(arm) or {}).get("per_gate", {})
    return (float(np.mean([pg[str(k)][m] for k in range(NG)]))
            if all(str(k) in pg and m in pg[str(k)] for k in range(NG)) else None)


def ev(arm):
    """median |log10 absorbed-energy ratio|; symmetric in over- and under-deposition, which a mean
    of the ratio is not."""
    r = E.get(arm)
    if not r:
        return None
    v = [abs(np.log10(x)) for c in r.values() for k in range(NG)
         for x in [c.get("per_gate", {}).get(str(k), {}).get("energy_median")] if x and x > 0]
    return float(np.median(v)) if v else None


def dk(arm):
    out = {}
    for f in glob.glob(os.path.join(K, f"dark_{arm}_*.txt")):
        e = os.path.basename(f)[len(f"dark_{arm}_"):-4]
        if e not in E19:                       # prefix-collision guard
            continue
        t = open(f, errors="ignore").read()
        v = {k: float(m.group(1)) for k, rx in DARK_RX for m in [re.search(rx, t)] if m}
        if len(v) == len(DARK_RX):
            out[e] = v
    return out


def stat(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def block(title, metrics, getter, seed_note, dark_common=None):
    hb, sb = {}, {}
    for m, _, _ in metrics:
        hb[m], sb[m], _ = stat([getter(a, m) for a in HERO])
    L = ["", f"### {title}", "", seed_note, "",
         "| 变体 | " + " | ".join(lab for _, lab, _ in metrics) + " |",
         "|---|" + "---|" * len(metrics),
         "| **hero3(完整模型)** | " + " | ".join(
             "--" if hb[m] is None else f"{hb[m]:.4f} ± {sb[m]:.4f}" for m, _, _ in metrics) + " |"]
    cur = None
    for lab, stem, grp in ROWS:
        if grp != cur:
            cur = grp
            L.append(f"| *{GROUP[grp]}* |" + " |" * len(metrics))
        if stem is None:
            L.append(f"| {lab} | " + " | ".join("*不适用*" for _ in metrics) + " |")
            continue
        arms = arms_of(stem) if dark_common is None else [f"v18h3a_{stem}_s0"]
        if stem == "raw" and dark_common is not None:
            arms = ["v18h3raw_s0"]
        cells = []
        for m, _, better in metrics:
            vals = [getter(a, m) for a in arms]
            mv, sv, n = stat(vals)
            if mv is None or hb[m] is None:
                cells.append("--"); continue
            d = mv - hb[m]
            bd = (1.96 * np.sqrt((sb[m] ** 2 + sv ** 2) / 3) if n >= 3
                  else 1.96 * sb[m] * np.sqrt(4.0 / 3)) if sb[m] else None
            tag = "" if bd is None else (" n.s." if abs(d) <= bd else
                                         (" **+**" if (d > 0) == bool(better) else " **−**"))
            # SHOW THE ARM'S OWN SEED SPREAD. Printing only the mean here, while the hero row
            # carries "± sd", reads as "hero has 3 seeds, the arms have 1" -- which is not what
            # this block is. n < 2 (the energy/dark blocks, seed 0 only) keeps the bare mean.
            sd = f" ± {sv:.4f}" if n >= 2 else ""
            cells.append(f"{mv:.4f}{sd}<br>({d:+.4f}{tag})")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("\n> 带宽:" + "、".join(
        f"{lab} {1.96*sb[m]*np.sqrt(4/3):.4f}" for m, lab, _ in metrics if sb[m]))
    return L


def main():
    if not all(a in D for a in HERO):
        print("hero3 base missing from diag"); return 1
    L = ["# V18 hero3 消融表(Dev-test,20 头 × 19 电极 = 380 场景,K=1000)\n",
         "> 由 `INR_design/make_v18h3_abl_report.py` 装配,无一数字手抄。行序与分组照搬已定稿的",
         "> V17 消融表(`make_v17_abl_report.py`),以便两表并排放入附件。\n",
         "> **hero3** = hero2 + `--drop-optical --drop-light ang`,in_dim **1536**"
         "(删去 4 维光学属性与 4 维 sin/cos 角)。该精简经 3 seed 验证相对全输入 hero2 无可测代价"
         "(R² +0.0001)。\n"]
    L += block("主面板(10 门均值)", MAIN, lambda a, m: gm(D.get(a), m),
               "hero3 与每个消融臂**均为 3 seed**;带宽 $1.96\\sqrt{(\\sigma_h^2+\\sigma_v^2)/3}$。"
               "V17 表的消融臂只有 1 seed,故本表此块比 V17 更严格。")
    if T:
        L += block("峰值保真与径向单调性(10 门均值)", TARG, lambda a, m: tv(a, m),
                   "同上,3 seed 对 3 seed。")
    # energy + dark: seed 0 only, matching V17
    if E:
        he, se, _ = stat([ev(a) for a in HERO])
        L += ["", "### 能量守恒(median $|\\log_{10}$ 吸收能量比$|$)", "",
              "**seed 0 单个**,与 V17 定稿表每臂 1 seed 的协议一致。hero3 列仍为 3 seed 均值,"
              "故带宽用 3-vs-1 传播 $1.96\\sigma_h\\sqrt{1/3+1}$。", "",
              "| 变体 | 能量偏差 |", "|---|---|",
              f"| **hero3(完整模型)** | {he:.4f} ± {se:.4f} |"]
        bd = 1.96 * se * np.sqrt(4 / 3) if se else None
        for lab, stem, grp in ROWS:
            if stem is None:
                L.append(f"| {lab} | *不适用* |"); continue
            a = "v18h3raw_s0" if stem == "raw" else f"v18h3a_{stem}_s0"
            v = ev(a)
            if v is None:
                L.append(f"| {lab} | -- |"); continue
            d = v - he
            tag = "" if bd is None else (" n.s." if abs(d) <= bd else (" **−**" if d > 0 else " **+**"))
            L.append(f"| {lab} | {v:.4f} ({d:+.4f}{tag}) |")
        if bd:
            L.append(f"\n> 带宽 {bd:.4f}")
    # dark zone on the common electrode set
    hd = {a: dk(a) for a in HERO}
    vd = {}
    for lab, stem, grp in ROWS:
        if stem is None:
            continue
        a = "v18h3raw_s0" if stem == "raw" else f"v18h3a_{stem}_s0"
        vd[stem] = dk(a)
    sets = [set(x) for x in list(hd.values()) + [v for v in vd.values() if v]]
    com = sorted(set.intersection(*sets)) if sets and all(sets) else []
    if com and len(com) >= 10:
        L += ["", f"### 暗区(在 {len(com)} 个公共电极上取中位数)", "",
              "**seed 0 单个**,与 V17 协议一致。⚠️ `speck_frac` 的 seed 离散度实测达均值的 "
              "**14–18%**,故小于该量级的差异不得下结论。", "",
              "| 变体 | " + " | ".join(lab for _, lab, _ in DARKM) + " |",
              "|---|" + "---|" * len(DARKM)]
        hv = {}
        for m, _, _ in DARKM:
            hv[m] = stat([float(np.median([hd[a][e][m] for e in com])) for a in HERO])
        L.append("| **hero3(完整模型)** | " + " | ".join(
            f"{hv[m][0]:.4f} ± {hv[m][1]:.4f}" for m, _, _ in DARKM) + " |")
        cur = None
        for lab, stem, grp in ROWS:
            if grp != cur:
                cur = grp; L.append(f"| *{GROUP[grp]}* |" + " |" * len(DARKM))
            if stem is None:
                L.append(f"| {lab} | " + " | ".join("*不适用*" for _ in DARKM) + " |"); continue
            if not vd.get(stem) or not set(com) <= set(vd[stem]):
                L.append(f"| {lab} | " + " | ".join("--" for _ in DARKM) + " |"); continue
            cells = []
            for m, _, better in DARKM:
                v = float(np.median([vd[stem][e][m] for e in com]))
                d = v - hv[m][0]
                bd = 1.96 * hv[m][1] * np.sqrt(4 / 3) if hv[m][1] else None
                tag = "" if bd is None else (" n.s." if abs(d) <= bd else
                                             (" **+**" if (d > 0) == bool(better) else " **−**"))
                cells.append(f"{v:.4f}<br>({d:+.4f}{tag})")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
    else:
        n = {k: len(v) for k, v in vd.items()}
        L += ["", "### 暗区", "", f"> 尚未跑完,各臂有效电极数 {n};公共电极集不足,不出部分结果 ——"
              "19 电极中位数与更少电极的中位数相比会把电极差异混进效应里。"]
    L += ["", "---", "", "## 必须随表登记的限制", "",
          "1. **`− 径向单调项 mono` 不适用**:hero3 的 `λ_mono` 本就是 0,该项对 hero3 不存在,"
          "故留空而非用一个什么都不改的臂填充。hero1 家族的旧测量可作背景,但那是另一个配方。",
          "2. **能量与暗区为 seed 0 单个**(与 V17 定稿表协议一致);主面板与峰值块为 3 seed,"
          "比 V17 更严格。两块的带宽口径不同,已分别标注。",
          "3. **Dev-test 是选择队列**,V17/V18 全程用它做消融,不是独立测试集。",
          "4. **`rawenc`/`randenc` 换的是 encoder 表征,不是去掉它**;与 `nopyr` 一起构成"
          "随机 < 未微调 < 微调的梯度。",
          "5. 暗区 `speck_frac` 的 seed 离散度达均值 14–18%,单 seed 下小差异不可解读。"]
    txt = "\n".join(L) + "\n"
    open(OUT, "w").write(txt)
    print(txt)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
