#!/usr/bin/env python3
"""Time-resolved MCX fluence simulation with the OpenCL ``pmcxcl`` backend.

The loader preserves the per-voxel ``(mu_a, mu_s, g, n)`` mapping, detects
channel-first inputs, fills enclosed low-index cavities with nearest CSF
properties, and caches the MCX label/property representation per head.
"""

import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
import os
import argparse
import time
import sys
import hashlib
import json

# ──────────────────────────────────────────────────────────────
# pmcxcl loader
# ──────────────────────────────────────────────────────────────
import importlib.util


def _load_pmcxcl():
    """Load pmcxcl normally, or from ``PMCXCL_LIBRARY`` when explicitly set."""
    try:
        import pmcxcl
        return pmcxcl
    except ImportError as exc:
        library = os.environ.get("PMCXCL_LIBRARY")
        if not library:
            class MissingPmcxcl:
                @staticmethod
                def run(*args, **kwargs):
                    raise ImportError(
                        "pmcxcl is required for Monte Carlo simulation. Install pmcxcl or "
                        "set PMCXCL_LIBRARY to its compiled extension."
                    )

            return MissingPmcxcl()
        spec = importlib.util.spec_from_file_location("_pmcxcl", library)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load pmcxcl extension: {library}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


pmcx = _load_pmcxcl()


# ──────────────────────────────────────────────────────────────
# 路径 / 默认参数
# ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import repro_config as RC

DATASET_DIR = RC.DATASET_DIR
PROJECT_DIR = RC.SIM_ROOT
OUTPUT_DIR = os.path.join(PROJECT_DIR, "sim_t1")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")

WAVELENGTH_LIST = ["670", "810", "850", "980", "1064"]
WAVELENGTH      = "810"

# 光源默认方向角（可被 CLI 覆盖）
SRC_THETA_DEG = 90.0
SRC_PHI_DEG   = 90.0
SRC_RADIUS_MM = 10.0
SRC_R_INI     = 50        # 初始搜索半径（体素）
SRC_R_MARGIN  = 15        # 确认出界后再向外追加的余量

NPHOTON   = 5e6
GPU_ID    = 1
TIME_GATE = 5e-8

# CSF 各波长光学参数（用于填充空腔默认值，若空腔体素无有效参数）
# 格式: [mua, mus, g, n]   单位: mm⁻¹
CSF_PROPS = {
    "670" : [0.0002, 0.10, 0.90, 1.33],
    "810" : [0.0002, 0.10, 0.90, 1.33],
    "850" : [0.0002, 0.10, 0.90, 1.33],
    "980" : [0.0025, 0.10, 0.90, 1.33],
    "1064": [0.0002, 0.10, 0.90, 1.33],
}


# ══════════════════════════════════════════════════════════════
# 1.  数据加载 & 轴顺序修正
# ══════════════════════════════════════════════════════════════

def load_vol4d(wavelength: str, sample_id: str = "ADT1") -> np.ndarray:
    """
    从 .mat 文件加载 vol_4D，并确保返回 shape 为 (X, Y, Z, 4)。

    MATLAB 保存时是 column-major，scipy.io.loadmat 读入后各空间轴顺序已经过
    转置（第一轴对应 MATLAB 的第一维，即 X），通常 shape 为 (X, Y, Z, 4)。
    但若 shape[0] == 4，说明光学参数轴被放到了最前面，需要 moveaxis 修正。
    """
    prop_path = os.path.join(
        DATASET_DIR, f"{sample_id}_copmri_withHermiteF{wavelength}.mat"
    )
    if not os.path.isfile(prop_path):
        raise FileNotFoundError(f"Property file not found: {prop_path}")

    print(f"[load] {prop_path}")
    mat   = sio.loadmat(prop_path)
    key   = "vol_prop_eye_aseg"
    if key not in mat:
        user_keys = [k for k in mat if not k.startswith("_")]
        print(f"  [warn] '{key}' not found, using: {user_keys[0]}")
        key = user_keys[0]

    vol_4D = mat[key].astype(np.float64)
    print(f"  raw shape from loadmat: {vol_4D.shape}")

    # ── 轴顺序自动修正 ──
    # 期望: (X, Y, Z, 4)，若光学参数维在轴0则转置
    if vol_4D.ndim == 4 and vol_4D.shape[0] == 4 and vol_4D.shape[0] != vol_4D.shape[3]:
        vol_4D = np.moveaxis(vol_4D, 0, -1)   # (4,X,Y,Z) → (X,Y,Z,4)
        print(f"  [fix] axis reordered → {vol_4D.shape}")
    elif vol_4D.ndim == 4 and vol_4D.shape[-1] == 4:
        pass  # 已经是 (X,Y,Z,4)，正常
    else:
        raise ValueError(
            f"Unexpected vol_4D shape {vol_4D.shape}. "
            "Expected (X,Y,Z,4) or (4,X,Y,Z)."
        )

    print(f"  vol_4D shape (final): {vol_4D.shape}")

    # 快速验证：折射率 n（channel 3）的典型范围应在 1.0 ~ 1.8
    n_ch = vol_4D[..., 3]
    valid = n_ch[(n_ch > 0.5) & (n_ch < 2.5)]
    if len(valid) > 0:
        print(f"  n-channel stats: min={valid.min():.3f}  max={valid.max():.3f}"
              f"  mean={valid.mean():.3f}")
    return vol_4D


# ══════════════════════════════════════════════════════════════
# 2.  Mask 构建：外部二值化 + 内部空腔填充
# ══════════════════════════════════════════════════════════════

def build_filled_mask(vol_4D: np.ndarray, n_threshold: float = 1.05) -> np.ndarray:
    """
    构建包含内部空腔的头部完整 mask。

    步骤
    ────
    1. 以折射率通道 n > threshold 做初始二值化，得到 raw_mask
       （颅内空腔如脑室、眼球内腔因 n ≈ 1.33 > 1.05，通常已包含在内；
        若某些空腔 n ≤ 1.05，则通过下一步填充）
    2. 取最大连通域（去除头部以外的孤立噪声体素）
    3. 对最大连通域做逐层/三维 binary_fill_holes，将完全被组织包围的
       空腔体素填入 mask（这正是 MATLAB discrete_model 中 Cavity 标签
       所覆盖的区域）
    4. 返回 filled_mask (bool, shape X×Y×Z)
    """
    n_ch     = vol_4D[..., 3]
    raw_mask = (n_ch > n_threshold)

    # ── Step 2: 取最大连通域 ──
    labeled, n_comp = ndi.label(raw_mask)
    if n_comp == 0:
        raise RuntimeError("No tissue voxels found! Check n_threshold or vol_4D axes.")
    comp_sizes = ndi.sum(raw_mask, labeled, range(1, n_comp + 1))
    largest    = int(np.argmax(comp_sizes)) + 1
    largest_cc = (labeled == largest)
    print(f"  [mask] {n_comp} components found; largest has "
          f"{largest_cc.sum():,} voxels")

    # ── Step 3: 三维空洞填充（包含所有内部空腔）──
    filled_mask = ndi.binary_fill_holes(largest_cc)
    n_cavity    = int(filled_mask.sum()) - int(largest_cc.sum())
    print(f"  [mask] {n_cavity:,} cavity voxels added by fill_holes")

    return filled_mask


# ══════════════════════════════════════════════════════════════
# 3.  cfg_vol / cfg_prop 构建（完全复现 MATLAB 逻辑）
# ══════════════════════════════════════════════════════════════

def build_cfg_exact(
    vol_4D     : np.ndarray,
    filled_mask: np.ndarray,
    wavelength : str,
) -> tuple:
    """
    与 MATLAB 脚本完全等价的标签-属性映射。

    MATLAB 关键逻辑（精简版）
    ──────────────────────────
        label = 0
        for i,j,k in all_voxels:
            if discrete_model(i,j,k)==0 and vol_4D(i,j,k,1 or 2) < 1e-5:
                continue            ← 跳过真背景
            label += 1
            cfg.vol(i,j,k) = label
            cfg.prop(label+1,:) = vol_4D(i,j,k,:)

    Python 等价版本
    ────────────────
    - 迭代顺序: numpy 默认 C-order (i→j→k)，与 MATLAB 的 for i for j for k 一致
    - 筛选条件: filled_mask 已包含空腔；对空腔体素若 mua/mus 为 0，
                使用该波长 CSF 默认参数填充（与 MATLAB 中 Cavity 赋参数等价）
    - cfg.prop 行 0: [0,0,1,1] 背景（与 MATLAB 一致）
    - cfg.prop 行 label: vol_4D[x,y,z,:] （直接用体素坐标索引）

    Returns
    ───────
    cfg_vol  : np.ndarray uint32 (X,Y,Z)  — 标签体积
    cfg_prop : np.ndarray float64 (N+1,4) — 第0行背景，第1..N行各体素属性
    coords   : np.ndarray int32   (N,3)   — 每个标签对应的体素坐标（调试用）
    """
    vol_size = vol_4D.shape[:3]
    csf_prop = np.array(CSF_PROPS[wavelength], dtype=np.float64)

    # 获取 mask 内全部体素坐标（C-order，与 MATLAB for i for j for k 一致）
    xs, ys, zs = np.where(filled_mask)          # C-order: x changes slowest
    n_valid     = len(xs)
    print(f"  [build] {n_valid:,} mask voxels → building label table...")

    cfg_vol  = np.zeros(vol_size, dtype=np.uint32)
    cfg_prop = np.zeros((n_valid + 1, 4), dtype=np.float64)
    cfg_prop[0] = [0.0, 0.0, 1.0, 1.0]    # label=0: 背景（真空）

    # 向量化提取所有体素的 4 通道光学参数
    # 关键: 用 (xs, ys, zs) 作为坐标索引，保证每个体素的参数严格来自其空间位置
    props_all = vol_4D[xs, ys, zs, :]     # shape (n_valid, 4)

    # 对空腔体素（mua≈0 且 mus≈0，即 discrete_model==Cavity 的区域）用 CSF 参数填充
    # 判断：mua < 1e-5 AND mus < 1e-5（与 MATLAB skip 条件取反并赋默认值）
    cavity_vox = (props_all[:, 0] < 1e-5) & (props_all[:, 1] < 1e-5)
    if cavity_vox.sum() > 0:
        print(f"  [build] {cavity_vox.sum():,} cavity voxels → assigned CSF props {csf_prop}")
        props_all[cavity_vox] = csf_prop

    # 赋标签（从 1 开始，label=k 对应 cfg_prop[k]，即偏移量为 +1）
    labels = np.arange(1, n_valid + 1, dtype=np.uint32)
    cfg_vol[xs, ys, zs]    = labels
    cfg_prop[1:, :]        = props_all

    # 坐标表（用于缓存和调试）
    coords = np.stack([xs, ys, zs], axis=1).astype(np.int32)

    print(f"  [build] cfg_vol nonzero: {(cfg_vol > 0).sum():,}")
    print(f"  [build] cfg_prop rows: {len(cfg_prop):,}")

    # 光学参数合理性快速报告
    _report_prop_stats(cfg_prop[1:])

    return cfg_vol, cfg_prop, coords


def build_cfg_discrete(vol_4D, filled_mask, wavelength):
    """FEW-MEDIA discrete cfg: background = label 0, each distinct tissue optic = label 1..N.

    Physically identical input to ``build_cfg_exact`` (same geometry, same cavity->CSF fill),
    but collapses the per-voxel 1:1 media table (~3.6M rows) down to the handful of DISTINCT
    tissue media via ``np.unique``. pmcxcl (OpenCL, mediaformat=8) mis-handles the 1:1 table
    when the exterior is label 0 -- the per-voxel optics (notably the low CSF scattering) are
    NOT applied, corrupting the field by many decades. The discrete table is honoured exactly
    (validated against single-label and 2-layer-slab references up to 4.1M media: diff=0.000).
    scb heads -> 6 media (+bg), bw heads -> <=9 media (+bg).  See pmcxcl-large-media-csf-bug.
    """
    vol_size = vol_4D.shape[:3]
    csf_prop = np.array(CSF_PROPS[wavelength], dtype=np.float64)
    xs, ys, zs = np.where(filled_mask)
    n_valid = len(xs)
    props_all = vol_4D[xs, ys, zs, :].astype(np.float64)

    # cavity voxels (mua<1e-5 & mus<1e-5) -> CSF, identical to build_cfg_exact
    cavity_vox = (props_all[:, 0] < 1e-5) & (props_all[:, 1] < 1e-5)
    if cavity_vox.sum() > 0:
        print(f"  [build-discrete] {cavity_vox.sum():,} cavity voxels -> CSF props {csf_prop}")
        props_all[cavity_vox] = csf_prop

    rows, inv = np.unique(props_all, axis=0, return_inverse=True)   # distinct tissue media
    # CRITICAL: label volume MUST be uint8 (or int32). pmcxcl mis-handles uint32 labels
    # (mediaformat=8 path) -> it collapses the whole volume to a SINGLE medium (verified:
    # split phantom transparent-vs-scattering gave dL-R=0 for uint32 but +14.7 dec for
    # uint8/int32). Discrete heads have <=9 media, so uint8 is exact. See pmcxcl bug memo.
    if len(rows) > 255:
        raise ValueError(f"{len(rows)} media > 255; uint8 label volume insufficient")
    cfg_vol = np.zeros(vol_size, dtype=np.uint8)
    cfg_vol[xs, ys, zs] = (inv + 1).astype(np.uint8)              # mask voxels -> labels 1..N
    cfg_prop = np.zeros((len(rows) + 1, 4), dtype=np.float64)
    cfg_prop[0] = [0.0, 0.0, 1.0, 1.0]                            # label 0 = exterior/vacuum
    cfg_prop[1:] = rows
    coords = np.stack([xs, ys, zs], axis=1).astype(np.int32)

    print(f"  [build-discrete] {n_valid:,} mask voxels, {len(rows)} discrete media (+bg)")
    _report_prop_stats(cfg_prop[1:])
    return cfg_vol, cfg_prop, coords


def _report_prop_stats(props: np.ndarray):
    """打印 mua / mus / g / n 的统计，用于验证不同组织参数是否正确区分。"""
    names = ["mua(mm⁻¹)", "mus(mm⁻¹)", "g", "n"]
    print("  ┌─── Optical property statistics (all tissue voxels) ───")
    for c, name in enumerate(names):
        col = props[:, c]
        nonzero = col[col > 0]
        if len(nonzero) == 0:
            print(f"  │  {name:12s}: all zero (check data!)")
        else:
            uvals = np.unique(np.round(nonzero, 4))
            n_unique = len(uvals)
            print(f"  │  {name:12s}: min={nonzero.min():.4f}  max={nonzero.max():.4f}"
                  f"  mean={nonzero.mean():.4f}  unique≈{n_unique:,}")
    print("  └────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════
# 4.  缓存 I/O
# ══════════════════════════════════════════════════════════════

def _cache_tag(sample_id: str, wavelength: str) -> str:
    return f"{sample_id}_F{wavelength}"


def save_cache(
    sample_id: str,
    wavelength: str,
    cfg_vol: np.ndarray,
    cfg_prop: np.ndarray,
    coords: np.ndarray,
    filled_mask: np.ndarray,
    origin: np.ndarray,
):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag  = _cache_tag(sample_id, wavelength)
    path = os.path.join(CACHE_DIR, f"{tag}_cfg.npz")
    np.savez_compressed(
        path,
        cfg_vol=cfg_vol,
        cfg_prop=cfg_prop,
        coords=coords,
        filled_mask=filled_mask,
        origin=origin,
    )
    print(f"  [cache] saved → {path}")


def load_cache(sample_id: str, wavelength: str):
    tag  = _cache_tag(sample_id, wavelength)
    path = os.path.join(CACHE_DIR, f"{tag}_cfg.npz")
    if not os.path.isfile(path):
        return None
    print(f"  [cache] loading → {path}")
    d = np.load(path)
    return (
        d["cfg_vol"],
        d["cfg_prop"],
        d["coords"],
        d["filled_mask"],
        d["origin"],
    )


# ══════════════════════════════════════════════════════════════
# 5.  光源方向 & 位置
# ══════════════════════════════════════════════════════════════

def compute_src_dir(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = np.radians(theta_deg)
    phi   = np.radians(phi_deg)
    raw   = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    return -raw / np.linalg.norm(raw)   # 指向颅内


def compute_mask_centroid(filled_mask: np.ndarray) -> np.ndarray:
    """mask 几何中心（与 MATLAB 的 O = MNI 空间原点等效）。"""
    coords = np.argwhere(filled_mask)
    return coords.mean(axis=0)


def find_source_position(
    cfg_vol : np.ndarray,
    vol_size: tuple,
    src_dir : np.ndarray,
    origin  : np.ndarray,
    r_ini   : int = SRC_R_INI,
    margin  : int = SRC_R_MARGIN,
) -> np.ndarray:
    """
    沿 -src_dir 方向从 origin 向外搜索，找到第一个 cfg_vol==0（空气）
    且其后 10 步也均为空气的位置，再加 margin 体素余量。

    与 MATLAB 的双层 while 循环逻辑完全对应。
    """
    r     = float(r_ini)
    max_r = int(max(vol_size) * 2)

    while r < max_r:
        OS = np.round(r * (-src_dir)).astype(int)
        S  = (origin + OS).astype(int)
        if not all(0 <= S[i] < vol_size[i] for i in range(3)):
            break
        if cfg_vol[S[0], S[1], S[2]] == 0:
            # 确认接下来 10 步也在空气中
            clear = True
            for step in range(1, 11):
                Sc = np.round(origin + (r + step) * (-src_dir)).astype(int)
                if not all(0 <= Sc[i] < vol_size[i] for i in range(3)):
                    break
                if cfg_vol[Sc[0], Sc[1], Sc[2]] != 0:
                    clear = False
                    break
            if clear:
                break
        r += 1

    r  += margin
    OS  = np.round(r * (-src_dir)).astype(int)
    S   = np.clip((origin + OS).astype(int), 0, np.array(vol_size) - 1)
    print(f"  [src] position={S.tolist()}  r={r:.1f} vox")
    return S


# ══════════════════════════════════════════════════════════════
# 6.  主仿真函数
# ══════════════════════════════════════════════════════════════

def run_simulation(
    wavelength      : str   = WAVELENGTH,
    theta_deg       : float = SRC_THETA_DEG,
    phi_deg         : float = SRC_PHI_DEG,
    radius_mm       : float = SRC_RADIUS_MM,
    r_ini           : int   = SRC_R_INI,
    margin          : int   = SRC_R_MARGIN,
    nphoton         : float = NPHOTON,
    gpu_id          : int   = GPU_ID,
    output_dir      : str   = OUTPUT_DIR,
    save_mat        : bool  = True,
    sample_id       : str   = "ADT1",
    force_rebuild   : bool  = False,
    origin_override : tuple = None,
) -> tuple:
    t0 = time.time()

    # ── 尝试从缓存加载 ──
    cached = None if force_rebuild else load_cache(sample_id, wavelength)

    if cached is not None:
        cfg_vol, cfg_prop, coords, filled_mask, origin = cached
        vol_size = cfg_vol.shape
        print(f"  [cache] cfg_vol shape: {vol_size}")
    else:
        vol_4D = load_vol4d(wavelength, sample_id)
        vol_size = vol_4D.shape[:3]

        print("[mask] Building filled mask (largest CC + fill_holes)...")
        filled_mask = build_filled_mask(vol_4D)

        print("[build] Building cfg_vol / cfg_prop (exact 1:1 mapping)...")
        t1 = time.time()
        cfg_vol, cfg_prop, coords = build_cfg_exact(vol_4D, filled_mask, wavelength)
        print(f"  build time: {time.time()-t1:.1f}s")

        # origin: 使用 mask 几何中心（或 MNI 空间原点如已知）
        if origin_override is not None:
            origin = np.array(origin_override, dtype=float)
        else:
            origin = compute_mask_centroid(filled_mask)
        print(f"  [origin] {origin}")

        save_cache(sample_id, wavelength, cfg_vol, cfg_prop, coords, filled_mask, origin)

    if origin_override is not None:
        origin = np.array(origin_override, dtype=float)

    # ── 光源方向 & 位置 ──
    src_dir = compute_src_dir(theta_deg, phi_deg)
    src_pos = find_source_position(cfg_vol, vol_size, src_dir, origin, r_ini, margin)

    # ── MCX 配置（与 MATLAB cfg 字段完全对应）──
    cfg = {
        # 几何
        "vol"        : cfg_vol,
        "prop"       : cfg_prop,

        # 光源
        "srcpos"     : src_pos.astype(float).tolist(),
        "srcdir"     : src_dir.tolist(),
        "srctype"    : "disk",
        "srcparam1"  : [float(radius_mm), 0.0, 0.0, 0.0],

        # 时间门
        "tstart"     : 0.0,
        "tend"       : float(TIME_GATE),
        "tstep"      : float(TIME_GATE),

        # 仿真控制
        "issrcfrom0" : 1,
        "seed"       : 29012392,
        "nphoton"    : int(nphoton),
        "unitinmm"   : 1.0,

        # 输出
        "outputtype" : "fluence",       # 与 MATLAB otype='fluence' 一致

        # 边界
        "isreflect"  : 0,               # 与 MATLAB cfg.isreflect=0 一致
        "isspecular" : 0,               # 与 MATLAB cfg.isspecular=0 一致

        # GPU
        "gpuid"      : gpu_id,
        "autopilot"  : 1,
    }

    print(f"\n[run]  λ={wavelength}nm  θ={theta_deg}°  φ={phi_deg}°"
          f"  r_disk={radius_mm}mm  nphoton={int(nphoton):.2e}")
    print(f"  srcpos  = {src_pos.tolist()}")
    print(f"  srcdir  = [{src_dir[0]:.4f}, {src_dir[1]:.4f}, {src_dir[2]:.4f}]")
    print(f"  origin  = {origin.tolist()}")

    t2  = time.time()
    res = pmcx.run(cfg)
    print(f"  sim time: {time.time()-t2:.1f}s")

    # ── 后处理（与 MATLAB 完全一致）──
    #   MATLAB: fluence → odata * 1000   (mJ/mm²)
    flux    = res["flux"]
    if flux.ndim == 4 and flux.shape[3] == 1:
        flux = flux[:, :, :, 0]
    fluence = flux * 1000.0    # mJ/mm²

    # ── 保存 ──
    if save_mat:
        os.makedirs(output_dir, exist_ok=True)
        tag      = f"{sample_id}_F{wavelength}_th{theta_deg:.0f}_ph{phi_deg:.0f}_r{radius_mm:.0f}"
        out_path = os.path.join(output_dir, f"fluence_{tag}.mat")
        sio.savemat(out_path, {
            "fluence"   : fluence,
            "wavelength": wavelength,
            "theta_deg" : theta_deg,
            "phi_deg"   : phi_deg,
            "radius_mm" : radius_mm,
            "srcpos"    : src_pos,
            "srcdir"    : src_dir,
            "vol_size"  : np.array(vol_size),
            "origin"    : origin,
        })
        print(f"  [save] {out_path}")

        # 同时保存关键光源参数（便于外部脚本读取）
        meta = {
            "sample_id" : sample_id,
            "wavelength": wavelength,
            "theta_deg" : float(theta_deg),
            "phi_deg"   : float(phi_deg),
            "radius_mm" : float(radius_mm),
            "srcpos"    : src_pos.tolist(),
            "srcdir"    : src_dir.tolist(),
            "origin"    : origin.tolist(),
            "nphoton"   : int(nphoton),
        }
        meta_path = os.path.join(output_dir, f"meta_{tag}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [save] {meta_path}")

    nz = fluence[fluence > 0]
    print(f"\n[done] total={time.time()-t0:.1f}s | "
          f"fluence shape={fluence.shape} | "
          f"max={fluence.max():.3e}  mean(nz)={nz.mean():.3e} mJ/mm²")

    return fluence, cfg


# ══════════════════════════════════════════════════════════════
# 7.  CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="MCX Fluence Simulation v2 (exact per-voxel mapping + cavity fix)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wl",     default=WAVELENGTH,    choices=WAVELENGTH_LIST)
    p.add_argument("--theta",  default=SRC_THETA_DEG, type=float,
                   help="polar angle θ (deg)")
    p.add_argument("--phi",    default=SRC_PHI_DEG,   type=float,
                   help="azimuthal angle φ (deg)")
    p.add_argument("--radius", default=SRC_RADIUS_MM, type=float,
                   help="disk source radius (mm)")
    p.add_argument("--r_ini",  default=SRC_R_INI,     type=int)
    p.add_argument("--margin", default=SRC_R_MARGIN,  type=int)
    p.add_argument("--nphoton",default=NPHOTON,        type=float)
    p.add_argument("--gpu",    default=GPU_ID,         type=int)
    p.add_argument("--outdir", default=OUTPUT_DIR)
    p.add_argument("--sample", default="ADT1",         type=str)
    p.add_argument("--no-save",    action="store_true")
    p.add_argument("--force-rebuild", action="store_true",
                   help="忽略缓存，重新构建 cfg_vol/cfg_prop")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        wavelength    = args.wl,
        theta_deg     = args.theta,
        phi_deg       = args.phi,
        radius_mm     = args.radius,
        r_ini         = args.r_ini,
        margin        = args.margin,
        nphoton       = args.nphoton,
        gpu_id        = args.gpu,
        output_dir    = args.outdir,
        save_mat      = not args.no_save,
        sample_id     = args.sample,
        force_rebuild = args.force_rebuild,
    )
