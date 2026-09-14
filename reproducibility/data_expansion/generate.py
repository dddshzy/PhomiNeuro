"""
Dataset expansion orchestration + on-disk scene IO.

Writes every generated scene in the EXACT layout discover_scenes()/SceneStore
read (so they join the training set once pyramids are extracted in the post-FM
re-coherence chain):

  property : {DATASET_DIR}/{sample}_copmri_withHermiteF{wl}.mat   key 'vol_prop_eye_aseg'  (X,Y,Z,4)
  fluence  : {SIM_DIR}/wl{wl}/fluence_{sample}_F{wl}_th75_ph75_r10.mat   key 'fluence'      (X,Y,Z)
  meta     : {SIM_DIR}/wl{wl}/meta_{sample}_F{wl}_th75_ph75_r10.json     {srcpos, srcdir, ...}

The `th75_ph75_r10` suffix is the constant tag discover_scenes expects; the ACTUAL
(varied) illumination lives in the meta json (srcpos/srcdir), which is what the
pipeline reads.  Per-scene illumination variety is namespaced by unique `sample`.

Two routes:
  - augment_heads : rigid (flip/rot90) equivariant transforms of existing head
    scenes — reuses the head MCX fluence (free, no re-sim). Same units as heads.
  - generate_phantoms : virtual phantoms + forward solve. Default backend 'mcx'
    so phantom GT is in the SAME units as the head GT (avoids a cross-source scale
    bias in the pooled training set); 'diffusion' available for fast bulk (needs a
    scale calibration to MCX units — see solvers.log10_agreement offset).

All generated optical values are asserted to lie inside optical_config.PHYS_RANGE
(fail loudly — no silent clipping), per the amendment.
"""
import os
import re
import sys
import glob
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "INR_design"))
import optical_config as OC
import inr_dataset as D                      # DATASET_DIR / PYRAMID_DIR / SIM_DIR
from data_expansion import augment as AUG
from data_expansion.phantom import generate_phantom, PhantomConfig
from data_expansion.solvers import Illumination, solve_fluence

SCENE_SUFFIX = "th75_ph75_r10"               # constant tag discover_scenes expects


# ---------------------------------------------------------------------------
@dataclass
class ExpansionConfig:
    dataset_dir: str = D.DATASET_DIR
    sim_dir: str = D.SIM_DIR
    aug_per_head: int = 2
    aug_proper_only: bool = False            # include reflections (valid for RTE)
    n_phantoms: int = 16
    phantom_backend: str = "mcx"             # 'mcx' (unit-consistent) or 'diffusion'
    phantom_shape: Tuple[int, int, int] = (96, 96, 96)
    phantom_wl: str = "9999"                 # synthetic wl tag (wavelength is not an INR input)
    phantom_nphoton: float = 5e6
    seed: int = 0


# ---------------------------------------------------------------------------
# range assertion (amendment: fail loudly, no clipping)
# ---------------------------------------------------------------------------
def assert_in_range(optical_volume: np.ndarray, name: str,
                    mask: Optional[np.ndarray] = None):
    """Assert every optical value (over `mask`, default all) is inside PHYS_RANGE."""
    for c, ch in enumerate(OC.CHANNELS):
        lo, hi = OC.PHYS_RANGE[ch]
        v = optical_volume[..., c]
        v = v[mask] if mask is not None else v
        vmin, vmax = float(v.min()), float(v.max())
        if vmin < lo - 1e-9 or vmax > hi + 1e-9:
            raise ValueError(
                f"[{name}] channel '{ch}' out of PHYS_RANGE {(lo, hi)}: "
                f"got [{vmin:.5f}, {vmax:.5f}] — widen PHYS_RANGE or fix generator")


# ---------------------------------------------------------------------------
# on-disk scene writer (discover_scenes layout)
# ---------------------------------------------------------------------------
def scene_paths(sample: str, wl: str, dataset_dir: str, sim_dir: str):
    prop = os.path.join(dataset_dir, f"{sample}_copmri_withHermiteF{wl}.mat")
    wld = os.path.join(sim_dir, f"wl{wl}")
    flu = os.path.join(wld, f"fluence_{sample}_F{wl}_{SCENE_SUFFIX}.mat")
    meta = os.path.join(wld, f"meta_{sample}_F{wl}_{SCENE_SUFFIX}.json")
    return prop, flu, meta


def write_scene(sample: str, wl: str, optical_volume: np.ndarray,
                fluence: np.ndarray, srcpos, srcdir, extra_meta: dict,
                dataset_dir: str, sim_dir: str):
    assert optical_volume.shape[:3] == fluence.shape
    assert_in_range(optical_volume, f"{sample}_F{wl}")
    prop, flu, meta = scene_paths(sample, wl, dataset_dir, sim_dir)
    os.makedirs(os.path.dirname(flu), exist_ok=True)
    sio.savemat(prop, {"vol_prop_eye_aseg": optical_volume.astype(np.float32)},
                do_compression=True)
    sio.savemat(flu, {"fluence": fluence.astype(np.float32)}, do_compression=True)
    md = dict(sample_id=sample, wavelength=wl,
              srcpos=[float(x) for x in np.asarray(srcpos).tolist()],
              srcdir=[float(x) for x in np.asarray(srcdir).tolist()])
    md.update(extra_meta)
    with open(meta, "w") as f:
        json.dump(md, f, indent=2)
    return prop, flu, meta


# ---------------------------------------------------------------------------
# head loader (read an existing head scene to augment)
# ---------------------------------------------------------------------------
def list_head_scenes(dataset_dir: str, sim_dir: str) -> List[dict]:
    """Original head scenes (exclude already-generated aug/phantom samples)."""
    out = []
    for p in sorted(glob.glob(os.path.join(dataset_dir, "*_copmri_withHermiteF*.mat"))):
        base = os.path.basename(p)
        m = re.match(r"(.+)_copmri_withHermiteF(\d+)\.mat$", base)
        if not m:
            continue
        sample, wl = m.group(1), m.group(2)
        if "__aug" in sample or sample.startswith("phantom"):
            continue
        _, flu, meta = scene_paths(sample, wl, dataset_dir, sim_dir)
        if os.path.isfile(flu) and os.path.isfile(meta):
            out.append(dict(sample=sample, wl=wl, prop=p, fluence=flu, meta=meta))
    return out


def _load_head(scene: dict):
    prop = sio.loadmat(scene["prop"])["vol_prop_eye_aseg"].astype(np.float32)   # (X,Y,Z,4)
    flu = sio.loadmat(scene["fluence"])["fluence"].astype(np.float32)           # (X,Y,Z)
    with open(scene["meta"]) as f:
        meta = json.load(f)
    return prop, flu, np.array(meta["srcpos"], float), np.array(meta["srcdir"], float)


# ---------------------------------------------------------------------------
# PART A — augment heads
# ---------------------------------------------------------------------------
def augment_heads(cfg: ExpansionConfig, rng: np.random.Generator) -> List[dict]:
    heads = list_head_scenes(cfg.dataset_dir, cfg.sim_dir)
    written = []
    for sc in heads:
        prop, flu, srcpos, srcdir = _load_head(sc)
        for j, t in enumerate(AUG.sample_nonidentity(rng, cfg.aug_per_head,
                                                     cfg.aug_proper_only)):
            pT, fT, posT, dirT = AUG.transform_scene(prop, flu, srcpos, srcdir, t)
            sample = f"{sc['sample']}__aug{j:02d}"
            write_scene(sample, sc["wl"], pT, fT, posT, dirT,
                        extra_meta=dict(kind="aug", base=sc["sample"],
                                        transform=dict(perm=list(t.perm), sign=list(t.sign))),
                        dataset_dir=cfg.dataset_dir, sim_dir=cfg.sim_dir)
            written.append(dict(sample=sample, wl=sc["wl"]))
    return written


# ---------------------------------------------------------------------------
# PART B — generate phantoms
# ---------------------------------------------------------------------------
def generate_phantoms(cfg: ExpansionConfig, rng: np.random.Generator) -> List[dict]:
    pcfg = PhantomConfig(shape=cfg.phantom_shape)
    written = []
    for i in range(cfg.n_phantoms):
        optical, illum, pmeta = generate_phantom(rng, pcfg)
        kw = {} if cfg.phantom_backend == "diffusion" else dict(nphoton=cfg.phantom_nphoton)
        phi = solve_fluence(optical, illum, backend=cfg.phantom_backend, **kw)
        sample = f"phantom{i:04d}"
        write_scene(sample, cfg.phantom_wl, optical, phi, illum.srcpos, illum.dir_unit(),
                    extra_meta=dict(kind="phantom", backend=cfg.phantom_backend,
                                    radius_mm=illum.radius_mm, **pmeta),
                    dataset_dir=cfg.dataset_dir, sim_dir=cfg.sim_dir)
        written.append(dict(sample=sample, wl=cfg.phantom_wl))
    return written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(cfg: ExpansionConfig):
    rng = np.random.default_rng(cfg.seed)
    heads = list_head_scenes(cfg.dataset_dir, cfg.sim_dir)
    print(f"[expand] heads original: {len(heads)}")

    aug = augment_heads(cfg, rng)
    print(f"[expand] heads augmented: {len(aug)} "
          f"({cfg.aug_per_head}/head, proper_only={cfg.aug_proper_only})")

    phan = generate_phantoms(cfg, rng)
    print(f"[expand] phantoms: {len(phan)} "
          f"(backend={cfg.phantom_backend}, shape={cfg.phantom_shape})")

    total = len(heads) + len(aug) + len(phan)
    # suggested split: hold out a few phantoms (MC-quality) + 1 head for val
    n_holdout = max(1, min(2, len(phan)))
    print(f"\n[expand] DATASET COUNTS: heads_original={len(heads)}  "
          f"heads_augmented={len(aug)}  phantoms={len(phan)}  TOTAL={total}")
    print(f"[expand] suggested split: holdout(test)={n_holdout} phantoms, "
          f"val=1 head, train={total - n_holdout - 1} "
          f"(pyramids extracted in the post-FM re-coherence chain)")
    return dict(heads=len(heads), aug=len(aug), phantoms=len(phan), total=total)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--aug-per-head", type=int, default=2)
    p.add_argument("--n-phantoms", type=int, default=16)
    p.add_argument("--phantom-backend", default="mcx", choices=["mcx", "diffusion"])
    p.add_argument("--phantom-size", type=int, default=96)
    p.add_argument("--nphoton", type=float, default=5e6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(ExpansionConfig(aug_per_head=a.aug_per_head, n_phantoms=a.n_phantoms,
                        phantom_backend=a.phantom_backend,
                        phantom_shape=(a.phantom_size,) * 3,
                        phantom_nphoton=a.nphoton, seed=a.seed))
