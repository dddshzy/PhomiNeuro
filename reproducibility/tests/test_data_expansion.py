"""
Acceptance tests for physics-consistent data expansion.

- test_equivariance      : flip/rot90 a phantom + source, re-solve (diffusion),
                           compare to the transformed Phi (max rel err < 1e-3).
- test_solver_agreement  : diffusion vs MCX on one phantom, RMSE in log10 over the
                           diffusive interior (reported; near-source/deep excluded).
- test_scene_discoverable: a generated scene loads through discover_scenes /
                           SceneStore and feeds the INR without shape errors.
"""
import os
import sys
import json
import importlib.util

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "INR_design"))

from data_expansion import augment as AUG
from data_expansion.phantom import generate_phantom, PhantomConfig
from data_expansion import solvers as S
from data_expansion.solvers import Illumination, solve_diffusion, DiffusionConfig

_have_mcx = importlib.util.find_spec("pmcxcl") is not None or bool(
    os.environ.get("PMCXCL_LIBRARY")
)


# ---------------------------------------------------------------------------
def test_transform_consistency():
    """Volume/point/direction transforms agree (no interpolation, exact)."""
    rng = np.random.default_rng(1)
    vol = rng.random((6, 7, 8))
    for t in AUG.all_transforms():
        volT = t.apply_volume(vol)
        assert volT.shape == t.new_shape(vol.shape)
        # a known voxel maps consistently
        p = np.array([1, 2, 3])
        pT = t.apply_point(p, vol.shape).astype(int)
        assert np.isclose(vol[tuple(p)], volT[tuple(pT)])


def test_equivariance():
    """Re-solving a rigidly transformed phantom == transforming the solution."""
    rng = np.random.default_rng(3)
    opt, illum, _ = generate_phantom(rng, PhantomConfig(shape=(48, 48, 48)))
    cfg = DiffusionConfig(cg_tol=1e-10)

    # base solve with a pinned source voxel (isolates PDE-solver equivariance)
    mu_sp = (opt[..., 1] * (1 - opt[..., 2]))
    s0 = S._find_source_voxel(mu_sp > 0, mu_sp, illum, cfg.h_mm, cfg.src_depth_mfp)
    phi0 = solve_diffusion(opt, illum, cfg=cfg, src_voxel=s0)

    worst = 0.0
    for t in AUG.sample_nonidentity(rng, 4):
        optT = t.apply_volume(opt)
        s0T = t.apply_point(s0, opt.shape[:3]).astype(int)
        phiT_ref = t.apply_volume(phi0)                      # transform the solution
        phiT = solve_diffusion(optT, illum, cfg=cfg, src_voxel=tuple(s0T))
        m = phiT_ref > phiT_ref.max() * 1e-6
        rel = np.abs(phiT[m] - phiT_ref[m]) / (phiT_ref[m] + 1e-30)
        worst = max(worst, float(rel.max()))
    assert worst < 1e-3, f"equivariance violated: max rel err {worst:.2e}"


@pytest.mark.skipif(not _have_mcx, reason="pmcxcl .so not found")
def test_solver_agreement():
    """Diffusion vs MCX agree in the diffusive interior (RMSE in log10)."""
    rng = np.random.default_rng(0)
    opt, illum, _ = generate_phantom(rng, PhantomConfig(shape=(80, 80, 80)))
    phi_d, info = solve_diffusion(opt, illum, return_info=True)
    phi_m = S.solve_mcx(opt, illum, nphoton=2e6)

    mu_sp = opt[..., 1] * (1 - opt[..., 2])
    mask = S.diffusive_interior_mask(phi_d, phi_m, info["src_voxel"], mu_sp)
    rmse, offset, nvox = S.log10_agreement(phi_d, phi_m, mask)
    print(f"\n[solver_agreement] RMSE(log10 Phi)={rmse:.3f}  scale_offset={offset:.2f}"
          f"  nvox={nvox}  (deep/near-source/boundary excluded)")
    assert rmse < 0.4, f"diffusion-MCX disagree: RMSE(log10)={rmse:.3f}"


def _write_stub_pyramid(path, vol_shape, channels=(48, 96, 192, 384, 768)):
    """Minimal 5-level feature pyramid (real extraction happens post-FM-retrain)."""
    X, Y, Z = vol_shape
    pyr = []
    for li, c in enumerate(channels):
        f = 2 ** li
        d = max(1, X // f), max(1, Y // f), max(1, Z // f)
        pyr.append(torch.zeros(1, c, *d))
    torch.save({"pyramid": pyr, "shapes": [p.shape for p in pyr],
                "volume_shape": tuple(vol_shape)}, path)


def test_scene_discoverable(tmp_path, monkeypatch):
    """A generated scene loads via discover_scenes/SceneStore and feeds the INR."""
    import inr_dataset as D
    from data_expansion import generate as G
    from train_inr import OpticalFluenceINR

    ds = tmp_path / "dataset"; pyr = tmp_path / "pyr"; sim = tmp_path / "sim"
    ds.mkdir(); pyr.mkdir(); sim.mkdir()
    monkeypatch.setattr(D, "DATASET_DIR", str(ds))
    monkeypatch.setattr(D, "PYRAMID_DIR", str(pyr))
    monkeypatch.setattr(D, "SIM_DIR", str(sim))

    # generate + write one small phantom scene (diffusion backend; no GPU needed)
    rng = np.random.default_rng(7)
    cfg = G.ExpansionConfig(dataset_dir=str(ds), sim_dir=str(sim),
                            phantom_backend="diffusion", phantom_shape=(40, 40, 40))
    opt, illum, pmeta = generate_phantom(rng, PhantomConfig(shape=(40, 40, 40)))
    phi = solve_diffusion(opt, illum)
    G.write_scene("phantom0000", cfg.phantom_wl, opt, phi, illum.srcpos,
                  illum.dir_unit(), extra_meta=dict(kind="phantom"),
                  dataset_dir=str(ds), sim_dir=str(sim))
    _write_stub_pyramid(str(pyr / f"phantom0000_copmri_withHermiteF{cfg.phantom_wl}_pyramid.pt"),
                        (40, 40, 40))

    scenes = D.discover_scenes()
    assert len(scenes) == 1, f"expected 1 scene, got {len(scenes)}"

    store = D.SceneStore(torch.device("cpu"))
    sd = store.get(scenes[0])
    xyz, optical, gt_log = D.sample_points(sd, 256)
    assert optical.shape == (256, 4) and gt_log.shape == (256, 1)
    # optical (normalized) must be finite and within [0,1]-ish for in-range physical vals
    assert torch.isfinite(optical).all()

    model = OpticalFluenceINR()
    pred = model(xyz, sd["vol_shape"], optical, sd["light"], sd["pyramid"])
    assert pred.shape == (256, 1) and torch.isfinite(pred).all()
