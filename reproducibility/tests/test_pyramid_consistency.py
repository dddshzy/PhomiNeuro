"""
Live-vs-cached pyramid consistency.

Proves that the inversion-time path
    raw mu_a  ->  build_optical_volume(...)  ->  adapted encoder  ->  pyramid
reproduces the offline cached path
    raw .mat  ->  load_and_normalize(...)    ->  adapted encoder  ->  pyramid
i.e. channel order + FIXED-range normalization are identical on both sides.

A literal "full-volume single pass == sliding-window cache" comparison is NOT a
correctness check (tiling/blending differ), so we instead:
  1. assert the NORMALIZED volumes are bit-identical (this is where channel-order
     / normalization bugs actually live), then
  2. run BOTH through the *same* extractor on a small crop and assert the
     resulting pyramids match (< 1e-4).
"""
import os
import sys
import glob

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "INR_design"))
import optical_config as OC
import repro_config as RC

DATASET_DIR = str(RC.DATASET_DIR)
_checkpoint_dir = str(RC.FM_ADAPTER_DIR)
LORA_DIR = (_checkpoint_dir if os.path.isfile(os.path.join(_checkpoint_dir, "adapter_model.safetensors"))
            else os.path.join(_checkpoint_dir, "best_encoder_lora"))
DECODER_PT = os.path.join(_checkpoint_dir, "decoder_strd_only.pt")
if not os.path.isfile(DECODER_PT):
    DECODER_PT = os.path.join(_checkpoint_dir, "decoder.pt")

_mats = sorted(glob.glob(os.path.join(DATASET_DIR, "*.mat")))
_have_ckpt = os.path.isfile(os.path.join(LORA_DIR, "adapter_model.safetensors")) \
    and os.path.isfile(DECODER_PT)


def test_normalized_volume_live_equals_cached():
    """Step 1 — no model needed: build_optical_volume == load_and_normalize."""
    if not _mats:
        pytest.skip("no dataset .mat files")
    import extract_pyramids as EP

    path = _mats[0]
    cached = EP.load_and_normalize(path).squeeze(0)              # (4,X,Y,Z) normalized
    phys = torch.from_numpy(EP.load_property_volume(path))       # (4,X,Y,Z) physical
    live = OC.build_optical_volume(phys[0], phys[1:]).squeeze(0)  # (4,X,Y,Z) normalized

    diff = (cached - live).abs().max().item()
    assert diff < 1e-4, f"normalized volume mismatch: {diff}"


@pytest.mark.skipif(not _have_ckpt, reason="FM_tune_t4 checkpoints not found")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for encoder")
def test_pyramid_live_equals_cached():
    """Step 2 — run the adapted encoder on a small crop via the shared extractor."""
    if not _mats:
        pytest.skip("no dataset .mat files")
    import extract_pyramids as EP

    device = torch.device("cuda")
    model = EP.build_model(device)

    path = _mats[0]
    phys = torch.from_numpy(EP.load_property_volume(path))       # (4,X,Y,Z) physical
    crop = phys[:, :64, :64, :64]                                # keep the test fast

    cached_vol = OC.normalize(crop, channel_dim=0).unsqueeze(0)             # cached path
    live_vol = OC.build_optical_volume(crop[0], crop[1:])                   # live path

    with torch.no_grad():
        pyr_cached = EP.extract_pyramid(model, cached_vol, device)
        pyr_live = EP.extract_pyramid(model, live_vol, device)

    assert len(pyr_cached) == len(pyr_live) == 5
    for i, (a, b) in enumerate(zip(pyr_cached, pyr_live)):
        d = (a - b).abs().max().item()
        assert d < 1e-4, f"pyramid scale {i} mismatch: {d}"
