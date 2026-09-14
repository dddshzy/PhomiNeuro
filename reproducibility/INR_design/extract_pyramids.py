#!/usr/bin/env python3
"""
Extract multi-scale feature pyramids from the FM_tune_t4 domain-adapted encoder.

Updated vs the original (FM_tune_t3) version:
  - Loads FM_tune_t4.MAEDomainAutoencoder (VISTA3D + LoRA + Spatial-STRD)
  - LoRA adapter loaded from checkpoints/best_encoder_lora
  - STRD attention weights loaded from checkpoints/decoder.pt
  - Bottleneck (scale 4) features are STRD-refined, matching training
  - One pyramid .pt cached per .mat file (i.e. per (sample, wavelength))

Each output .pt contains:
  pyramid       : list of 5 CPU tensors (1, C, d, h, w), C in [48,96,192,384,768]
  shapes        : list of torch.Size
  volume_shape  : (X, Y, Z) of the source property volume
"""
import os
import sys
import torch
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from safetensors.torch import load_file
from peft import set_peft_model_state_dict

# Find the encoder model and shared configuration.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from FM_tune_t4.train_lora_autoencoder import MAEDomainAutoencoder
import optical_config as OC
import repro_config as RC

DATASET_DIR = RC.DATASET_DIR
MODEL_CKPT = RC.VISTA3D_CKPT
CKPT_DIR = RC.FM_ADAPTER_DIR
LORA_DIR = (
    CKPT_DIR
    if os.path.isfile(os.path.join(CKPT_DIR, "adapter_model.safetensors"))
    else os.path.join(CKPT_DIR, "best_encoder_lora")
)
DECODER_PT = os.environ.get(
    "FM_STRD_CKPT", os.path.join(CKPT_DIR, "decoder_strd_only.pt")
)
if not os.path.isfile(DECODER_PT):
    DECODER_PT = os.path.join(CKPT_DIR, "decoder.pt")
OUT_DIR = RC.PYRAMID_DIR


def load_property_volume(mat_path):
    """Load (X,Y,Z,4) property volume -> physical (4, X, Y, Z) float32 (channel-first)."""
    mat = sio.loadmat(mat_path)
    vol = mat['vol_prop_eye_aseg']
    return np.transpose(vol, (3, 0, 1, 2)).astype(np.float32)  # (4, X, Y, Z), physical


def load_and_normalize(mat_path):
    """
    Load (X,Y,Z,4) property volume and normalize with FIXED physical ranges
    (optical_config.normalize), NOT per-volume min/max — so the encoder input is
    well-defined at inversion time when mu_a is unknown.  Returns (1, 4, X, Y, Z).
    """
    vol = load_property_volume(mat_path)                       # (4, X, Y, Z) physical
    normalized = OC.normalize(vol, channels=OC.CHANNELS, channel_dim=0)
    return torch.from_numpy(normalized).unsqueeze(0)          # (1, 4, X, Y, Z)


def extract_pyramid(model, vol_norm, device, roi_size=(64, 64, 64),
                    sw_batch_size=4, overlap=0.5):
    """
    Run the adapted encoder over a NORMALIZED (1,4,X,Y,Z) volume via sliding
    window and return the 5-scale feature pyramid (list of CPU tensors).

    This is the SINGLE extraction routine; both the offline cache (main) and the
    live inversion path call it, so a pyramid built from build_optical_volume(...)
    reproduces the cached pyramid exactly (same normalization + same inferer).
    """
    vol_norm = vol_norm.to(device)

    def make_inferer(scale_idx):
        def _infer(x):
            feats = model.encoder(x)
            if scale_idx == 4:
                feats = model._bottleneck_with_strd(feats)
            return feats[scale_idx]
        return _infer

    pyramid = []
    for scale_idx in range(5):
        stitched = sliding_window_inference(
            inputs=vol_norm, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=make_inferer(scale_idx), overlap=overlap, mode='gaussian',
        )
        pyramid.append(stitched.cpu())
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return pyramid


def build_model(device, seed=0):
    # DETERMINISM FIX: VISTA3DEncoder loads model.pt with strict=False, so the
    # 4-channel `conv_init` base layer (shape-mismatched vs the pretrained ckpt) is
    # left RANDOMLY initialized — different every process, and never saved.  That made
    # pyramid extraction non-reproducible (disk pyramids != a fresh extraction), which
    # silently breaks anything needing re-extraction (e.g. mu_a inversion).  Seeding
    # before model construction makes conv_init (and all unloaded inits) reproducible.
    # NOTE: the existing on-disk pyramids were made UNSEEDED -> to fully fix, re-extract
    # all pyramids with this build_model and retrain the INR so train==inversion.
    torch.manual_seed(seed)
    print(f"Building FM_tune_t4 MAEDomainAutoencoder ... (seed={seed})")
    model = MAEDomainAutoencoder(
        encoder_ckpt_path=MODEL_CKPT,
        lora_rank=32, lora_alpha=64,
        patch_size=(8, 8, 8), mask_ratio=0.75,
        strd_num_heads=8, strd_L_s=1,
    ).to(device)

    # 1. Load fine-tuned LoRA adapter weights into the PEFT-wrapped encoder
    adapter_sd = load_file(os.path.join(LORA_DIR, "adapter_model.safetensors"))
    res = set_peft_model_state_dict(model.encoder, adapter_sd)
    print(f"  LoRA adapter loaded from {LORA_DIR} ({res})")

    # 2. Load STRD attention weights
    dec = torch.load(DECODER_PT, map_location='cpu')
    model.strd_attention.load_state_dict(dec['strd_attention_state'], strict=True)
    print(f"  STRD attention loaded (trained {dec.get('epoch')} epochs, loss {dec.get('loss'):.4f})")

    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    model = build_model(device)

    file_list = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.mat')])
    # shard across GPUs: launch N instances with FM_NSHARD=N, FM_SHARD=0..N-1
    nshard = int(os.environ.get("FM_NSHARD", 1))
    shard  = int(os.environ.get("FM_SHARD", 0))
    skip_existing = os.environ.get("FM_SKIP_EXISTING", "1") == "1"
    file_list = file_list[shard::nshard]
    print(f"[shard {shard}/{nshard}] {len(file_list)} property volumes\n")

    with torch.no_grad():
        for fname in tqdm(file_list, desc=f"shard{shard}"):
            out_path = os.path.join(OUT_DIR, fname.replace('.mat', '_pyramid.pt'))
            if skip_existing and os.path.isfile(out_path):
                continue
            vol_tensor = load_and_normalize(os.path.join(DATASET_DIR, fname))
            pyramid = extract_pyramid(model, vol_tensor, device)
            torch.save({
                "pyramid": pyramid,
                "shapes": [t.shape for t in pyramid],
                "volume_shape": tuple(vol_tensor.shape[-3:]),
            }, out_path)

    print(f"\n[shard {shard}/{nshard}] done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
