"""FM encoder: build it, and extract the 5-scale feature pyramid a head model needs.

THE SEED IS PART OF THE WEIGHTS. VISTA3D is a 1-channel (CT/MR) model and our input has 4 channels,
so the input projection `conv_init` cannot be loaded directly. `VISTA3DEncoder.__init__` builds it:

    w_expanded = w.repeat(1, 4, 1, 1, 1) / 4.0                  # channel inflation
    w_expanded[:, i] += randn_like(...) * w.std() * 0.1         # symmetry breaking

The pretrained kernel IS transferred -- measured, per channel: correlation 0.93 with the pretrained
kernel and a least-squares scale of 0.2500. The added noise is deliberate and necessary: without it
all four channels would be identical and the layer could see only the mean of (mu_a, mu_s, g, n),
never the difference between them. It contributes 14% of the weight variance (residual std 0.0126
against 0.0314 for the inflated kernel), and it is drawn from the global RNG.

That noise is the one irreproducible part. It is not in the LoRA adapter (`modules_to_save` is
empty), the base layer is frozen (`requires_grad=False`) so it is not in any optimiser state either,
and it cannot be recovered from the VISTA3D checkpoint. Only `torch.manual_seed(0)` immediately
before construction reproduces it.

This is measured, not assumed. Extracting a head's level-4 pyramid with `manual_seed(0)` reproduces the
published pyramid EXACTLY (max|diff| = 0.0, and 0.0 again from a second process); two different seeds
give input layers correlated at 0.861 -- exactly the 86% the deterministic part accounts for -- whose
residuals are uncorrelated (-0.022). Change the seed and every number here changes with it.

One consequence worth registering. The MAE adaptation ran in a process with a different draw than the
extraction did, so the LoRA weights were adapted on top of a slightly different input projection than
they are deployed with. The difference is only that 10%-of-std noise term, but it is not negligible
downstream: substituting the training draw changes the level-4 bottleneck features by 73% relative.
Surrogate training and surrogate inference both use the extraction draw, so everything published is
self-consistent and reproduces exactly -- this is a note about the FM stage, not a defect in the
release.
"""
import os
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from safetensors.torch import load_file
from peft import set_peft_model_state_dict

from . import optical_config as OC
from .fm_autoencoder import MAEDomainAutoencoder

PYRAMID_CHANNELS = (48, 96, 192, 384, 768)     # 1488 total
EXTRACTION_SEED = 0


def build_fm_encoder(vista3d_ckpt, weights_dir, device, seed=EXTRACTION_SEED, verbose=True):
    """VISTA3D backbone + our LoRA adapter + the STRD bottleneck block."""
    for p in (vista3d_ckpt, os.path.join(weights_dir, "adapter_model.safetensors"),
              os.path.join(weights_dir, "decoder_strd_only.pt")):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"{p} not found. The VISTA3D backbone is a third-party 872 MB file and is NOT in "
                f"this repository; see the README for the one-line download.")
    torch.manual_seed(seed)                    # see the module docstring -- do not remove
    model = MAEDomainAutoencoder(encoder_ckpt_path=vista3d_ckpt, lora_rank=32, lora_alpha=64,
                                 patch_size=(8, 8, 8), mask_ratio=0.75,
                                 strd_num_heads=8, strd_L_s=1).to(device)
    adapter = load_file(os.path.join(weights_dir, "adapter_model.safetensors"))
    res = set_peft_model_state_dict(model.encoder, adapter)
    dec = torch.load(os.path.join(weights_dir, "decoder_strd_only.pt"), map_location="cpu")
    model.strd_attention.load_state_dict(dec["strd_attention_state"])
    # `missing_keys` here is EXPECTED and long: a LoRA adapter holds only lora_A/lora_B, so every
    # frozen VISTA3D weight is reported missing. Printing the list makes a normal load look like a
    # failure. What actually matters is that nothing is UNEXPECTED -- an unexpected key would mean
    # the adapter does not match this module tree -- so that is what is checked and shown.
    if res.unexpected_keys:
        raise RuntimeError(f"adapter does not match the encoder: {len(res.unexpected_keys)} "
                           f"unexpected keys, first few {res.unexpected_keys[:3]}")
    if verbose:
        print(f"[encoder] LoRA {len(adapter)} tensors loaded, 0 unexpected "
              f"({len(res.missing_keys)} frozen base weights come from the VISTA3D checkpoint, as "
              f"expected); STRD {len(dec['strd_attention_state'])} tensors; seed={seed}")
    return model.eval()


def normalise_volume(prop_phys):
    """(4,X,Y,Z) physical optical properties -> (1,4,X,Y,Z) encoder input.

    Normalisation uses FIXED physical ranges, never per-volume statistics: at inverse-design time
    mu_a is the unknown, so a per-volume normaliser would make the encoder input depend on the
    answer.
    """
    return torch.from_numpy(OC.normalize(prop_phys, channels=OC.CHANNELS,
                                         channel_dim=0)).unsqueeze(0)


SW_BATCH_SIZE = 4      # DO NOT CHANGE -- see the note in extract_pyramid


@torch.no_grad()
def extract_pyramid(model, vol_norm, device, roi_size=(64, 64, 64), sw_batch_size=SW_BATCH_SIZE,
                    overlap=0.5, levels=range(5), out_device="cpu", verbose=True):
    """Sliding-window the encoder over a normalised volume -> list of 5 CPU tensors.

    Level 0 alone is 48 x X x Y x Z float32 (3.3 GB for a 224x256x300 head); all five come to about
    4.1 GB. That is why pyramids are not shipped and are extracted on demand instead.

    `out_device="cpu"` accumulates the stitched output in host memory rather than on the card.
    Measured on a 224x256x300 head: peak card memory 3.11 GB against 6.05 GB, for 14.0 s against
    10.5 s. The output is BIT-IDENTICAL either way (max|diff| = 0.0 against the published pyramid),
    so this is a free halving of the memory requirement and is the default. Pass out_device=device
    to accumulate on the card instead.

    `sw_batch_size` is a different matter and must be left alone. Changing it changes how windows are
    batched and therefore the order in which their contributions are accumulated: at sw_batch_size=1
    the level-4 output moves by 3.6e-3 relative against the published pyramid, while at 4 it is
    bit-identical. It is a reproducibility parameter wearing a performance parameter's clothes.
    """
    if sw_batch_size != SW_BATCH_SIZE:
        raise ValueError(
            f"sw_batch_size must stay at {SW_BATCH_SIZE}: it changes the accumulation order and "
            f"hence the extracted features (measured 3.6e-3 relative at 1). Lower the memory "
            f"requirement with out_device='cpu' instead, which is bit-identical.")
    vol_norm = vol_norm.to(device)
    out_device = device if out_device is None else torch.device(out_device)

    def make_inferer(scale_idx):
        def _infer(x):
            feats = model.encoder(x)
            if scale_idx == 4:
                feats = model._bottleneck_with_strd(feats)
            return feats[scale_idx]
        return _infer

    pyramid = []
    for i in levels:
        out = sliding_window_inference(inputs=vol_norm, roi_size=roi_size,
                                       sw_batch_size=sw_batch_size, predictor=make_inferer(i),
                                       overlap=overlap, mode="gaussian",
                                       device=out_device, sw_device=device)
        pyramid.append(out.cpu())
        if verbose:
            print(f"[encoder] level {i}: {tuple(out.shape)}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pyramid


def pyramid_path(cache_dir, head):
    return os.path.join(cache_dir, f"{head}_v11mni_F810_pyramid.pt")


def get_pyramid(head, prop_phys, cache_dir, vista3d_ckpt, weights_dir, device, verbose=True):
    """Cached extraction. Extraction is deterministic, so the cache is a pure speed-up."""
    p = pyramid_path(cache_dir, head)
    if os.path.isfile(p):
        if verbose:
            print(f"[encoder] cache hit: {p}")
        return torch.load(p, map_location="cpu")["pyramid"]
    os.makedirs(cache_dir, exist_ok=True)
    model = build_fm_encoder(vista3d_ckpt, weights_dir, device, verbose=verbose)
    pyr = extract_pyramid(model, normalise_volume(prop_phys), device, verbose=verbose)
    torch.save({"pyramid": pyr, "seed": EXTRACTION_SEED}, p)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pyr
