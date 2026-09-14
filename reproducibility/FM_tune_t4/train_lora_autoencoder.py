#!/usr/bin/env python3
"""LoRA adaptation of a VISTA3D/SegResNetDS2 encoder to optical volumes.

Training uses 64-cubed random crops, MAE-style patch masking, spatial STRD
attention at the bottleneck, a skip-connected decoder, and a composite
L1/SSIM/gradient/masked-patch loss. Optical channels use the fixed ranges in
``optical_config``.
"""

import os
import sys
import time
import json
import math
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from monai.losses import SSIMLoss

# Shared optical-channel config / FIXED-range normalization (single source of truth).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optical_config as OC
import repro_config as RC


# =============================================================================
# 1. Dataset
# =============================================================================
class OpticalPropertyDataset(Dataset):
    """Return random MAE crops from normalized ``.npy`` caches or source ``.mat`` files."""
    def __init__(self, data_dir, crop_size=(64, 64, 64), npy_dir=None):
        self.data_dir = data_dir
        self.npy_dir = npy_dir or os.environ.get("FM_NPY_DIR")
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
        if self.npy_dir:
            keep = [f for f in self.file_list
                    if os.path.isfile(os.path.join(self.npy_dir, self._npy_name(f)))]
            if len(keep) != len(self.file_list):
                raise FileNotFoundError(
                    f"FM_NPY_DIR={self.npy_dir} is missing {len(self.file_list)-len(keep)} of "
                    f"{len(self.file_list)} volumes -- run prep_encoder_npy.py first "
                    f"(a partially-populated cache would silently train on a subset)")
        rep = int(os.environ.get("FM_REPEAT", 1))      # oversample (more random crops/epoch)
        self.heads = [f.split("_v11mni")[0] for f in self.file_list]   # for weighted sampling
        self.file_list = self.file_list * rep
        self.crop_size = crop_size
        # Optional on-the-fly random rotation augmentation.
        self.rotaug = os.environ.get("FM_ROTAUG") == "1"
        # Preload normalized volumes for shared-memory workers, or use a bounded cache.
        self._cache = {}
        self._cache_max = int(os.environ.get("FM_VOL_CACHE", 16))
        self._min_tissue = float(os.environ.get("FM_MIN_TISSUE", 0.02))   # >=2% scattering voxels
        self._max_retry = int(os.environ.get("FM_CROP_RETRY", 12))
        self._preload = os.environ.get("FM_PRELOAD", "1") == "1" and self.npy_dir
        if self._preload:
            import time as _t
            _t0 = _t.time()
            uniq = list(dict.fromkeys(self.file_list))
            for f in uniq:
                self._cache[self._npy_name(f)] = np.load(
                    os.path.join(self.npy_dir, self._npy_name(f)))
            gb = sum(v.nbytes for v in self._cache.values()) / 1e9
            print(f"[Dataset] PRELOADED {len(uniq)} volumes ({gb:.0f} GB resident, shared "
                  f"copy-on-write with the workers) in {_t.time()-_t0:.0f}s -- zero I/O per epoch",
                  flush=True)
        print(f"[Dataset] Found {len(self.file_list)//rep} volumes x{rep} = "
              f"{len(self.file_list)} samples, crop_size={crop_size}, rotaug={self.rotaug}, "
              f"src={('npy(preload)' if self._preload else 'npy(LRU%d)' % self._cache_max) if self.npy_dir else 'mat'}")

    @staticmethod
    def _npy_name(fname):
        return fname.split("_v11mni")[0] + ".npy"

    def __len__(self):
        return len(self.file_list)

    def _load_data(self, fname):
        mat = sio.loadmat(os.path.join(self.data_dir, fname))
        vol = mat['vol_prop_eye_aseg']               # (X, Y, Z, C)
        vol = np.transpose(vol, (3, 0, 1, 2))       # (C, X, Y, Z)
        return vol.astype(np.float32)

    def _load_npy(self, fname):
        """Load a normalized ``(C, X, Y, Z)`` array through the per-worker cache."""
        key = self._npy_name(fname)
        v = self._cache.get(key)
        if v is None:                                             # LRU mode only; PRELOAD never misses
            v = np.load(os.path.join(self.npy_dir, key))          # sequential, page-cached
            if len(self._cache) >= self._cache_max:
                self._cache.pop(next(iter(self._cache)))          # FIFO is enough here
            self._cache[key] = v
        return v

    def _crop(self, vol, size):
        _, D, H, W = vol.shape
        cD, cH, cW = size
        pad_d = max(0, cD - D); pad_h = max(0, cH - H); pad_w = max(0, cW - W)
        if pad_d or pad_h or pad_w:
            vol = np.pad(vol, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            _, D, H, W = vol.shape
        d0 = np.random.randint(0, D - cD + 1)
        h0 = np.random.randint(0, H - cH + 1)
        w0 = np.random.randint(0, W - cW + 1)
        return vol[:, d0:d0+cD, h0:h0+cH, w0:w0+cW]

    def _random_crop(self, vol):
        return self._crop(vol, self.crop_size)

    def _rot_crop(self, vol):
        """Crop a margin window, rotate by a random axis+arbitrary angle, center-crop to
        crop_size (so rotated corners never enter the patch), then random flips."""
        import scipy.ndimage as ndi
        cD, cH, cW = self.crop_size
        m = 16
        sub = self._crop(vol, (cD + 2*m, cH + 2*m, cW + 2*m))
        ang = float(np.random.uniform(-180.0, 180.0))
        axes = [(1, 2), (1, 3), (2, 3)][np.random.randint(3)]
        sub = ndi.rotate(sub, ang, axes=axes, reshape=False, order=1, mode="nearest")
        _, D, H, W = sub.shape
        d0 = (D - cD)//2; h0 = (H - cH)//2; w0 = (W - cW)//2
        out = sub[:, d0:d0+cD, h0:h0+cH, w0:w0+cW]
        for ax in (1, 2, 3):
            if np.random.rand() < 0.5:
                out = np.flip(out, ax)
        return np.ascontiguousarray(out)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        if self.npy_dir:
            normalized = self._load_npy(fname)               # memmap, already normalised
        else:
            data = self._load_data(fname)                    # (C, X, Y, Z) physical
            # FIXED physical-range normalization (NOT per-volume min/max) so the encoder
            # input is identical at training and at inversion time, when mu_a is unknown.
            normalized = OC.normalize(data, channels=OC.CHANNELS, channel_dim=0)
        # REJECT information-free crops. A 64^3 window landing entirely in air is CONSTANT, so the
        # normalisation layers see zero variance; under fp16 the 1/sqrt(var + eps) then overflowed
        # and the loss went NaN from epoch 65 (weights stayed clean -- GradScaler skipped those
        # steps -- and all 225 volumes were verified NaN-free, which is how the crop was isolated).
        # The sqrt-balanced sampler is what SURFACED it: it lifts scb from 4.9% to 15.8% of each
        # epoch, and scb heads are the smallest (18.7% tissue vs sh's 25.6%), so 12.9% of their
        # random crops hold <1% tissue against sh's 1.1%. Such a crop carries no anatomy and is a
        # wasted sample whatever the numerics, so resample instead of feeding it.
        for _ in range(self._max_retry):
            patch = self._rot_crop(normalized) if self.rotaug else self._random_crop(normalized)
            if float((np.asarray(patch[1]) > 0).mean()) >= self._min_tissue:   # ch 1 = mu_s
                break
        return torch.from_numpy(np.ascontiguousarray(patch))          # (4, D, H, W)


# =============================================================================
# 2. 3D Patch Masking (MAE-style)
# =============================================================================
class PatchMasker3D(nn.Module):
    """
    Patchify a 3D volume into non-overlapping patches, randomly mask a fraction,
    and return the masked volume (masked patches replaced by learnable mask_token)
    plus the binary mask (1=masked, 0=visible) for loss computation.

    patch_size: (pD, pH, pW) — must evenly divide crop_size
    mask_ratio:  fraction of patches to mask (default 0.75)
    """
    def __init__(self, in_channels=4, patch_size=(8, 8, 8), mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        # Learnable mask token — same shape as one patch's flattened voxels
        patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.mask_token = nn.Parameter(torch.zeros(patch_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, x):
        """
        x: (B, C, D, H, W)
        returns patches: (B, N, patch_dim)
                coords:  (B, N, 3)   — patch grid indices (d_i, h_i, w_i)
        """
        B, C, D, H, W = x.shape
        pD, pH, pW = self.patch_size
        assert D % pD == 0 and H % pH == 0 and W % pW == 0, \
            f"Volume ({D},{H},{W}) must be divisible by patch_size ({pD},{pH},{pW})"

        nD, nH, nW = D // pD, H // pH, W // pW
        # Reshape: (B, C, nD, pD, nH, pH, nW, pW)
        x = x.reshape(B, C, nD, pD, nH, pH, nW, pW)
        # Permute → (B, nD, nH, nW, C, pD, pH, pW) → (B, N, patch_dim)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        N = nD * nH * nW
        patch_dim = C * pD * pH * pW
        patches = x.reshape(B, N, patch_dim)

        # Patch grid coordinates
        d_idx = torch.arange(nD, device=x.device)
        h_idx = torch.arange(nH, device=x.device)
        w_idx = torch.arange(nW, device=x.device)
        dd, hh, ww = torch.meshgrid(d_idx, h_idx, w_idx, indexing='ij')
        coords = torch.stack([dd, hh, ww], dim=-1).reshape(1, N, 3).expand(B, -1, -1)

        return patches, coords, (nD, nH, nW)

    def unpatchify(self, patches, grid_shape, C):
        """
        patches: (B, N, patch_dim) → (B, C, D, H, W)
        """
        B, N, patch_dim = patches.shape
        nD, nH, nW = grid_shape
        pD, pH, pW = self.patch_size
        patches = patches.reshape(B, nD, nH, nW, C, pD, pH, pW)
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return patches.reshape(B, C, nD * pD, nH * pH, nW * pW)

    def forward(self, x):
        """
        Returns:
          x_masked:   (B, C, D, H, W)  — original with masked patches replaced by mask_token
          patch_mask: (B, N)  bool     — True where patch is masked
          orig_patches:(B, N, patch_dim) — original patches (for loss computation)
          grid_shape: tuple (nD, nH, nW)
        """
        B, C, D, H, W = x.shape
        orig_patches, coords, grid_shape = self.patchify(x)  # (B, N, patch_dim)
        N = orig_patches.shape[1]
        num_mask = int(N * self.mask_ratio)

        # Random masking — independent per sample in batch
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)              # ascending
        mask_ids = ids_shuffle[:, :num_mask]                   # top num_mask = masked

        patch_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        patch_mask.scatter_(1, mask_ids, True)                 # True = masked

        # Replace masked patches with mask_token
        masked_patches = orig_patches.clone()
        masked_patches[patch_mask] = self.mask_token.to(dtype=orig_patches.dtype)

        # Reconstruct volume with masked patches zeroed / replaced
        x_masked = self.unpatchify(masked_patches, grid_shape, C)

        return x_masked, patch_mask, orig_patches, grid_shape


# =============================================================================
# 3. Spatial STRD Attention (3D-only version, no time axis)
# =============================================================================
class SpatialSTRDAttention(nn.Module):
    """
    Spatial Temporal Redundancy Dropout → Spatial-only variant for 3D volumes.

    Applied as a lightweight Multi-Head Self-Attention on the flattened bottleneck
    feature map.  After computing softmax attention weights A, a per-connection
    dropout probability W[i,j] is derived from spatial proximity:

        f_spat(i)  = max_{j in Ω_s(i)} A[i,j]          (how much i attends to spatial neighbours)
        W[i,j]     = clamp( f_spat * A[i,j] / (sum_{k in Ω_s(i)} A[i,k] + ε), 0, 1 )
                     for j in Ω_s(i)

    Connections that are both highly attended AND spatially proximal get dropped
    with high probability, pushing the model toward long-range dependencies.

    dim:       channel dimension of the bottleneck feature (e.g. 768)
    num_heads: number of attention heads
    L_s:       spatial window half-length in the bottleneck grid (default 1 = 3^3 cube)
    """
    def __init__(self, dim=768, num_heads=8, L_s=1, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.L_s       = L_s

        self.qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.proj  = nn.Linear(dim, dim)
        self.norm  = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(dropout)

    @staticmethod
    def _build_spatial_neighbor_mask(coords, L_s, device):
        """
        coords: (N, 3) — integer grid coordinates (d, h, w) for each token
        Returns is_neighbor: (N, N) bool — True if Chebyshev distance <= L_s (excluding self)
        """
        N = coords.shape[0]
        # Broadcast diff: (N, N, 3)
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)       # (N, N, 3)
        cheby = diff.abs().max(dim=-1).values                  # (N, N)
        is_neighbor = (cheby <= L_s) & (cheby > 0)            # exclude self (dist==0)
        return is_neighbor.to(device)

    def forward(self, x, grid_coords):
        """
        x:           (B, N, dim)   — flattened bottleneck tokens
        grid_coords: (N, 3)        — integer spatial coordinates per token
        Returns:     (B, N, dim)
        """
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, N, head_dim)
        Q, K, V = qkv.unbind(0)             # each (B, H, N, head_dim)

        # Standard scaled dot-product attention scores
        scores = (Q @ K.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        A = scores.softmax(dim=-1)                         # (B, H, N, N)

        # ---- Spatial STRD ----
        # Build neighbor mask once per forward pass (same for all heads/batches)
        is_nbr = self._build_spatial_neighbor_mask(
            grid_coords, self.L_s, device=x.device
        )  # (N, N)

        # f_spat(i) = max attention to spatial neighbours, per head per sample
        # Mask out non-neighbours and self, then take max
        nbr_mask = is_nbr.unsqueeze(0).unsqueeze(0).float()   # (1, 1, N, N)
        A_nbr = A * nbr_mask                                  # zero out non-neighbours

        # f_spat: (B, H, N, 1)
        f_spat = A_nbr.max(dim=-1, keepdim=True).values       # max over keys

        # STRD dropout probability W[i,j] for spatial neighbours
        sum_nbr = A_nbr.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, H, N, 1)
        W = (f_spat * A) / sum_nbr                            # (B, H, N, N)
        W = W * nbr_mask                                      # only neighbours
        W = W.clamp(0.0, 1.0)

        # Bernoulli drop: keep with prob (1 - W)
        if self.training:
            keep = torch.bernoulli(1.0 - W)                  # (B, H, N, N)
            A_drop = A * keep
            # Re-normalize rows (add small eps to avoid div-by-zero for fully-dropped rows)
            A_drop = A_drop / (A_drop.sum(dim=-1, keepdim=True).clamp(min=1e-6))
        else:
            # At inference, use expectation: scale down by (1-W) instead of sampling
            A_drop = A * (1.0 - W)
            A_drop = A_drop / (A_drop.sum(dim=-1, keepdim=True).clamp(min=1e-6))

        A_drop = self.drop(A_drop)

        # Weighted sum over values
        out = A_drop @ V                                       # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)            # (B, N, D)
        out = self.proj(out)
        return residual + out                                  # residual connection


# =============================================================================
# 4. VISTA3D Encoder Wrapper
# =============================================================================
class VISTA3DEncoder(nn.Module):
    def __init__(self, checkpoint_path, freeze=True):
        super().__init__()
        from monai.networks.nets import SegResNetDS2

        self.encoder = SegResNetDS2(
            in_channels=4,
            blocks_down=(1, 2, 2, 4, 4),
            norm='instance',
            out_channels=48,
            init_filters=48,
            dsdepth=1,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=device)
        filtered = {k.replace('image_encoder.', '', 1): v
                    for k, v in ckpt.items() if k.startswith('image_encoder.')}

        actual_key = [k for k in filtered if 'conv_init' in k.lower() or 'convInit' in k][0]
        w = filtered[actual_key]
        if w.shape[1] == 1:
            w_expanded = w.repeat(1, 4, 1, 1, 1) / 4.0
            for i in range(4):
                w_expanded[:, i] += torch.randn_like(w_expanded[:, i]) * w.std() * 0.1
            filtered[actual_key] = w_expanded

        self.encoder.load_state_dict(filtered, strict=False)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        total = sum(p.numel() for p in self.encoder.parameters())
        print(f"[VISTA3DEncoder] Loaded from {checkpoint_path}, params: {total:,}")

    def forward(self, x):
        """Returns list of 5 feature maps: scales 0-4, bottleneck at index 4."""
        return self.encoder.encoder(x)


# =============================================================================
# 5. Lightweight 3D Decoder (U-Net with skip connections)
# =============================================================================
class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.gelu  = nn.GELU()

    def forward(self, x):
        r = x
        out = self.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.gelu(r + out)


class UNetDecoder3D(nn.Module):
    def __init__(self, out_channels=4):
        super().__init__()
        self.up3   = nn.ConvTranspose3d(768, 384, 2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv3d(768, 384, 3, padding=1), ResBlock3D(384))
        self.up2   = nn.ConvTranspose3d(384, 192, 2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv3d(384, 192, 3, padding=1), ResBlock3D(192))
        self.up1   = nn.ConvTranspose3d(192, 96, 2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv3d(192, 96, 3, padding=1), ResBlock3D(96))
        self.up0   = nn.ConvTranspose3d(96, 48, 2, stride=2)
        self.conv0 = nn.Sequential(nn.Conv3d(96, 48, 3, padding=1), ResBlock3D(48))
        self.final = nn.Sequential(
            nn.Conv3d(48, 32, 3, padding=1), nn.InstanceNorm3d(32), nn.GELU(),
            nn.Conv3d(32, out_channels, 1),
        )

    def forward(self, features):
        x = features[4]
        x = self.up3(x);  x = torch.cat([x, features[3]], dim=1); x = self.conv3(x)
        x = self.up2(x);  x = torch.cat([x, features[2]], dim=1); x = self.conv2(x)
        x = self.up1(x);  x = torch.cat([x, features[1]], dim=1); x = self.conv1(x)
        x = self.up0(x);  x = torch.cat([x, features[0]], dim=1); x = self.conv0(x)
        return torch.sigmoid(self.final(x))


# =============================================================================
# 6. Full Model: MAE + STRD Domain Adaptation Autoencoder
# =============================================================================
class MAEDomainAutoencoder(nn.Module):
    """
    Pipeline:
      1. PatchMasker3D  — randomly mask 75% of input patches
      2. VISTA3DEncoder — extract multi-scale features from *masked* input
      3. SpatialSTRDAttention — refine bottleneck with redundancy-aware attention
      4. UNetDecoder3D  — reconstruct full volume with skip connections
      5. Loss: full reconstruction (unmasked + masked) + MSE on masked patches only
    """
    def __init__(self, encoder_ckpt_path,
                 lora_rank=32, lora_alpha=64,
                 patch_size=(8, 8, 8), mask_ratio=0.75,
                 strd_num_heads=8, strd_L_s=1):
        super().__init__()

        self.patch_masker = PatchMasker3D(
            in_channels=4, patch_size=patch_size, mask_ratio=mask_ratio
        )

        self.encoder = VISTA3DEncoder(encoder_ckpt_path, freeze=True)

        # Inject LoRA
        self._lora_rank  = lora_rank
        self._lora_alpha = lora_alpha
        self._setup_lora()

        # STRD attention on bottleneck (channel dim = 768 for SegResNetDS2)
        self.strd_attention = SpatialSTRDAttention(
            dim=768, num_heads=strd_num_heads, L_s=strd_L_s
        )

        self.decoder = UNetDecoder3D(out_channels=4)

        self._patch_size = patch_size

    def _setup_lora(self):
        lora_config = LoraConfig(
            r=self._lora_rank,
            lora_alpha=self._lora_alpha,
            target_modules=(
                ["encoder.conv_init"]
                + [f"encoder.layers.{l}.blocks.{b}.{c}"
                   for l in range(5) for b in range(4) for c in ["conv1", "conv2"]]
            ),
            lora_dropout=0.05,
            bias='none',
            modules_to_save=[],
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.encoder.print_trainable_parameters()

    def _bottleneck_with_strd(self, features):
        """
        Apply STRD attention to the bottleneck feature map (features[4]).
        features[4]: (B, 768, D16, H16, W16)
        Returns modified features list with updated bottleneck.
        """
        B, C, d, h, w = features[4].shape

        # Flatten spatial dims → token sequence
        # (B, 768, d, h, w) → (B, d*h*w, 768)
        tokens = features[4].flatten(2).transpose(1, 2)   # (B, N, 768)

        # Build integer spatial grid coordinates for STRD neighbor computation
        d_idx = torch.arange(d, device=tokens.device)
        h_idx = torch.arange(h, device=tokens.device)
        w_idx = torch.arange(w, device=tokens.device)
        dd, hh, ww = torch.meshgrid(d_idx, h_idx, w_idx, indexing='ij')
        coords = torch.stack([dd.flatten(), hh.flatten(), ww.flatten()], dim=1)  # (N, 3)

        # Apply STRD attention
        tokens = self.strd_attention(tokens, coords)       # (B, N, 768)

        # Reshape back to spatial feature map
        refined_bottleneck = tokens.transpose(1, 2).reshape(B, C, d, h, w)

        # Replace bottleneck in features (keep other scales unchanged for skip connections)
        new_features = list(features)
        new_features[4] = refined_bottleneck
        return new_features

    def forward(self, x):
        """
        Returns:
          recon:        (B, 4, D, H, W) — full reconstruction
          patch_mask:   (B, N) bool     — True = masked patch
          orig_patches: (B, N, patch_dim)
          grid_shape:   (nD, nH, nW)
        """
        # 1. Mask input patches
        x_masked, patch_mask, orig_patches, grid_shape = self.patch_masker(x)

        # 2. Encode masked input
        features = self.encoder(x_masked)

        # 3. STRD attention on bottleneck
        features = self._bottleneck_with_strd(features)

        # 4. Decode
        recon = self.decoder(features)

        return recon, patch_mask, orig_patches, grid_shape

    def extract_features(self, x):
        """Inference: extract full feature pyramid (no masking)."""
        with torch.no_grad():
            features = self.encoder(x)
            features = self._bottleneck_with_strd(features)
        return features


# =============================================================================
# 7. Loss Functions
# =============================================================================
def masked_patch_mse(recon, patch_masker, patch_mask, orig_patches, grid_shape):
    """
    Compute MSE loss *only* on masked patches — the core MAE objective.
    recon:        (B, 4, D, H, W)
    patch_mask:   (B, N) bool
    orig_patches: (B, N, patch_dim)
    Returns scalar loss.
    """
    # Patchify reconstruction
    recon_patches, _, _ = patch_masker.patchify(recon)   # (B, N, patch_dim)

    # Select only masked positions
    mask = patch_mask                                     # (B, N)
    recon_masked = recon_patches[mask]                   # (num_masked_total, patch_dim)
    orig_masked  = orig_patches[mask]

    return F.mse_loss(recon_masked, orig_masked)


def gradient_loss_3d(pred, target):
    dy_p = torch.abs(pred[:, :, 1:, :, :]   - pred[:, :, :-1, :, :])
    dy_t = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
    dx_p = torch.abs(pred[:, :, :, 1:, :]   - pred[:, :, :, :-1, :])
    dx_t = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
    dz_p = torch.abs(pred[:, :, :, :, 1:]   - pred[:, :, :, :, :-1])
    dz_t = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
    return (torch.abs(dy_p - dy_t).mean()
            + torch.abs(dx_p - dx_t).mean()
            + torch.abs(dz_p - dz_t).mean())


# =============================================================================
# 8. Checkpoint utilities
# =============================================================================
def save_checkpoint(model, save_dir, epoch, loss, is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    model.encoder.save_pretrained(os.path.join(save_dir, "encoder_lora"))
    torch.save({
        'decoder_state_dict':      model.decoder.state_dict(),
        'strd_attention_state':    model.strd_attention.state_dict(),
        'patch_masker_state':      model.patch_masker.state_dict(),
        'epoch': epoch, 'loss': loss,
    }, os.path.join(save_dir, "decoder.pt"))
    if is_best:
        model.encoder.save_pretrained(os.path.join(save_dir, "best_encoder_lora"))
    print(f"  → Checkpoints saved to {save_dir}")


# =============================================================================
# 9. Main Training Loop
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")

    DATA_DIR = RC.FM_TRAINING_DIR
    MODEL_CKPT = RC.VISTA3D_CKPT
    SAVE_DIR = RC.FM_SAVE_DIR

    FM_SELECT_ON_VAL = os.environ.get("FM_SELECT_ON_VAL", "1") == "1"
    FM_CKPT_EVERY    = int(os.environ.get("FM_CKPT_EVERY", 5))

    # Crop/patch are env-overridable so the SAME recipe can be pretrained on downsampled phantoms
    # (L2A ds2/ds4). They must scale WITH the downsampling factor: a fixed 64^3 crop does not fit a
    # ds4 volume (56,64,75) and the loader would silently zero-pad it (12.5% artificial boundary on
    # one side), teaching the encoder fake structure. Resolution-matched crops (ds2->32, ds4->16)
    # cover the same PHYSICAL volume with zero padding. Scale PATCH with it to keep the token count
    # (crop/patch)^3 constant, so the MAE task difficulty is comparable across scales.
    _cs = int(os.environ.get("FM_CROP", 64)); _ps = int(os.environ.get("FM_PATCH", 8))
    CROP_SIZE   = (_cs, _cs, _cs)
    PATCH_SIZE  = (_ps, _ps, _ps)   # (crop/patch)^3 tokens per crop (default 8^3 = 512)
    MASK_RATIO  = 0.75              # mask 75% of patches (MAE-style)
    STRD_L_S    = 1                 # spatial STRD window half-length (3×3×3 cube)
    # V15 ran this at 16 on a shared card. V16 has the GPU to itself, so the batch is
    # env-configurable: a larger batch cuts the wall-clock of a 300-epoch run, but it also cuts the
    # NUMBER of gradient steps at fixed epochs -- and the V15 post-mortem found this encoder
    # UNDER-trained (VAL still falling at epoch 100). So raise the batch only together with FM_REPEAT,
    # which restores the step count by drawing more crops per epoch.
    BATCH_SIZE  = int(os.environ.get("FM_BATCH", 16))
    LR          = float(os.environ.get("FM_LR", 5e-4))
    EPOCHS      = int(os.environ.get("FM_EPOCHS", 50))   # override via env for full runs
    LORA_RANK   = 32
    LORA_ALPHA  = 64

    print("=" * 65)
    print("FM_tune_t4 — MAE + Spatial STRD — LoRA Domain Adaptation (Optical Coeff.)")
    print("=" * 65)
    print(f"  Crop size    : {CROP_SIZE}")
    print(f"  Patch size   : {PATCH_SIZE}  (MAE patchification)")
    print(f"  Mask ratio   : {MASK_RATIO}")
    print(f"  STRD L_s     : {STRD_L_S}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  LR           : {LR}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  LoRA rank    : {LORA_RANK}")
    print()

    dataset    = OpticalPropertyDataset(DATA_DIR, crop_size=CROP_SIZE)

    # Optional source-balanced sampling; square-root weighting limits oversampling
    # of the smallest source cohort.
    sampler, shuffle = None, True
    mode = os.environ.get("FM_BALANCE", "none")
    if mode != "none":
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "data_expansion"))
        # Select the source-label mapping with ``FM_SPLIT``.
        import importlib
        _SP = importlib.import_module(os.environ.get("FM_SPLIT", "v11_split"))
        from torch.utils.data import WeightedRandomSampler
        from collections import Counter
        srcs = [_SP.source_of(h) for h in dataset.heads]
        n = Counter(srcs)
        f = (lambda c: c ** 0.5) if mode == "sqrt" else (lambda c: 1.0)   # sqrt | flat
        psrc = {k: f(v) for k, v in n.items()}
        tot = sum(psrc.values())
        w_head = {k: (psrc[k] / tot) / n[k] for k in n}          # per-HEAD weight
        rep = len(dataset.file_list) // len(dataset.heads)
        weights = [w_head[_SP.source_of(h)] for h in dataset.heads] * rep
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset.file_list), replacement=True)
        shuffle = False
        print(f"[Sampler] FM_BALANCE={mode}: per-source probability "
              + ", ".join(f"{k} {100*psrc[k]/tot:.1f}% ({n[k]} heads)" for k in sorted(n)))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, sampler=sampler,
                            num_workers=int(os.environ.get("FM_WORKERS", 8)),
                            pin_memory=True, drop_last=True, persistent_workers=True)
    # Optional held-out reconstruction monitor; it does not select checkpoints.
    VAL_DIR = os.environ.get("FM_VAL_DIR")
    val_loader = None
    if VAL_DIR and os.path.isdir(VAL_DIR):
        val_loader = DataLoader(OpticalPropertyDataset(VAL_DIR, crop_size=CROP_SIZE),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        print(f"[VAL monitor] held-out reconstruction on {VAL_DIR} (logging only)")

    model = MAEDomainAutoencoder(
        encoder_ckpt_path=MODEL_CKPT,
        lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
        patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO,
        strd_num_heads=8, strd_L_s=STRD_L_S,
    ).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-2,
    )
    # LR: linear warmup -> cosine decay (stable start, smooth finish)
    warmup_epochs = max(1, int(round(0.1 * EPOCHS)))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                              total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, EPOCHS - warmup_epochs), eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )
    # SSIM weight ramps in SMOOTHLY over ~[10%,40%] of training (no hard switch).
    ssim_lo = max(1, int(round(0.10 * EPOCHS)))
    ssim_hi = max(ssim_lo + 1, int(round(0.40 * EPOCHS)))
    def ssim_weight(ep):
        return 0.85 * min(1.0, max(0.0, (ep - ssim_lo) / (ssim_hi - ssim_lo)))
    ssim_loss_fn = SSIMLoss(spatial_dims=3)
    scaler       = torch.amp.GradScaler('cuda')

    history = {k: [] for k in ('total', 'l1', 'ssim', 'grad', 'mae', 'eval', 'lr', 'val_eval')}
    best_loss, no_improve_count = float('inf'), 0
    nan_batches = 0
    start_epoch = 1

    # Resume from the last full optimizer/scheduler state when requested.
    state_p = os.path.join(SAVE_DIR, "state.pt")
    if os.environ.get("FM_RESUME") == "1" and os.path.isfile(state_p):
        st = torch.load(state_p, map_location=device)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["opt"])
        scheduler.load_state_dict(st["sched"])
        history = st["history"]; best_loss = st["best_loss"]
        no_improve_count = st["no_improve"]; start_epoch = st["epoch"] + 1
        print(f"[RESUME] from epoch {st['epoch']} (best={best_loss:.4f}) -> continuing at "
              f"{start_epoch}/{EPOCHS}", flush=True)

    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        acc = {k: 0.0 for k in ('total', 'l1', 'ssim', 'grad', 'mae', 'eval')}
        w_ssim = ssim_weight(epoch)

        for inputs in dataloader:
            inputs = inputs.to(device)          # (B, 4, D, H, W)
            optimizer.zero_grad()

            # bfloat16 retains a wide exponent range for near-constant optical crops.
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                recon, patch_mask, orig_patches, grid_shape = model(inputs)

            recon = recon.float()

            # ---- Reconstruction losses (full volume) ----
            loss_l1   = torch.abs(recon - inputs).mean()
            # SSIM assumes [0, 1]; other loss terms retain the unconstrained output.
            loss_ssim = ssim_loss_fn(recon.clamp(0.0, 1.0), inputs)
            loss_grad = gradient_loss_3d(recon, inputs)

            # ---- MAE loss: only on masked patches ----
            loss_mae = masked_patch_mse(recon, model.patch_masker,
                                        patch_mask, orig_patches.float(), grid_shape)

            # ---- Composite loss: constant L1/grad/MAE + SMOOTHLY ramped SSIM ----
            # SSIM is ramped continuously during the early epochs.
            loss = 0.5 * loss_l1 + w_ssim * loss_ssim + 0.1 * loss_grad + 0.5 * loss_mae
            # consistent best-selection metric (fixed unit weights, ramp-independent)
            eval_loss = (loss_l1 + loss_ssim + loss_grad + loss_mae).item()

            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue                       # a poisoned batch must not reach the optimizer
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            acc['total'] += loss.item();      acc['l1']   += loss_l1.item()
            acc['ssim']  += loss_ssim.item(); acc['grad'] += loss_grad.item()
            acc['mae']   += loss_mae.item();  acc['eval'] += eval_loss

        nb = len(dataloader)
        for k in acc:
            history[k].append(acc[k] / nb)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        avg_total, avg_mae, avg_eval = history['total'][-1], history['mae'][-1], history['eval'][-1]

        # ---- held-out VAL reconstruction MONITOR (logging only; RNG saved/restored
        #      so TRAINING is bit-identical; NEVER used for selection/early-stop) ----
        avg_val = float('nan')
        if val_loader is not None:
            tstate = torch.get_rng_state()
            cstate = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            nstate = np.random.get_state()
            torch.manual_seed(20240622); np.random.seed(20240622)
            model.eval(); vacc, vnb = 0.0, 0
            with torch.no_grad():
                for vin in val_loader:
                    vin = vin.to(device)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        vrec, vpm, vop, vgs = model(vin)
                    vrec = vrec.float()
                    vacc += (torch.abs(vrec - vin).mean() + ssim_loss_fn(vrec, vin)
                             + gradient_loss_3d(vrec, vin)
                             + masked_patch_mse(vrec, model.patch_masker, vpm, vop.float(), vgs)).item()
                    vnb += 1
            avg_val = vacc / max(vnb, 1); model.train()
            torch.set_rng_state(tstate)
            if cstate is not None:
                torch.cuda.set_rng_state_all(cstate)
            np.random.set_state(nstate)
        history['val_eval'].append(avg_val)

        # SELECT ON VAL, not TRAIN. Selecting on the training metric is a methodological error
        # even where it is nearly harmless: the previous run picked ep92 by train eval, whose VAL
        # (0.2141) was only +0.17% off VAL's own optimum (0.2137 @ ep99). It also masked the real
        # problem, which the VAL curve shows plainly -- VAL was still FALLING at epoch 100
        # (-0.0040 over the last 20), i.e. the encoder was UNDER-trained, not over-trained.
        sel = avg_val if (val_loader is not None and FM_SELECT_ON_VAL) else avg_eval
        improved = sel < best_loss
        if improved:
            best_loss, no_improve_count = sel, 0
            save_checkpoint(model, SAVE_DIR, epoch, best_loss, is_best=True)
        else:
            no_improve_count += 1

        # Resumable state, so a login-node CPU-time kill (RLIMIT_CPU = 4 h) costs at most
        # FM_CKPT_EVERY epochs rather than the whole run.
        if epoch % FM_CKPT_EVERY == 0 or epoch == EPOCHS:
            # The temp name MUST be unique per process. With a SHARED "state.pt.tmp", two
            # concurrent trainers (an auto-restart that overlapped a still-live predecessor)
            # interleave as: P1 saves tmp, P2 overwrites tmp, P1 renames tmp->state.pt, P2 renames
            # a tmp that no longer exists -> FileNotFoundError, and state.pt is left half-written
            # and unreadable ("PytorchStreamReader ... archive is corrupted"). That is exactly how
            # this run lost its resume state at epoch ~160.
            tmp = os.path.join(SAVE_DIR, f"state.pt.tmp.{os.getpid()}")
            torch.save({"epoch": epoch, "best_loss": best_loss,
                        "no_improve": no_improve_count, "history": history,
                        "model": model.state_dict(), "opt": optimizer.state_dict(),
                        "sched": scheduler.state_dict()}, tmp)
            os.replace(tmp, os.path.join(SAVE_DIR, "state.pt"))   # rename is atomic on the same fs

        if epoch % 10 == 0 or improved:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{EPOCHS} | train_eval={avg_eval:.4f} VAL_eval={avg_val:.4f} "
                  f"ssim={history['ssim'][-1]:.4f} mae={avg_mae:.4f} "
                  f"lr={history['lr'][-1]:.2e} best(train)={best_loss:.4f}"
                  f"{' *' if improved else ''} | {elapsed:.0f}s")

        scheduler.step()

        if no_improve_count >= 60:
            print(f"\n[EarlyStop] eval loss no improvement for 60 epochs — stop at {epoch}")
            break

        if epoch % 5 == 0 or epoch == EPOCHS:
            hist_path = os.path.join(SAVE_DIR, "training_history.json")
            os.makedirs(SAVE_DIR, exist_ok=True)
            with open(hist_path, 'w') as f:
                json.dump({
                    **history,
                    'train_losses': history['total'],   # back-compat keys
                    'mae_losses':   history['mae'],
                    'best_loss':    best_loss,
                    'config': {
                        'crop_size': CROP_SIZE, 'patch_size': PATCH_SIZE,
                        'mask_ratio': MASK_RATIO, 'strd_L_s': STRD_L_S,
                        'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
                        'batch_size': BATCH_SIZE, 'lr': LR, 'epochs': EPOCHS,
                    }
                }, f, indent=2)

    save_checkpoint(model, SAVE_DIR, epoch, best_loss, is_best=False)  # final (best saved on improvement)
    print(f"\n[DONE] Best eval loss: {best_loss:.6f}")
    print(f"       Checkpoints: {SAVE_DIR}")


if __name__ == "__main__":
    main()
