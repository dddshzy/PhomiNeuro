"""FM encoder architecture — VERBATIM extract of FM_tune_t4/train_lora_autoencoder.py.

Only the MODEL classes are kept: PatchMasker3D, SpatialSTRDAttention, VISTA3DEncoder, ResBlock3D,
UNetDecoder3D, MAEDomainAutoencoder. The training loop, its dataset class and its loss functions are
dropped -- they are not reachable from inference, and the originals carry absolute paths into the
research machine that would only mislead a reader here.

Copied rather than reimplemented so the released adapter weights load into exactly the module tree
they were saved from.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


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
