#!/usr/bin/env python3
"""
Benchmark baseline surrogates for the FM-INR fluence task — apples-to-apples on the
v10 split.  Two interface families share the SAME inputs/target as our model:

COORDINATE family (point -> log10 Phi), drop-in for INRv6.forward_feats(raw, xyz_norm,
optical, light, srcfeat).  `raw` (FM pyramid) is IGNORED -> these are exactly our
model's inputs MINUS the FM pyramid, isolating the pyramid's contribution:
  - FourierMLP : Gaussian random Fourier features on the coord + MLP (Tancik 2020)
  - SIREN      : sine-activation MLP with principled init (Sitzmann 2020)

VOXEL family (property+source volume -> log10 Phi volume), on a fixed 128^3 grid:
  - FNO3d      : 3-D Fourier Neural Operator (self-impl SpectralConv3d; Li 2020)
  - make_voxel_model("unet"/"segresnet"/"dynunet") : MONAI 3-D CNNs

OPERATOR family (branch volume + trunk coord -> log10 Phi point):
  - DeepONet   : 3-D CNN branch (global code) x Fourier-MLP trunk (Lu 2021)

All coordinate/operator nets emit raw log10 Phi (no output activation), matching
INRv6 and the floored-decade target.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ============================================================================
# COORDINATE family  (forward_feats(raw, xyz_norm, optical, light, srcfeat))
# ============================================================================
COND_DIM = 4 + 10 + 2          # optical(4) + light(10) + srcfeat(2)  -- same as ours minus pyramid


class _TimeEnc(nn.Module):
    """V16: band-limited positional encoding of the gate time for the coord baselines.

    Mirrors train_inr_v11.PosEnc on a 1-D input so that `ours` and the coord baselines ingest time
    identically and the F_T comparison isolates F_T, not the architecture. num_freq=None keeps the
    V15 behaviour (raw scalar t appended, out_dim 1). The 10 gates sample t at dt=0.2 on [-1,1], so
    Nyquist is f<5 and F_T<=3 (f<=4) is the band-limited regime.
    """
    def __init__(self, num_freq=None):
        super().__init__()
        self.F = num_freq
        self.out_dim = 1 if not num_freq else 1 + 2 * num_freq

    def forward(self, t):
        if not self.F:
            return t
        freqs = 2.0 ** torch.arange(self.F, device=t.device, dtype=t.dtype)
        enc = [t]
        for fr in freqs:
            enc += [torch.sin(math.pi * fr * t), torch.cos(math.pi * fr * t)]
        return torch.cat(enc, dim=-1)


# ---------------------------------------------------------------------------
def _drop_plan(drop_feats, light_dim=10, optical_dim=4, srcfeat_dim=2):
    """Resolve DROP specs into (optical_dim, kept light indices, srcfeat_dim).

    Mirrors INRv7.drop_feats exactly, and for the same reason: the coordinate baselines were given
    optical(4) and the FULL light(10) while hero3 takes neither the optical block nor the sin/cos
    angles, so a "same inputs" claim needs the baselines to carry the same information -- no more
    and no less. Specs: "optical", "srcfeat", "light:<a>:<b>" (the range DROPPED).

    Absent -> () -> every checkpoint trained before 2026-08-06 keeps its original in_features.
    """
    df = tuple(drop_feats or ())
    rng = [tuple(int(v) for v in sp.split(":")[1:3]) for sp in df if sp.startswith("light:")]
    keep = [i for i in range(light_dim) if not any(a <= i < b for a, b in rng)]
    return (0 if "optical" in df else optical_dim,
            None if len(keep) == light_dim else keep,
            0 if "srcfeat" in df else srcfeat_dim)


class FourierMLP(nn.Module):
    """Gaussian random Fourier-feature MLP. Coord -> RFF, concat conditioning -> GELU MLP.
    path_dim>0 appends the source->point path integrals (fair physical input; NOT the FM pyramid)."""
    def __init__(self, num_features=128, sigma=6.0, hidden=512, cond_dim=COND_DIM, path_dim=0,
                 time_dim=0, num_freq_t=None, drop_feats=()):
        super().__init__()
        B = torch.randn(num_features, 3) * sigma
        self.register_buffer("B", B)                       # fixed (Tancik): not trained
        self.path_dim = path_dim; self.time_dim = time_dim
        self.tenc = _TimeEnc(num_freq_t) if time_dim else None
        tdim = self.tenc.out_dim if self.tenc is not None else 0
        self.drop_feats = tuple(drop_feats or ())
        self._odim, self.light_keep, self._sfdim = _drop_plan(self.drop_feats)
        if self.light_keep is not None:
            self.register_buffer("_lk", torch.tensor(self.light_keep, dtype=torch.long),
                                 persistent=False)
        cond_dim = self._odim + (len(self.light_keep) if self.light_keep is not None else 10) \
                   + self._sfdim
        in_dim = 2 * num_features + cond_dim + path_dim + tdim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1))

    def _rff(self, xyz_norm):
        proj = 2 * math.pi * xyz_norm @ self.B.t()         # (N, m)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward_feats(self, raw, xyz_norm, optical, light, srcfeat, domain=None, path=None, t=None):
        n = xyz_norm.shape[0]
        if light.dim() == 1:
            light = light.unsqueeze(0).expand(n, -1)
        if self.light_keep is not None:
            light = light.index_select(1, self._lk)
        parts = [self._rff(xyz_norm)]
        if self._odim:
            parts.append(optical)
        parts.append(light)
        if self._sfdim:
            parts.append(srcfeat)
        if self.path_dim and path is not None:
            parts.append(path)
        if self.time_dim and t is not None:
            parts.append(self.tenc(t))                     # (N,1) or band-limited PE of gate time
        return self.mlp(torch.cat(parts, dim=-1))


class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_f, 1.0 / in_f)
            else:
                b = math.sqrt(6.0 / in_f) / omega_0
                self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """Conditioned SIREN: [xyz, optical, light, srcfeat(, path)] -> sine MLP -> log10 Phi."""
    def __init__(self, hidden=256, depth=5, omega_0=3.0, cond_dim=COND_DIM, path_dim=0,
                 time_dim=0, out_bias=-6.0, out_scale=30.0, num_freq_t=None, drop_feats=()):
        # omega_0=3, NOT the canonical 30. The published value is tuned for PURE-COORDINATE inputs
        # (images/SDFs, 2-3 dims in [-1,1]); our input is conditioning-dominated (3 coords + 22 dims
        # of optical/light/srcfeat/path/time), where omega_0=30 makes the sine argument oscillate far
        # faster than the target field and the net cannot fit at all. Measured on a fixed 16k batch
        # (200 Adam steps, identical seed/data): omega_0=30 -> loss 1.41, 10 -> 0.214, 3 -> 0.212,
        # 1 -> 0.403. Leaving the canonical 30 would have reported a broken SIREN as a fair baseline.
        super().__init__()
        self.path_dim = path_dim; self.time_dim = time_dim
        self.tenc = _TimeEnc(num_freq_t) if time_dim else None
        tdim = self.tenc.out_dim if self.tenc is not None else 0
        self.drop_feats = tuple(drop_feats or ())
        self._odim, self.light_keep, self._sfdim = _drop_plan(self.drop_feats)
        if self.light_keep is not None:
            self.register_buffer("_lk", torch.tensor(self.light_keep, dtype=torch.long),
                                 persistent=False)
        cond_dim = self._odim + (len(self.light_keep) if self.light_keep is not None else 10) \
                   + self._sfdim
        in_dim = 3 + cond_dim + path_dim + tdim
        # SIREN's first layer computes sin(omega_0 * W x), which ASSUMES inputs are ~[-1,1]. The
        # path-integral features are raw physical magnitudes (up to ~181 here), so omega_0=30 drives
        # the sine to ~5400 rad -> saturated, gradients vanish, training flatlines (measured: train
        # loss stuck at 1.15 from ep10 to ep60, val 3.03 vs RFF's 0.31). GELU-MLP baselines and our
        # INRv7 are scale-agnostic and need no such treatment, so normalising the conditioning vector
        # HERE removes an architecture-specific handicap rather than granting SIREN an advantage --
        # without it the SIREN row would understate the coord family for a purely numerical reason.
        self.cond_norm = nn.LayerNorm(cond_dim + path_dim + tdim) \
            if (cond_dim + path_dim + tdim) > 0 else None
        layers = [SineLayer(in_dim, hidden, is_first=True, omega_0=omega_0)]
        for _ in range(depth - 1):
            layers.append(SineLayer(hidden, hidden, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 1)
        with torch.no_grad():
            b = math.sqrt(6.0 / hidden) / omega_0
            self.out.weight.uniform_(-b, b)
            # SIREN's canonical final-layer init is deliberately tiny (output std ~0.01), which is
            # right for targets centred on 0 -- but log10(fluence) targets sit near -6.25 with std
            # 1.09. From a ~0 start the net has to climb 6 decades THROUGH that tiny layer, and it
            # never does (measured: loss flat at 1.15 for 50 epochs). Seeding the bias at the target
            # mean and widening the last layer to the target scale fixes the offset without touching
            # the sine trunk (verified healthy: per-layer std ~0.70, no saturation).
            self.out.weight.mul_(out_scale)
            self.out.bias.fill_(out_bias)

    def forward_feats(self, raw, xyz_norm, optical, light, srcfeat, domain=None, path=None, t=None):
        n = xyz_norm.shape[0]
        if light.dim() == 1:
            light = light.unsqueeze(0).expand(n, -1)
        if self.light_keep is not None:
            light = light.index_select(1, self._lk)
        cond = []
        if self._odim:
            cond.append(optical)
        cond.append(light)
        if self._sfdim:
            cond.append(srcfeat)
        if self.path_dim and path is not None:
            cond.append(path)
        if self.time_dim and t is not None:
            cond.append(self.tenc(t))                      # (N,1) or band-limited PE of gate time
        c = torch.cat(cond, dim=-1)
        if self.cond_norm is not None:
            c = self.cond_norm(c)                          # keep sine arguments in range
        return self.out(self.net(torch.cat([xyz_norm, c], dim=-1)))


# ============================================================================
# VOXEL family  (volume (B,Cin,X,Y,Z) -> (B,1,X,Y,Z))
# ============================================================================
class SpectralConv3d(nn.Module):
    """3-D spectral convolution: keep the lowest (m1,m2,m3) Fourier modes (Li 2020)."""
    def __init__(self, in_c, out_c, m1, m2, m3):
        super().__init__()
        self.in_c, self.out_c, self.m1, self.m2, self.m3 = in_c, out_c, m1, m2, m3
        s = 1.0 / (in_c * out_c)
        # 4 independent weight blocks for the 4 corners of the real-FFT spectrum.
        # Stored REAL (...,2) -> view_as_complex in forward, so AMP/GradScaler (which
        # cannot unscale ComplexFloat grads) sees real-valued parameters.
        def w():
            return nn.Parameter(s * torch.rand(in_c, out_c, m1, m2, m3, 2))
        self.w1, self.w2, self.w3, self.w4 = w(), w(), w(), w()

    @staticmethod
    def _mul(a, b):
        return torch.einsum("bixyz,ioxyz->boxyz", a, torch.view_as_complex(b))

    def forward(self, x):
        B, _, X, Y, Z = x.shape
        # FFT path forced to fp32/complex64 — ComplexHalf has no baddbmm (breaks under AMP)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            xf = torch.fft.rfftn(x, dim=[-3, -2, -1])
            out = torch.zeros(B, self.out_c, X, Y, Z // 2 + 1, dtype=torch.cfloat, device=x.device)
            m1, m2, m3 = self.m1, self.m2, self.m3
            out[:, :, :m1, :m2, :m3] = self._mul(xf[:, :, :m1, :m2, :m3], self.w1)
            out[:, :, -m1:, :m2, :m3] = self._mul(xf[:, :, -m1:, :m2, :m3], self.w2)
            out[:, :, :m1, -m2:, :m3] = self._mul(xf[:, :, :m1, -m2:, :m3], self.w3)
            out[:, :, -m1:, -m2:, :m3] = self._mul(xf[:, :, -m1:, -m2:, :m3], self.w4)
            return torch.fft.irfftn(out, s=(X, Y, Z), dim=[-3, -2, -1])


class FNO3d(nn.Module):
    """3-D Fourier Neural Operator: lift -> 4x(spectral + 1x1 conv) -> project."""
    def __init__(self, in_c=4, width=20, modes=12, depth=4, out_c=1):
        super().__init__()
        self.lift = nn.Conv3d(in_c, width, 1)
        self.spec = nn.ModuleList([SpectralConv3d(width, width, modes, modes, modes) for _ in range(depth)])
        self.w = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv3d(width, 128, 1), nn.GELU(), nn.Conv3d(128, out_c, 1))

    def forward(self, x):
        x = self.lift(x)
        for sp, w in zip(self.spec, self.w):
            x = F.gelu(sp(x) + w(x))
        return self.proj(x)


# ============================================================================
# OPERATOR family  (DeepONet: branch(volume) . trunk(coord))
# ============================================================================
class DeepONet(nn.Module):
    """Branch = 3-D CNN over (mu_a,mu_s,src,dir) volume -> global code p (K).
    Trunk = Fourier-MLP over the query coord -> q (K).  Output = <p,q> + bias."""
    def __init__(self, in_c=4, K=256, trunk_features=128, sigma=6.0):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv3d(ci, co, 3, 2, 1), nn.InstanceNorm3d(co), nn.GELU())
        self.branch = nn.Sequential(blk(in_c, 32), blk(32, 64), blk(64, 128), blk(128, 128),
                                    nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(128, K))
        B = torch.randn(trunk_features, 3) * sigma
        self.register_buffer("B", B)
        self.trunk = nn.Sequential(nn.Linear(2 * trunk_features, 256), nn.GELU(),
                                   nn.Linear(256, 256), nn.GELU(), nn.Linear(256, K))
        self.bias = nn.Parameter(torch.zeros(1))
        self.K = K

    def branch_code(self, vol):                            # vol (B,Cin,X,Y,Z) -> (B,K)
        return self.branch(vol)

    def _rff(self, xyz_norm):
        proj = 2 * math.pi * xyz_norm @ self.B.t()
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def trunk_code(self, xyz_norm):                        # (N,3) -> (N,K)
        return self.trunk(self._rff(xyz_norm))

    def forward(self, vol_code, xyz_norm):
        """vol_code (B,K) per-volume branch code; xyz_norm (N,3) -> (B,N) log10 Phi."""
        q = self.trunk_code(xyz_norm)                      # (N,K)
        return vol_code @ q.t() + self.bias                # (B,N)


# ============================================================================
# MONAI 3-D CNNs (UNet / SegResNet / DynUNet) — lazy import so torch-only paths
# never depend on MONAI
# ============================================================================
def make_voxel_model(arch, in_c=4, out_c=1):
    a = arch.lower()
    if a == "fno":
        return FNO3d(in_c=in_c, out_c=out_c)
    from monai.networks.nets import UNet, SegResNet, DynUNet
    if a in ("unetr", "vit"):                         # grid TRANSFORMER (ViT encoder + CNN decoder)
        from monai.networks.nets import UNETR
        return UNETR(in_channels=in_c, out_channels=out_c, img_size=(128, 128, 128),
                     feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                     proj_type="conv", norm_name="instance", res_block=True)
    if a == "unet":
        return UNet(spatial_dims=3, in_channels=in_c, out_channels=out_c,
                    channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2),
                    num_res_units=2)
    if a in ("resunet", "segresnet"):
        return SegResNet(spatial_dims=3, in_channels=in_c, out_channels=out_c,
                         init_filters=16, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
    if a in ("dynunet", "nnunet"):
        ks = [[3, 3, 3]] * 5
        st = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        return DynUNet(spatial_dims=3, in_channels=in_c, out_channels=out_c,
                       kernel_size=ks, strides=st, upsample_kernel_size=st[1:],
                       res_block=True)
    raise ValueError(f"unknown voxel arch {arch}")


def make_coord_model(arch, path_dim=0, time_dim=0, num_freq_t=None, drop_feats=()):
    """time_dim=1 builds the TIME-RESOLVED variant: the normalized gate time is appended to the
    conditioning vector, exactly how our INRv7 ingests time, so the coord baseline stays
    apples-to-apples with ours (same physical inputs, no FM pyramid)."""
    a = arch.lower()
    if a in ("rff", "fourier", "fouriermlp"):
        return FourierMLP(path_dim=path_dim, time_dim=time_dim, num_freq_t=num_freq_t,
                          drop_feats=drop_feats)
    if a == "siren":
        return SIREN(path_dim=path_dim, time_dim=time_dim, num_freq_t=num_freq_t,
                     drop_feats=drop_feats)
    raise ValueError(f"unknown coord arch {arch}")


if __name__ == "__main__":
    # smoke: param counts + forward shapes
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N = 4096
    xn = torch.rand(N, 3, device=dev) * 2 - 1
    opt = torch.rand(N, 4, device=dev); light = torch.rand(10, device=dev); sf = torch.rand(N, 2, device=dev)
    for nm, m in [("FourierMLP", FourierMLP()), ("SIREN", SIREN())]:
        m = m.to(dev); y = m.forward_feats(None, xn, opt, light, sf)
        print(f"{nm:12s} params={count_params(m):,}  out={tuple(y.shape)}")
    vol = torch.rand(1, 4, 64, 64, 64, device=dev)
    fno = FNO3d().to(dev); print(f"FNO3d        params={count_params(fno):,}  out={tuple(fno(vol).shape)}")
    do = DeepONet().to(dev); code = do.branch_code(vol); y = do(code, xn)
    print(f"DeepONet     params={count_params(do):,}  branch={tuple(code.shape)} out={tuple(y.shape)}")
