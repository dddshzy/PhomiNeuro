"""INR model definition — VERBATIM extract of train_inr_v11.py lines 147-360.

Copied rather than reimplemented on purpose: `build_model` reconstructs the exact architecture a
checkpoint's cfg describes, and any divergence here would load the published weights into a network
that means something else. Do not edit; re-extract from the source if it ever changes.
"""
import math
import torch
import torch.nn as nn


class PosEnc(nn.Module):
    """NeRF positional encoding on an arbitrary-D coord (here 4-D: x,y,z,t).

    `freqs=None` keeps the historical octave ladder f = 2^0 .. 2^(F-1), which every checkpoint
    before 2026-08-03 was trained with. An explicit list replaces it.

    WHY AN EXPLICIT LIST EXISTS. The octave ladder SKIPS integer harmonics: with F=3 it spans
    f = 1,2,4 and omits 3, even though the Nyquist limit set by the 10 gates at dt = 0.2 ns is
    f < 5, so 3 is legal and simply absent. Measured on the domain [-1,1], a candidate f=3 feature
    is 82% (sin) to 100% (cos) linearly INDEPENDENT of the existing basis, whereas a sub-fundamental
    f=0.5 is only 0.4% (sin) / 4% (cos) independent -- because the raw coordinate x is already in
    the encoding (the `enc = [x]` below) and a sinusoid that cannot complete a cycle on the domain
    is nearly collinear with it. So the gap worth filling is in the MIDDLE of the band, not below it.
    """
    def __init__(self, in_dim, num_frequencies=8, freqs=None):
        super().__init__()
        f = [2.0 ** j for j in range(num_frequencies)] if freqs is None else [float(x) for x in freqs]
        self.F = len(f)
        self.freq_list = f                       # kept plain so it round-trips through cfg as JSON
        self.register_buffer("_freqs", torch.tensor(f, dtype=torch.float32), persistent=False)
        self.out_dim = in_dim * (1 + 2 * self.F)

    def forward(self, x):
        fr = self._freqs.to(device=x.device, dtype=x.dtype)
        enc = [x]
        for f in fr:
            enc += [torch.sin(math.pi * f * x), torch.cos(math.pi * f * x)]
        return torch.cat(enc, dim=-1)


def _act(name):
    """Activation factory for the V18b5 ablation. Default 'gelu' is the only behaviour every
    checkpoint before 2026-08-02 was trained with, so build_model's cfg.get(..., "gelu") keeps them
    all loading unchanged.

    ReLU is viable here: the PINN's Laplacian is a discrete 7-point stencil (see the finite-
    difference gradient below), NOT an autograd second derivative, so a piecewise-linear network
    does not make it degenerate. The radial-monotonicity term uses only a first derivative."""
    return {"gelu": nn.GELU, "relu": nn.ReLU}[name]()


class _ResBlock(nn.Module):
    """Pre-activation residual block for the h->h trunk: x + GELU(Linear(x)).

    NOTE on INR convention: plain deep MLPs are fine to ~8 layers with GELU, so this is not
    required to avoid vanishing gradients at the depths we test. It is included because skip
    connections ARE conventional in the neural-field literature (NeRF re-injects the input at
    layer 5 of 8), and because it lets depth be varied without confounding it with optimisation
    difficulty. Our model is an ENCODING-based INR (1488-d FM pyramid), the family where the
    prevailing convention is a SMALL trunk with capacity in the encoding."""

    def __init__(self, h, activation="gelu"):
        super().__init__()
        self.fc = nn.Linear(h, h); self.act = _act(activation)

    def forward(self, x):
        return x + self.act(self.fc(x))


class INRv7(nn.Module):
    """log10 Phi(x,y,z,t); FM pyramid (xyz) + optical + light(10) + srcfeat(2)."""
    def __init__(self, pyramid_channels=(48, 96, 192, 384, 768), light_dim=10,
                 srcfeat_dim=2, num_frequencies=8, mlp_width=512, mlp_extra=0, use_pyramid=True,
                 path_dim=0, no_time=False, mlp_residual=False, num_frequencies_t=None,
                 activation="gelu", pe_freqs=None, pe_freqs_t=None, zero_feats=(),
                 drop_feats=()):
        super().__init__()
        self.use_pyramid = use_pyramid
        # INPUT-FEATURE ABLATION, applied INSIDE forward_feats rather than to the training buffers.
        #
        # WHY HERE AND NOWHERE ELSE. The predecessor of this mechanism, `--no-srcfeat`, zeroed the
        # SF buffer at train time only: it was never written to cfg and no evaluator ever read it,
        # so the ablated model was scored with the REAL srcfeat it had never been trained on. The
        # damage was measured, not assumed -- the first layer's two srcfeat columns shrank to
        # ||w|| = 0.32 against 5.59 in the unablated model, i.e. the network did learn to ignore
        # them, but 5.7% of the sensitivity survived and was fed live values at test time.
        # There are 60+ call sites of forward_feats across this repo; patching them is not a thing
        # that can be done reliably. Zeroing inside the module makes train and eval the same code
        # path by construction, and every present and future caller inherits it.
        #
        # Specs: "optical", "srcfeat", "light:<a>:<b>" (half-open column range of the 10-vector).
        # Absent from cfg -> () -> every checkpoint trained before 2026-08-04 behaves bit-identically.
        self.zero_feats = tuple(zero_feats or ())
        # DROP, as distinct from ZERO. zero_feats blanks a column group but keeps in_dim, which is
        # the right form for an ABLATION (capacity held fixed, only the information removed).
        # drop_feats removes the columns outright, which is the right form for a RECIPE that no
        # longer takes those inputs: in_dim and the first layer shrink with them.
        # Specs are the same strings: "optical", "srcfeat", "light:<a>:<b>" (the range DROPPED).
        # Absent from cfg -> () -> every earlier checkpoint keeps its original in_dim exactly.
        self.drop_feats = tuple(drop_feats or ())
        drop_opt = "optical" in self.drop_feats
        drop_sf = "srcfeat" in self.drop_feats
        # Which light columns SURVIVE, as an explicit index list. Kept as a list rather than a slice
        # so a non-contiguous drop stays expressible, and registered as a buffer so it follows the
        # model to the GPU without a per-call transfer.
        keep = [i for i in range(light_dim)
                if not any(a <= i < b for a, b in
                           [tuple(int(v) for v in s.split(":")[1:3])
                            for s in self.drop_feats if s.startswith("light:")])]
        self.light_keep = None if len(keep) == light_dim else keep
        if self.light_keep is not None:
            self.register_buffer("_lk", torch.tensor(keep, dtype=torch.long), persistent=False)
        light_dim = len(keep)
        optical_dim = 0 if drop_opt else 4
        srcfeat_dim = 0 if drop_sf else srcfeat_dim
        self._drop_opt, self._drop_sf = drop_opt, drop_sf
        self.no_time = no_time                              # CW ablation: 3-D PE (drop the dead t input)
        # path_dim>0 adds the SOURCE->POINT path integrals (tau_a, tau_sp, tau_eff, L_csf,
        # L_skull, r). Everything else the net sees is LOCAL to the point, so without these it
        # cannot know what tissue the light crossed to get there -- which is why the deep bands
        # overfit (deep TEST 2.1x TRAIN) and land worse than a constant predictor. Defaults to 0
        # so pre-path checkpoints still load.
        self.path_dim = path_dim
        # V16: BAND-LIMITED TEMPORAL ENCODING.
        # The default (num_frequencies_t=None) is V15's behaviour: one PosEnc over the 4-D (x,y,z,t)
        # coord, i.e. t gets the SAME 8-octave ladder f = 1..128 as space. But the 10 gates sample t at
        # dt = 0.2 on [-1,1], whose Nyquist limit is f < 5, so f = 8,16,32,64,128 alias. The damage is
        # not "wrong values between gates" (we only ever query the 10 gates) -- it is that the temporal
        # smoothness prior is INVERTED: measured on the gate values, adjacent gates have cosine
        # similarity -0.016 while gates 5 apart have +0.70 (for even f, cos(pi f (t+1)) == cos(pi f t)
        # exactly, so gates 1.0 apart share 7 of 8 bands). The net therefore sees 10 unordered
        # categories rather than an ordered sequence, and is free to ring between them -- which is what
        # the gate-1..4 diagnosis shows (var_ratio 1.365 vs the grid baselines' 1.086, p90AE/medAE 4.05
        # vs 3.17, median error normal).
        # With num_frequencies_t set, space and time are encoded separately and t is capped at the
        # sampling limit (F_T <= 3 keeps f <= 4 < 5). Adjacent/far contrast goes -0.054 -> +0.324.
        self.num_frequencies_t = num_frequencies_t
        # an explicit temporal frequency list implies the SPLIT encoding, exactly as an integer
        # num_frequencies_t does; otherwise a list could be silently ignored on the shared path
        split_t = (pe_freqs_t is not None) or (num_frequencies_t is not None)
        if not split_t or no_time:
            self.pos = PosEnc(3 if no_time else 4, num_frequencies, pe_freqs)
            self.pos_t = None
        else:
            self.pos = PosEnc(3, num_frequencies, pe_freqs)
            self.pos_t = PosEnc(1, num_frequencies_t or 8, pe_freqs_t)
        if use_pyramid:
            self.feat_norm = nn.LayerNorm(sum(pyramid_channels))
        in_dim = self.pos.out_dim + optical_dim + light_dim + srcfeat_dim + path_dim
        if self.pos_t is not None:
            in_dim += self.pos_t.out_dim
        if use_pyramid:
            in_dim += sum(pyramid_channels)
        h = mlp_width
        self.activation = activation
        layers = [nn.Linear(in_dim, h), _act(activation)]
        if mlp_residual:
            layers += [_ResBlock(h, activation) for _ in range(1 + mlp_extra)]
        else:
            layers += [nn.Linear(h, h), _act(activation)]
            for _ in range(mlp_extra):
                layers += [nn.Linear(h, h), _act(activation)]
        layers += [nn.Linear(h, h // 2), _act(activation), nn.Linear(h // 2, 1)]
        self.mlp = nn.Sequential(*layers)

    def _apply_zero(self, optical, light, srcfeat):
        """Blank the ablated feature groups. `light` arrives as an expand() VIEW of a (10,) scene
        tensor whose storage is shared with SceneStore's cache, so it must be materialised before
        any in-place write or the ablation would leak into every later scene in the same process."""
        for spec in self.zero_feats:
            if spec == "optical":
                optical = torch.zeros_like(optical)
            elif spec == "srcfeat":
                srcfeat = torch.zeros_like(srcfeat)
            elif spec.startswith("light:"):
                a, b = (int(v) for v in spec.split(":")[1:3])
                light = light.contiguous().clone()
                light[:, a:b] = 0.0
            else:
                raise ValueError(f"unknown zero_feats spec {spec!r}")
        return optical, light, srcfeat

    def forward_feats(self, raw, xyzt_norm, optical, light, srcfeat, path=None):
        n = xyzt_norm.shape[0]
        if light.dim() == 1:
            light = light.unsqueeze(0).expand(n, -1)
        if self.zero_feats:                       # after the expand, so a light: range sees (n,10)
            optical, light, srcfeat = self._apply_zero(optical, light, srcfeat)
        parts = []
        if self.use_pyramid:
            parts.append(self.feat_norm(raw))
        if self.pos_t is not None:                                  # V16: separate, band-limited t
            parts += [self.pos(xyzt_norm[:, :3]), self.pos_t(xyzt_norm[:, 3:4])]
        else:
            coords = xyzt_norm[:, :3] if self.no_time else xyzt_norm  # drop the dead t column if no_time
            parts.append(self.pos(coords))
        if self.light_keep is not None:
            light = light.index_select(1, self._lk)
        if not self._drop_opt:
            parts.append(optical)
        parts.append(light)
        if not self._drop_sf:
            parts.append(srcfeat)
        if self.path_dim:
            parts.append(path)
        return self.mlp(torch.cat(parts, dim=-1))


# --------------------------------------------------------------------------- #
# scenes
def build_model(cfg):
    """Reconstruct the model from a checkpoint cfg. Use this everywhere instead of calling INRv7
    directly: `num_freq` (the PE band-limit) MUST come from the cfg, or a band-limited checkpoint is
    silently rebuilt with the default 8 bands and the loaded weights mean something else. Old
    checkpoints have no `num_freq` -> 8, their historical value."""
    return INRv7(mlp_width=cfg["mlp_width"], mlp_extra=cfg["mlp_extra"],
                 use_pyramid=cfg["use_pyramid"], path_dim=cfg.get("path_dim", 0),
                 num_frequencies=cfg.get("num_freq", 8), no_time=cfg.get("no_time", False),
                 num_frequencies_t=cfg.get("num_freq_t"),   # None for every pre-V16 checkpoint
                 activation=cfg.get("activation", "gelu"),  # 'gelu' for every pre-V18 checkpoint
                 pe_freqs=cfg.get("pe_freqs"), pe_freqs_t=cfg.get("pe_freqs_t"),  # None = octave ladder
                 zero_feats=cfg.get("zero_feats", ()),      # () for every pre-2026-08-04 checkpoint
                 drop_feats=cfg.get("drop_feats", ()),      # () keeps every earlier in_dim exact
                 mlp_residual=cfg.get("mlp_residual", False))
