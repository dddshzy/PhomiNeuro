#!/usr/bin/env python3
"""Train the time-resolved INR surrogate for ``log10 Phi(x, y, z, t)``.

The optical feature pyramid is spatial and fixed over the simulation window;
time enters the coordinate encoding. Training samples one of ten time gates per
spatial point and supports decade-aware data terms, censored-floor losses, and
optional time-domain diffusion regularization.
"""
import os, sys, re, glob, json, time, math, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
import optical_config as OC
import repro_config as RC
import inr_dataset as D1                       # sample_volume
from train_inr_v3 import xyz_to_norm, sample_raw
# Select the cohort split with ``INR_SPLIT``.
import importlib
SPLIT = importlib.import_module(os.environ.get("INR_SPLIT", "v11_split"))

PROP_KEY = "vol_prop_eye_aseg"
FEATDIM = sum((48, 96, 192, 384, 768))
LIGHT_DIM = 10
MONO_VS = torch.tensor([224.0, 256.0, 300.0])   # rigid-MNI grid (constant); for the mono unit-scale
N_STEP = 10
R_SCALE = 128.0
FLU_FLOOR_DECADES = 8.0
# POINT-SAMPLING pools (baked into the buffer at precompute -> changing these needs a rebuild).
# REACH_DEC defines the Level-1 region-B band (within this many decades of the scene peak); a
# REACH_FRAC share of every scene's points is drawn from it, reproducing v10's dense near-source
# sampling on the band the surrogate is actually deployed in. REACH_FRAC=0 -> historical behaviour.
REACH_DEC   = float(os.environ.get("INR_REACH_DEC", 6.0))
REACH_FRAC  = float(os.environ.get("INR_REACH_FRAC", 0.0))
BRIGHT_FRAC = float(os.environ.get("INR_BRIGHT_FRAC", 0.5))
DATASET = RC.MNI_DATASET_DIR
PYRDIR = RC.PYRAMID_DIR
SIMDIR = RC.MNI_SIM_DIR
CKPT = RC.INR_CHECKPOINT_DIR
# Decade thresholds define shallow, intermediate, and censored regions.
DEC_SH, DEC_MID, DEC_DEEP = 2.5, 6.5, 8.0
SHALLOW_BOOST, CSF_BOOST = 3.0, 3.0

# ---- time-domain diffusion PINN ------------------------------------------------
# The gate values are TIME-INTEGRALS of the fluence rate over dt=0.2 ns, not
# instantaneous rates. Because the diffusion equation is LINEAR and D, mu_a are
# time-independent, integrating it over a gate gives EXACTLY
#     (1/c)[phi(t_{k+1}) - phi(t_k)] = div(D grad Phi_k) - mu_a Phi_k
# and approximating the gate mean by the rate at the gate CENTRE (mean-value thm)
# turns the edge-rate difference into a CENTRED difference across neighbouring gates:
#     R_k = (Phi_{k+1} - Phi_{k-1}) / (2 c dt)  -  D lap(Phi_k)  +  mu_a Phi_k  ~= 0
# valid for k = 1..8 only (k=0/9 have no neighbour; k=0 also breaks the centre
# approximation, phi varies by decades inside gate 0, and diffusion is invalid there).
C_MM_PER_NS = 299.792458 / 1.37      # speed of light in tissue (n=1.37) = 218.8 mm/ns
DT_NS = 0.2                           # gate width
TWO_C_DT = 2.0 * C_MM_PER_NS * DT_NS  # = 87.5 mm
PINN_H = 2.0                          # spatial stencil half-step (voxels = mm)
# STOCHASTIC finite-difference stencil. A FIXED step h has a fixed high-frequency null space (modes
# the discrete Laplacian cannot see); in the UNSUPERVISED dark zone the PINN populates that null
# space with a spurious checkerboard of above-floor blobs. Jittering h per collocation point means
# no single spatial frequency is aliased for every point, so the checkerboard gets penalised
# somewhere and is suppressed. "lo,hi" (voxels) enables it; empty keeps the fixed PINN_H.
_hj = os.environ.get("INR_PINN_HJIT", "").strip()
PINN_H_LO, PINN_H_HI = (float(x) for x in _hj.split(",")) if _hj else (PINN_H, PINN_H)
# Deep-zone monotonic-decay fixes for the time-term escape (confirmed by probe_pinn_time):
DEEP_STEADY = os.environ.get("INR_DEEP_STEADY", "0") == "1"     # Plan A: drop dPhi/dt in the deep band
RADIAL_MONO = float(os.environ.get("INR_RADIAL_MONO", "0"))     # Plan B: penalise dPhi/dr>0 (needs P.RHAT)
# Source-dependent features can be referenced to the MCX source or the scalp entry point.
SRC_REF = os.environ.get("INR_SRC_REF", "source")

# ---------------------------------------------------------------------------
# Evaluation protocol constants -- ONE definition, imported by every evaluator and runner, so the
# electrode set cannot drift between scripts.
#
# DEV-TEST vs HELD-OUT. The 20 heads below (bw 3 + scb 2 + sh 15) have been scored repeatedly during
# ablation and architecture selection, so they are a DEVELOPMENT test set and must be named as such;
# the 84 OASIS heads have never taken part in any selection and are the true held-out set.
#
# E19 = the standard 10-20 montage. Every one of the 20 dev-test heads has all 19 (verified: 380
# scenes, zero missing r5_t10 fluence files, exactly 19 per head = a balanced design). The dataset
# also holds 45 aug* random directions, but only 5 heads have all of them and the rest have 15-20,
# so including them would unbalance the per-head averages; they are deliberately excluded.
E19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
       "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
DEVTEST_HEADS = ["bw14", "bw15", "bw16", "scb15", "scb16",
                 "sh001", "sh027", "sh031", "sh032", "sh039", "sh076", "sh077", "sh102",
                 "sh113", "sh127", "sh132", "sh145", "sh171", "sh181", "sh191"]

# Six-electrode subset for the metrics that sum over EVERY tissue voxel (energy conservation, dark
# zone) and so cannot afford all 19 on the 84-head held-out set. Chosen by ANATOMY and frozen BEFORE
# any model was scored on the held-out split -- never by looking at performance:
#   Fp1 frontal pole   -- frontal sinus air cavity under the thinnest skull, the hardest geometry
#   F7  left frontotemporal -- temporalis muscle, thick soft tissue
#   Cz  vertex         -- largest scalp-to-deep-target distance, the conventional reference site
#   T4  right temporal -- temporalis + lateral, mirrors F7 across the midline
#   Pz  parietal midline
#   O1  left occipital -- thickest skull, cerebellum beneath
# Together they span anterior->posterior, midline<->lateral, left<->right, and both ends of the
# difficulty range.
E6_ENERGY = ["Fp1", "F7", "Cz", "T4", "Pz", "O1"]
PINN_PER_SCENE = int(os.environ.get("INR_PINN_PER", 120))
# D2 (sub-floor checkerboard): the artefact lives in the UNCONSTRAINED gaps between collocation
# points (120/scene = 1 point per ~42,000 voxels), so DENSITY (INR_PINN_PER) is the main lever and
# this is the aim -- the share of points forced into the deep band where the checkerboard lives.
PINN_DEEP_FRAC = float(os.environ.get("INR_PINN_DEEP_FRAC", 0.0))
PINN_DEEP_DEC  = float(os.environ.get("INR_PINN_DEEP_DEC", 6.0))
# Load-time cap on the collocation count (0 = use all of the buffer's npin). Lets a DENSE buffer be
# trained as if it were sparse, so the density lever is attributable without a second rebuild.
PINN_NCAP = int(os.environ.get("INR_PINN_NCAP", 0))
# Clamp on the log-RATIO used inside the residual. The loss (num/den)^2 is BOUNDED by 1
# (triangle inequality), but its GRADIENT is not: d(10^dlog)/d(logPhi) = r*ln10, so a +-20
# clamp lets r reach 1e20 and produces ~1e20 gradients that blow the model up (observed:
# train loss -> 72). Physically, across the h=2 mm stencil even the most attenuating tissue
# (WM: mu_eff = sqrt(3 mu_a mu_s') = 1.17 /mm) drops only ~1 decade, so |dlog| <= 2.5 is
# already generous, and it bounds the gradient at ~10^2.5 * ln10.
LOGD_CLAMP = float(os.environ.get("INR_LOGD_CLAMP", 2.5))
PINN_WARMUP = int(os.environ.get("INR_PINN_WARMUP", 20))   # ramp lambda_pinn 0 -> full


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
        # An absent setting preserves the full input feature vector.
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
                 drop_feats=cfg.get("drop_feats", ()),
                 mlp_residual=cfg.get("mlp_residual", False))


# --------------------------------------------------------------------------- #
def discover_scenes():
    scenes = []
    for flu in sorted(glob.glob(os.path.join(SIMDIR, "fluence_*_F810_*_r5_t10.mat"))):
        b = os.path.basename(flu)
        m = re.match(r"fluence_([a-z0-9]+)_F810_(.+)_r5_t10\.mat$", b)
        head, tag = m.group(1), m.group(2)
        meta = flu.replace("fluence_", "meta_").replace(".mat", ".json")
        pyr = os.path.join(PYRDIR, f"{head}_v11mni_F810_pyramid.pt")
        prop = os.path.join(DATASET, f"{head}_v11mni_F810.mat")
        if os.path.isfile(meta) and os.path.isfile(pyr):
            scenes.append(dict(head=head, tag=tag, geom_key=head, pyramid=pyr,
                               prop=prop, fluence=flu, meta=meta))
    return scenes


def label_of(head):
    if head.startswith(getattr(SPLIT, "OASIS_PREFIX", "\0")): return "test"   # external validation
    if head in SPLIT.TEST_HEADS: return "test"
    if head in SPLIT.VAL_HEADS:  return "val"
    return "train"


def src_features(xyz, srcpos, srcdir):
    u = xyz - srcpos.view(1, 3)
    r = u.norm(dim=1, keepdim=True).clamp_min(1e-3)
    cos = (u * srcdir.view(1, 3)).sum(1, keepdim=True) / r
    return torch.cat([r / R_SCALE, cos], dim=1)


class SceneStore:
    def __init__(self, device):
        self.device = device; self._gk = self._geo = self._ik = self._il = None

    def _geom(self, sc):
        prop = sio_loadmat(sc["prop"])[PROP_KEY].astype(np.float32)
        prop = np.transpose(prop, (3, 0, 1, 2))
        prop_t = torch.from_numpy(prop).unsqueeze(0).to(self.device)
        X, Y, Z = prop_t.shape[-3:]
        csf = (prop_t[0, 0] <= 0.005) & (prop_t[0, 1] > 1e-6) & (prop_t[0, 1] <= 0.5)
        valid_idx = torch.nonzero(prop_t[0, 1] > 0, as_tuple=False)
        pyr = torch.load(sc["pyramid"], map_location="cpu")["pyramid"]
        pyramid = [p.to(self.device) for p in pyr]
        return dict(prop=prop_t, pyramid=pyramid, csf=csf, valid_idx=valid_idx,
                    tissue_np=prop[1] > 0, vol_shape=torch.tensor([X, Y, Z], device=self.device))

    def _illum(self, sc, geo):
        X, Y, Z = [int(v) for v in geo["vol_shape"].tolist()]
        flu = sio_loadmat(sc["fluence"])["fluence"].astype(np.float32)   # (X,Y,Z,10)
        fmax = float(flu.max()); floor = max(1e-12, fmax * 10.0 ** (-FLU_FLOOR_DECADES))
        logflu = np.log10(np.clip(flu, floor, None))                    # (X,Y,Z,10)
        logflu_t = torch.from_numpy(logflu).to(self.device)             # (X,Y,Z,10)
        # brightest voxel over all gates -> bright sampling pool
        f0 = flu.max(axis=3)
        lf0 = np.log10(np.clip(f0, floor, None))
        bright_np = geo["tissue_np"] & (lf0 > np.log10(floor) + 1e-4)
        bright_idx = torch.from_numpy(np.argwhere(bright_np)).to(self.device)
        # LEVEL-1 (region-B) pool: voxels within REACH_DEC decades of the scene peak -- the
        # high-confidence band the surrogate is actually deployed in. `bright_idx` spans the whole
        # >floor field (~everything), so it is NOT a near-source lever; this one is. v10's dense
        # near-source sampling is reproduced by drawing a fixed fraction from here.
        reach_np = geo["tissue_np"] & (lf0 > math.log10(fmax) - REACH_DEC)
        reach_idx = torch.from_numpy(np.argwhere(reach_np)).to(self.device)
        meta = json.load(open(sc["meta"]))
        srcpos = torch.tensor(meta["srcpos"], dtype=torch.float32)
        srcdir = torch.tensor(meta["srcdir"], dtype=torch.float32)
        if SRC_REF == "entry":                       # V17: reference the scalp entry point instead
            from make_entryfeat import entry_of
            occ = (geo["prop"][0, 1] > 0)
            srcpos = entry_of(srcpos.to(self.device), srcdir.to(self.device),
                              occ, geo["vol_shape"]).cpu()
        srcpos_norm = srcpos / (torch.tensor([X, Y, Z], dtype=torch.float32) - 1)
        # A,B derived from srcdir (always present) so old metas without A_deg still load:
        # outward normal n=-srcdir; A=asin(n_z) elevation, B=atan2(n_x,n_y) azimuth.
        nrm = -srcdir
        A = math.asin(float(torch.clamp(nrm[2], -1, 1)))
        B = math.atan2(float(nrm[0]), float(nrm[1]))
        ang = torch.tensor([math.sin(A), math.cos(A), math.sin(B), math.cos(B)])
        light = torch.cat([srcpos_norm, srcdir, ang]).to(self.device)
        return dict(logflu=logflu_t, bright_idx=bright_idx, reach_idx=reach_idx, light=light,
                    srcpos=srcpos.to(self.device), srcdir=srcdir.to(self.device),
                    lfmax=math.log10(fmax))

    def get(self, sc):
        if sc["geom_key"] != self._gk:
            self._geo = None
            if self.device.type == "cuda": torch.cuda.empty_cache()
            self._geo = self._geom(sc); self._gk = sc["geom_key"]; self._ik = None
        if sc["fluence"] != self._ik:
            self._il = self._illum(sc, self._geo); self._ik = sc["fluence"]
        return {**self._geo, **self._il}


def sio_loadmat(p):
    import scipy.io as sio
    return sio.loadmat(p)


def sample_bright(sd, n, bright_frac=BRIGHT_FRAC, reach_frac=REACH_FRAC):
    """n voxel coords: reach_frac from the LEVEL-1 region-B pool (within REACH_DEC of the peak --
    v10's dense near-source strategy), bright_frac from the >floor bright pool, rest uniform in
    tissue. reach_frac=0 reproduces the historical two-pool behaviour exactly."""
    bi = sd["bright_idx"]; vi = sd["valid_idx"]; vs = sd["vol_shape"]
    ri = sd.get("reach_idx", bi)
    nr = int(reach_frac * n); nb = int(bright_frac * n)
    nb = min(nb, max(n - nr, 0))                       # keep the three counts consistent
    def pick(pool, k):
        if k <= 0: return torch.empty((0, 3), device=vi.device)
        if pool.shape[0] == 0: pool = vi
        return pool[torch.randint(0, pool.shape[0], (k,), device=vi.device)].float()
    xyz = torch.cat([pick(ri, nr), pick(bi, nb), pick(vi, n - nr - nb)], 0)
    xyz = xyz + (torch.rand_like(xyz) - 0.5)
    return torch.minimum(xyz.clamp(min=0), (vs.float() - 1).view(1, 3))


def gather_gates(logflu, xyz):
    """(K,3) float coords -> (K,10) log-fluence by nearest voxel over the 10 gates."""
    idx = xyz.round().long()
    X, Y, Z, _ = logflu.shape
    idx[:, 0].clamp_(0, X - 1); idx[:, 1].clamp_(0, Y - 1); idx[:, 2].clamp_(0, Z - 1)
    return logflu[idx[:, 0], idx[:, 1], idx[:, 2], :]        # (K,10)


T_ENC = (2.0 * (torch.arange(N_STEP).float() + 0.5) / N_STEP - 1.0)   # gate -> t in [-1,1]
T_CW = 0.0                                                            # CW mode: a single fixed time input


def integ_t(loggates):
    """(...,G) log10 per-gate -> (...,1) log10 of the linear time-integral, via a base-10
    log-sum-exp so a censored/large value cannot overflow 10**x. The CW (steady-state) target."""
    mx = loggates.max(dim=-1, keepdim=True).values
    return mx + torch.log10(torch.clamp((10.0 ** (loggates - mx)).sum(-1, keepdim=True), min=1e-30))


def csf_at(xyz, csf_vol, vs):
    j = torch.minimum(xyz.round().long().clamp(min=0), (vs - 1).view(1, 3).long())
    return csf_vol[j[:, 0], j[:, 1], j[:, 2]].float().unsqueeze(1)


def main():
    global DEC_SH, DEC_MID, SHALLOW_BOOST, CSF_BOOST
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds torch/numpy/cuda: init, batch shuffling and gate sampling")
    ap.add_argument("--tag", default="v11")
    ap.add_argument("--name", default="base")
    ap.add_argument("--mlp-width", type=int, default=512)
    # PE BAND-LIMIT. The positional encoding spans periods of 200/2^i voxels: at the default F=8 the
    # top bands are 6.2 / 3.1 / 1.6 voxels. Training supervises ~2242 points/scene (mean spacing
    # ~12.5 voxels), so it can only constrain wavelengths > ~25 voxels -- bands 4..7 are FINER than
    # anything the data can see, and the val objective (scored on the same sampled points) cannot see
    # them either. Those free bands are where the near-source spikes (D1) and the sub-floor
    # checkerboard (D2) live. Lowering F band-limits the model to what the supervision can police:
    # F=5 caps the finest period at 12.5 voxels. Trade-off to MEASURE, not assume: the near-source
    # gradient is steep, so too low an F may cost reachable fidelity.
    ap.add_argument("--num-freq", type=int, default=8, help="PE frequency bands (8 = historical)")
    ap.add_argument("--num-freq-t", type=int, default=None,
                    help="V16: encode t SEPARATELY with this many bands. Unset = V15 (t shares the "
                         "4-D 8-octave PE, of which 5 bands alias on a 10-gate grid). 3 is the "
                         "sampling limit (f<=4 < Nyquist 5).")
    ap.add_argument("--mlp-extra", type=int, default=0)
    ap.add_argument("--pe-freqs", type=float, nargs="+", default=None,
                    help="explicit SPATIAL PE frequencies, e.g. 1 2 3 4 5 6 7 8. Overrides\n"
                         "--num-freq's octave ladder, which skips 3,5,6,7. Sub-fundamental values\n"
                         "(<1) are near-collinear with the raw coordinate already in the encoding\n"
                         "and measured only 0.4-4% independent of it, so they are not worth adding.")
    ap.add_argument("--pe-freqs-t", type=float, nargs="+", default=None,
                    help="explicit TEMPORAL PE frequencies, e.g. 1 2 3 4. Must stay below the\n"
                         "Nyquist limit f < 5 set by the 10 gates at dt = 0.2 ns, or t aliases.\n"
                         "Implies the split encoding just as --num-freq-t does.")
    ap.add_argument("--activation", choices=["gelu", "relu"], default="gelu",
                    help="trunk activation. V18b5 ablation. Everything up to V17 is gelu; a\n                         checkpoint without this key loads as gelu (see build_model).")
    ap.add_argument("--mlp-residual", action="store_true",
                    help="residual h->h trunk blocks instead of plain ones (depth ablation).")
    ap.add_argument("--raw-cpu", action="store_true",
                    help="keep RAW.npy in host RAM instead of on the GPU (~64GB). Needed when the "
                         "card is shared; gather_raw already handles either device.")
    ap.add_argument("--shallow-boost", type=float, default=SHALLOW_BOOST)
    ap.add_argument("--dec-sh", type=float, default=DEC_SH)
    ap.add_argument("--dec-mid", type=float, default=DEC_MID)
    ap.add_argument("--csf-boost", type=float, default=CSF_BOOST)
    # PER-GATE data-term rebalancing. dec is measured against the GLOBAL (space+time) peak, which sits
    # at gate 0, so the shallow-boost only reaches gates 0-1 and gates 2-9 are stuck at mid weight ~1
    # (~4x under-weighted) -> the mid-gate R2 gap vs the grid CNNs. gate-ramp multiplies the data weight
    # by a per-gate factor (1 + gate_ramp*k), RENORMALISED to mean 1 (pure redistribution, total data-
    # loss scale unchanged), up-weighting later gates. 0 = off (V15/V16 behaviour). Modest values only
    # (0.1-0.3): later gates are photon-starved/noisier, so aggressive up-weighting fits MC noise.
    ap.add_argument("--gate-ramp", type=float, default=0.0)
    ap.add_argument("--lambda-lin", type=float, default=0.3)
    # PHYSICAL CEILING hinge: fluence cannot exceed the near-source peak, so any prediction above
    # lfmax + ceil_offset is unphysical (the +100..+200-decade near-source blow-ups that destroy R2/
    # RMSE and overflow 10**x). A dense one-sided penalty on relu(pred - (lfmax+offset)) suppresses
    # them at the source. Mirror of the floor hinge; 0 = off (back-compatible).
    ap.add_argument("--lambda-ceil", type=float, default=0.0)
    ap.add_argument("--ceil-offset", type=float, default=0.5,
                    help="ceiling = lfmax + this (decades of MC-noise headroom above the peak)")
    # CENSORED-DATA (floor) hinge: sub-floor voxels are LEFT-CENSORED ("true value <= floor",
    # the MC detection limit, NOT a physical zero). Penalise ONLY over-prediction, so the model
    # learns where light STOPS without being forced onto a photon-budget-dependent hard zero.
    ap.add_argument("--lambda-floor", type=float, default=1.0)
    ap.add_argument("--hinge-offset", type=float, default=0.0,
                    help="PRODUCTION sharp-boundary lever (hinge mode only): move the hinge "
                         "threshold DOWN to floor - offset, so the censored penalty is "
                         "relu(pred - (floor - offset))^2. offset>0 pushes dark-zone predictions "
                         "below the detection limit -> leak%%->0 and a cleaner reachable boundary. "
                         "NOTE: this is a deployment lever, NOT for the hero's Level-2 dark-zone "
                         "claim -- floor-1 aligns with the [8,10] eval band, so using it to reduce "
                         "the measured Level-2 bias would be tuning to the reference. Keep 0.0 for "
                         "the scientific hero.")
    ap.add_argument("--floor-mode", choices=["hinge", "zero"], default="hinge",
                    help="hinge: LEFT-CENSORED likelihood -- only relu(pred-floor) is penalised, "
                         "so the model may extrapolate below the detection limit (this is what "
                         "makes the Level-2 dark-zone claim possible). "
                         "zero: the dark zone is TREATED AS fluence 0 -- a symmetric MSE pins the "
                         "prediction TO the floor there. Use for the deployment model, whose only "
                         "job is to replace the 5e7 MC inside its reachable set and report ~0 "
                         "outside it; it gives up all dark-zone extrapolation by construction.")
    # time-domain diffusion PINN on the middle/deep/CENSORED bands (non-CSF). In the
    # censored region this is the ONLY shape supervision the model gets (hinge only
    # bounds it from above), so it is what makes the sub-floor extrapolation physical.
    ap.add_argument("--lambda-pinn", type=float, default=0.0)
    # V16 dense DARK-ZONE radial-monotonicity regulariser (autograd, no stencil).
    # The checkerboard is the position-encoding ringing in the hinge's unconstrained sub-floor region.
    # In non-CSF tissue with a single source the diffuse fluence decays monotonically with distance
    # (Phi ~ exp(-mu_eff r)/r), so the physical field has dPhi/dr <= 0 there; the above-floor specks
    # violate this. We penalise relu(dlogPhi/dr) DENSELY over the deep dark-zone points of the MAIN
    # batch (not the 120 sparse collocation points -- that is why Plan B was a no-op), via autograd of
    # logPhi w.r.t. the input coord (which flows through the PE where the checkerboard lives; the
    # precomputed pyramid RAW is detached, so this targets exactly the PE ringing). CSF excluded (light
    # pipes there, non-monotone). This forbids the ringing WITHOUT pinning to the floor, so unlike
    # floor_mode=zero it should not leak.
    ap.add_argument("--lambda-mono", type=float, default=0.0)
    ap.add_argument("--mono-dec-min", type=float, default=5.0,
                    help="apply the radial-monotonicity penalty only where dec >= this (deep dark zone)")
    ap.add_argument("--mono-center", choices=["source", "gate0", "src0"], default="source",
                    help="radial centre for mono: 'source'=srcpos (LIGHT[:,0:3]); 'gate0'=the GT gate-0 "
                         "max-fluence voxel (data-driven effective source, ~1 mfp inside), loaded from "
                         "CENTER.npy. gate0 is more physically accurate (GT is 90%% monotone along it).")
    ap.add_argument("--mono-rate", type=float, default=0.0,
                    help="if >0, penalise dPhi/dr_phys > -rate*mu_eff/ln10 (require decay >= rate*mu_eff), "
                         "i.e. constrain the RATE not just the sign. mu_eff = path tau_eff/r. 0.9 leaves "
                         "a 10%% margin below the pure-exponential rate. Physical units (per mm).")
    # DECOUPLED near-source exclusion for the PINN. DEC_SH is a *statistically tuned* data-
    # weighting boundary (v10 ablation -> 2.5); the PINN's near-source cut-off is a *physical*
    # validity boundary (diffusion needs r >> 1/mu_s' and t >> 1/(c mu_s')). They happen to be
    # similar but they are NOT the same quantity, so give the PINN its own knob.
    ap.add_argument("--pinn-dec-min", type=float, default=None,
                    help="min DEC for PINN collocation (default: DEC_SH)")
    # DECOUPLED scope for the shallow LINEAR term. DEC_SH is triple-coupled -- it is (a) the shallow
    # WEIGHT band, (b) this term's scope, (c) the PINN near-source cutoff (via pinn_dec_min's default).
    # Widening the weight band to cover the later gates' cores would otherwise also drag the linear
    # relative-error term into much dimmer, lower-SNR territory (where (10^d-1)^2 is not the right
    # objective) and push the PINN cutoff deeper. This flag pins the linear term's scope so the
    # weight-band experiment is single-variable. Default None = DEC_SH (unchanged behaviour).
    ap.add_argument("--lin-dec-max", type=float, default=None,
                    help="apply the shallow LINEAR term only where dec < this (default: DEC_SH)")
    # INPUT-FEATURE ABLATIONS. These blank a feature group inside INRv7.forward_feats and are
    # recorded in cfg as `zero_feats`, so build_model reconstructs the same ablation at eval time.
    # Zeroing rather than dropping the columns keeps in_dim -- and therefore model capacity --
    # identical to the baseline, which the `--pyramid 0` / `--no-path-feat` ablations do NOT: those
    # shrink in_dim. The two forms are not comparable as effect sizes and the table says so.
    ap.add_argument("--no-optical", action="store_true",
                    help="ABLATION: blank the 4 local optical properties (mua, mus, g, n).")
    ap.add_argument("--zero-light", choices=["all", "pos", "dir", "ang", "dirang"], default=None,
                    help="ABLATION on the 10-element illumination vector "
                         "light = [srcpos_norm(3), srcdir(3), sin/cos of the two angles(4)]. "
                         "`ang` and `dir` are two encodings of the SAME 2 degrees of freedom -- "
                         "srcdir = -[cosA sinB, cosA cosB, sinA] with A=asin(-srcdir_z) in "
                         "[-pi/2,pi/2] so cosA>=0 and the inversion is unique. Removing either "
                         "alone therefore leaves the beam direction fully recoverable; only "
                         "`dirang` removes it. pos = the entry-point coordinate, the one "
                         "independent part.")
    # DROP flags remove the columns (in_dim shrinks); the --no-optical / --zero-light pair above
    # only blanks them. Use DROP for a recipe that genuinely no longer takes the input, ZERO for a
    # single-variable ablation where capacity must be held fixed.
    ap.add_argument("--drop-optical", action="store_true",
                    help="RECIPE: remove the 4 optical columns entirely (in_dim -4).")
    ap.add_argument("--drop-light", choices=["pos", "dir", "ang", "dirang"], default=None,
                    help="RECIPE: remove that part of the 10-element illumination vector entirely. "
                         "`ang` is the DERIVED encoding (computed from srcdir via asin/atan2), so "
                         "dropping it and keeping srcdir removes the duplicate rather than the "
                         "primitive.")
    ap.add_argument("--no-pe", action="store_true",
                    help="ABLATION: replace both Fourier positional encodings with the plain "
                         "coordinates. PosEnc already emits the raw coordinate first (enc = [x]), "
                         "so an EMPTY frequency list is exactly the traditional xyzt input: "
                         "27+7 dims collapse to 3+1.")
    ap.add_argument("--pyramid", type=int, default=1)
    ap.add_argument("--save-buffers", default=None)
    ap.add_argument("--load-buffers", default=None)
    # MEMORY-MAPPED buffers (see consolidate_buffers.py). The 46 GB RAW / 17 GB PINN-stencil
    # feature tensors are served from the OS page cache instead of anonymous RSS, so memory
    # pressure EVICTS pages instead of OOM-killing us (RSS 117 GB -> ~5 GB). Essential on a
    # shared login node: we were 30x the next-largest process and got SIGKILLed three times.
    # --no-path-feat lets the ABLATION switch path features off even though the launcher hard-codes
    # --path-feat (argparse: same dest, last flag wins). PATHDIM=0 -> PATH.npy simply is not loaded,
    # so the ablation reuses the existing buffer with no rebuild.
    ap.add_argument("--no-path-feat", dest="path_feat", action="store_false")
    ap.add_argument("--no-srcfeat", action="store_true",
                    help="ABLATION: zero the source-relative coordinate encoding SF=(r/R_scale, "
                         "cos theta_to_srcdir), the per-point geometry-to-source feature. Keeps the "
                         "input dim (architecture unchanged) but removes the information.")
    ap.add_argument("--path-feat", action="store_true",
                    help="feed the source->point path integrals (PATH.npy) to the INR")
    ap.add_argument("--path-nseg", type=int, default=0,
                    help="mu_eff profile segments in the path feature (0 = 6 scalars only)")
    ap.add_argument("--path-suffix", default="", help="load PATH<suffix>.npy")
    ap.add_argument("--path-trilinear", action="store_true",
                    help="METADATA: declare that the buffer's PATH file was built with trilinear "
                         "grid_sample (make_pathfeat/make_entryfeat --trilinear). Recorded in "
                         "cfg['path_tri'] so evaluators sample the path the SAME way the model was "
                         "trained; it does not change training itself (PATH is pre-baked).")
    ap.add_argument("--src-ref", choices=["source", "entry"], default="source",
                    help="V17: reference point for the source-dependent features. 'entry' loads the "
                         "SF_entry/LIGHT_entry/PATH_entry side files built by make_entryfeat.py and "
                         "sets the module-level SRC_REF so SceneStore (hence every evaluator) agrees. "
                         "Recorded in cfg['src_ref']; evaluators MUST honour it or the features are "
                         "silently mismatched.")
    ap.add_argument("--lin-clamp", type=float, default=1.0,
                    help="clamp on d=pred-gt inside the shallow LINEAR term (was 2.0)")
    ap.add_argument("--load-mm", default=None, help="dir with RAW.npy / P_RAW.npy / meta.pt")
    # Full-state checkpoint + resume, so an interrupted run continues instead of restarting.
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--snap-every", type=int, default=0,
                    help="save a model SNAPSHOT every N epochs (snap_<tag>_ep<N>.pt) so the hero can "
                         "be selected post-hoc on the Level-1 instrument instead of on val loss, "
                         "which is blind to the unsupervised gaps where the artefacts live. 0=off")
    ap.add_argument("--cw", action="store_true",
                    help="V10t2 STEADY-STATE (CW) mode: target = time-INTEGRAL of the gates (single "
                         "scalar), time input fixed, and the PINN becomes the V10 steady-state Laplacian "
                         "residual (-D lap + mu_a Phi = 0; NO time term). Reuses a time-resolved buffer "
                         "(GT10 summed on the fly), so no rebuild is needed.")
    ap.add_argument("--dec-deep", type=float, default=8.0,
                    help="censoring/floor boundary in decades below the reference peak (weight->0, hinge "
                         "region). Default 8; CW integ compresses the range ~1 dec so 7 fits the CW floor.")
    ap.add_argument("--no-time", action="store_true",
                    help="CW ABLATION: build a native 3-D model (PE on x,y,z only) instead of the "
                         "4-D-with-fixed-t INRv7 -- removes the dead time input entirely. Only meaningful "
                         "with --cw.")
    a = ap.parse_args()
    # Seed model initialization, shuffling, and gate sampling together.
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed)
    DEC_SH, DEC_MID, SHALLOW_BOOST, CSF_BOOST = a.dec_sh, a.dec_mid, a.shallow_boost, a.csf_boost
    SRC_REF = a.src_ref              # SceneStore reads this; evaluators set it from cfg["src_ref"]
    DEC_DEEP = a.dec_deep                                  # local override (CW uses ~7; per-gate 8)
    pinn_dec_min = DEC_SH if a.pinn_dec_min is None else a.pinn_dec_min
    lin_dec_max = DEC_SH if a.lin_dec_max is None else a.lin_dec_max
    use_pyr = bool(a.pyramid)
    if torch.cuda.is_available(): torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    K = int(os.environ.get("INR_K", 2000)); EPOCHS = int(os.environ.get("INR_EPOCHS", 200))
    BATCH = int(os.environ.get("INR_BATCH", 65536)); LR = float(os.environ.get("INR_LR", 1e-3))
    os.makedirs(CKPT, exist_ok=True)
    tag = f"{a.tag}_{a.name}"; tenc = T_ENC.to(dev)
    PATHDIM = (6 + a.path_nseg) if a.path_feat else 0
    # Resolve the input-feature ablations into the cfg-recorded specs. Column ranges follow
    # SceneStore's `light = cat([srcpos_norm(3), srcdir(3), ang(4)])` (see the light build below).
    LIGHT_SLICE = {"all": (0, 10), "pos": (0, 3), "dir": (3, 6), "ang": (6, 10), "dirang": (3, 10)}
    zero_feats = []
    if a.no_optical:
        zero_feats.append("optical")
    if a.zero_light:
        zero_feats.append("light:{}:{}".format(*LIGHT_SLICE[a.zero_light]))
    if a.no_srcfeat:
        zero_feats.append("srcfeat")
    zero_feats = tuple(zero_feats)
    drop_feats = []
    if a.drop_optical:
        drop_feats.append("optical")
    if a.drop_light:
        drop_feats.append("light:{}:{}".format(*LIGHT_SLICE[a.drop_light]))
    drop_feats = tuple(drop_feats)
    # --no-pe: an EMPTY frequency list, not a small one. PosEnc emits the raw coordinate first, so
    # F=0 gives out_dim == in_dim and forward returns x unchanged. `[] is not None` keeps the split
    # space/time path alive, so this yields pos(3) + pos_t(1) = the plain xyzt input.
    pe_freqs = [] if a.no_pe else a.pe_freqs
    pe_freqs_t = [] if a.no_pe else a.pe_freqs_t
    model = INRv7(drop_feats=drop_feats,
                  activation=a.activation, pe_freqs=pe_freqs, pe_freqs_t=pe_freqs_t,
                  mlp_width=a.mlp_width, mlp_extra=a.mlp_extra, use_pyramid=use_pyr,
                  num_frequencies=a.num_freq, num_frequencies_t=a.num_freq_t,
                  mlp_residual=a.mlp_residual, zero_feats=zero_feats,
                  path_dim=PATHDIM, no_time=a.no_time).to(dev)
    if zero_feats:
        print(f"[{tag}] input-feature ablation: zero_feats={zero_feats}", flush=True)
    if drop_feats:
        print(f"[{tag}] input-feature DROP: drop_feats={drop_feats} -> "
              f"in_dim {model.mlp[0].in_features}", flush=True)
    if a.no_pe:
        print(f"[{tag}] --no-pe: PE removed, pos {model.pos.out_dim}-d + "
              f"pos_t {model.pos_t.out_dim if model.pos_t is not None else 0}-d (plain xyzt)",
              flush=True)
    code = {"train": 0, "val": 1}

    LIGHT_SRC = None      # V17: original source-referenced LIGHT, only for --mono-center src0
    if a.load_mm:
        # Stream the .npy files off disk SEQUENTIALLY into their final home. Reading them as
        # live np.memmaps instead was catastrophic: the per-step random gather of ~16k 3 KB rows
        # made the kernel readahead ~40x, and Lustre does not retain mmap pages, so the trainer
        # re-read 638 GB from a 46 GB file at 460 MB/s and sat at 0% GPU.
        def stream(path, device, chunk=500_000):
            mm = np.load(path, mmap_mode="r")
            t = torch.empty(mm.shape, dtype=torch.float16, device=device)
            for i in range(0, mm.shape[0], chunk):
                t[i:i + chunk] = torch.from_numpy(np.asarray(mm[i:i + chunk])).to(device)
            return t
        z = torch.load(os.path.join(a.load_mm, "meta.pt"), map_location="cpu")
        total = z["meta"]["total"]
        RAW = stream(os.path.join(a.load_mm, "RAW.npy"),
                     "cpu" if a.raw_cpu else dev) if use_pyr else None
        PATH = (stream(os.path.join(a.load_mm, f"PATH{a.path_suffix}.npy"), dev)
                if PATHDIM else None)
        _cpath = os.path.join(a.load_mm, "CENTER.npy")
        CENTER = (stream(_cpath, dev) if a.mono_center == "gate0" and os.path.exists(_cpath) else None)
        if a.mono_center == "gate0" and CENTER is None:
            raise SystemExit(f"--mono-center gate0 needs {_cpath} -- run make_center.py first")
        XN = z["XN"].to(dev); OPT = z["OPT"].to(dev); LIGHT = z["LIGHT"].to(dev)
        SF = z["SF"].to(dev); GT10 = z["GT10"].to(dev); LFM = z["LFM"].to(dev)
        CSFB = z["CSFB"].to(dev); SP = z["SP"].to(dev)
        if a.src_ref == "entry":      # V17: swap in the entry-referenced side files
            for _n, _f in (("SF", "SF_entry.npy"), ("LIGHT", "LIGHT_entry.npy")):
                _p = os.path.join(a.load_mm, _f)
                if not os.path.exists(_p):
                    raise SystemExit(f"--src-ref entry needs {_p} -- run make_entryfeat.py first")
            # V17 CONTROL: with src_ref=entry the mono term's radial centre follows LIGHT[:,0:3] to
            # the entry point. mono acts in the DEEP band (dec >= mono_dec_min), which is where the
            # late gates live, so that move is a candidate cause of the gate7-9 change. Keeping the
            # ORIGINAL (source-referenced) LIGHT lets `--mono-center src0` pin the centre where V16
            # had it while every FEATURE stays entry-referenced -- a single-variable test.
            LIGHT_SRC = LIGHT if a.mono_center == "src0" else None
            SF = stream(os.path.join(a.load_mm, "SF_entry.npy"), dev)
            LIGHT = stream(os.path.join(a.load_mm, "LIGHT_entry.npy"), dev)
            print(f"[{tag}] --src-ref entry: SF/LIGHT loaded from *_entry.npy"
                  + (" | mono centre pinned to the SOURCE point" if LIGHT_SRC is not None else ""),
                  flush=True)
        P = {"n": 0}
        if a.lambda_pinn > 0 and z["meta"]["npin"]:
            P = {k: v.to(dev) for k, v in z["P"].items()}
            # P_RAW stays in host RAM (17 GB): the PINN batch is small, so a RAM gather costs
            # nothing, and this keeps ~29 GB of headroom on a GPU we share with other users.
            P["RAW"] = stream(os.path.join(a.load_mm, "P_RAW.npy"), "cpu")
            if PATHDIM:
                P["PATH"] = stream(os.path.join(a.load_mm, f"P_PATH{a.path_suffix}.npy"), dev)
            if a.src_ref == "entry":
                P["SF"] = stream(os.path.join(a.load_mm, "P_SF_entry.npy"), dev)
                P["LIGHT"] = stream(os.path.join(a.load_mm, "P_LIGHT_entry.npy"), dev)
            P["n"] = int(z["meta"]["npin"])
            # Collocation DENSITY control at load time: the buffer bakes in npin, but P["n"] is only
            # the sampling bound (randint(0, P["n"])), so capping it yields a SPARSE-PINN control on
            # the SAME buffer -- no rebuild. This is what makes the density lever attributable:
            # dense (full npin) vs sparse (capped to the old 828k) with every other byte identical.
            if PINN_NCAP > 0:
                P["n"] = min(P["n"], PINN_NCAP)
                print(f"[{tag}] PINN collocation CAPPED to {P['n']} (density control)", flush=True)
        gb = lambda t: 0 if t is None else t.numel() * 2 / 1e9
        print(f"[{tag}] buffers total={total} pinn={P['n']} | "
              f"RAW {gb(RAW):.1f} GB {'in RAM' if a.raw_cpu else 'on GPU'}, "
              f"P_RAW {gb(P.get('RAW')):.1f} GB in RAM", flush=True)
    elif a.load_buffers:
        import glob as _g
        paths = (sorted(_g.glob(os.path.join(a.load_buffers, "*.pt")))
                 if os.path.isdir(a.load_buffers) else a.load_buffers.split(","))
        zs = [torch.load(p, map_location="cpu") for p in paths]
        cat = lambda k: torch.cat([z[k] for z in zs], 0)
        RAW = cat("RAW") if use_pyr else None
        XN = cat("XN").to(dev); OPT = cat("OPT").to(dev); LIGHT = cat("LIGHT").to(dev)
        SF = cat("SF").to(dev); GT10 = cat("GT10").to(dev); LFM = cat("LFM").to(dev)
        CSFB = cat("CSFB").to(dev); SP = cat("SP").to(dev); total = XN.shape[0]
        P = {"n": 0}
        if a.lambda_pinn > 0 and all("P" in z for z in zs):
            Ps = [z["P"] for z in zs]
            ns = [int(p["n"]) for p in Ps]
            def pcat(k):
                xs = [p[k][:n] for p, n in zip(Ps, ns) if p.get(k) is not None and n > 0]
                return torch.cat(xs, 0) if xs else None
            P = {k: pcat(k) for k in ("RAW", "XN", "OPT", "SF", "LIGHT", "D", "MUA", "GT10", "LFM")}
            for k in P:
                if k != "RAW" and P[k] is not None:
                    P[k] = P[k].to(dev)
            P["n"] = sum(ns)
        print(f"[{tag}] loaded {len(paths)} buffer shards total={total} pinn={P['n']}", flush=True)
    else:
        scenes = [s for s in discover_scenes() if label_of(s["head"]) != "test"]
        scenes.sort(key=lambda s: s["geom_key"])
        maxsc = int(os.environ.get("INR_MAXSCENES", 0))
        if maxsc: scenes = scenes[:maxsc]
        # parallel precompute: shard BY HEAD (keeps per-head pyramid caching contiguous)
        nshard = int(os.environ.get("INR_NSHARD", 1)); shard = int(os.environ.get("INR_SHARD", 0))
        if nshard > 1:
            heads = sorted(set(s["head"] for s in scenes))
            myheads = set(heads[shard::nshard])
            scenes = [s for s in scenes if s["head"] in myheads]
            print(f"[shard {shard}/{nshard}] {len(myheads)} heads, {len(scenes)} scenes", flush=True)
        from collections import Counter
        print(f"[{tag}] {len(scenes)} scenes | {dict(Counter(label_of(s['head']) for s in scenes))}", flush=True)
        store = SceneStore(dev); total = len(scenes) * K
        RAW = torch.empty(total, FEATDIM, dtype=torch.float16, device="cpu") if use_pyr else None
        XN = torch.empty(total, 3, device=dev); OPT = torch.empty(total, 4, device=dev)
        LIGHT = torch.empty(total, LIGHT_DIM, device=dev); SF = torch.empty(total, 2, device=dev)
        GT10 = torch.empty(total, N_STEP, device=dev); LFM = torch.empty(total, 1, device=dev)
        CSFB = torch.zeros(total, 1, device=dev); SP = torch.zeros(total, dtype=torch.int8, device=dev)
        # ---- time-domain PINN collocation buffer (7-point spatial stencil, NON-CSF only;
        #      the diffusion approximation fails in the clear CSF layer) ----
        PER = PINN_PER_SCENE if a.lambda_pinn > 0 else 0
        ntr = sum(1 for s in scenes if label_of(s["head"]) == "train")
        npin = ntr * PER
        P = dict(n=0,
                 RAW=(torch.empty(npin, 7, FEATDIM, dtype=torch.float16, device="cpu") if (use_pyr and npin) else None),
                 XN=torch.empty(npin, 7, 3, device=dev), OPT=torch.empty(npin, 7, 4, device=dev),
                 SF=torch.empty(npin, 7, 2, device=dev), LIGHT=torch.empty(npin, LIGHT_DIM, device=dev),
                 D=torch.empty(npin, 1, device=dev), MUA=torch.empty(npin, 1, device=dev),
                 GT10=torch.empty(npin, N_STEP, device=dev), LFM=torch.empty(npin, 1, device=dev),
                 H=torch.empty(npin, 1, device=dev),              # per-point stencil step (jitterable)
                 RHAT=torch.empty(npin, 3, device=dev))           # source->point unit dir (Plan B radial mono)
        unit_off = torch.tensor([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
                                dtype=torch.float32, device=dev)   # scaled by per-point h below
        t0 = time.time(); cur = 0
        with torch.no_grad():
            for n, sc in enumerate(scenes):
                sd = store.get(sc); vs = sd["vol_shape"]
                xyz = sample_bright(sd, K)
                xn = xyz_to_norm(xyz, vs)
                if use_pyr:
                    RAW[cur:cur+K] = sample_raw(None, xn, sd["pyramid"]).half().cpu()
                XN[cur:cur+K] = xn
                OPT[cur:cur+K] = OC.normalize_points(D1.sample_volume(sd["prop"], xyz, vs))
                LIGHT[cur:cur+K] = sd["light"].unsqueeze(0).expand(K, -1)
                SF[cur:cur+K] = src_features(xyz, sd["srcpos"], sd["srcdir"])
                GT10[cur:cur+K] = gather_gates(sd["logflu"], xyz)
                LFM[cur:cur+K] = sd["lfmax"]
                CSFB[cur:cur+K] = csf_at(xyz, sd["csf"], vs)
                SP[cur:cur+K] = code[label_of(sc["head"])]; cur += K

                if PER and label_of(sc["head"]) == "train":
                    vi = sd["valid_idx"].float()
                    noncsf = ~csf_at(vi, sd["csf"], vs).squeeze(1).bool()     # DE invalid in CSF
                    pool = vi[noncsf] if noncsf.any() else vi
                    if PINN_DEEP_FRAC > 0:
                        # DARK-ZONE TARGETING (D2): the sub-floor checkerboard lives in the gaps
                        # between collocation points, so put a fixed share of them in the deep band
                        # (>= PINN_DEEP_DEC below the peak) instead of leaving placement to the
                        # uniform draw. Density (INR_PINN_PER) is the main lever; this is the aim.
                        dsel = float(sd["lfmax"]) - gather_gates(sd["logflu"], pool).max(1).values
                        deep_pool = pool[dsel >= PINN_DEEP_DEC]
                        nd = int(PINN_DEEP_FRAC * PER)
                        if deep_pool.shape[0] == 0 or nd <= 0:
                            sel = pool[torch.randint(0, pool.shape[0], (PER,), device=dev)]
                        else:
                            sel = torch.cat([
                                deep_pool[torch.randint(0, deep_pool.shape[0], (nd,), device=dev)],
                                pool[torch.randint(0, pool.shape[0], (PER - nd,), device=dev)]], 0)
                    else:
                        sel = pool[torch.randint(0, pool.shape[0], (PER,), device=dev)]
                    hj = PINN_H_LO + (PINN_H_HI - PINN_H_LO) * torch.rand(PER, 1, 1, device=dev)  # (PER,1,1) voxels
                    pts = torch.minimum((sel.unsqueeze(1) + hj * unit_off.unsqueeze(0)).clamp(min=0),
                                        (vs.float()-1).view(1,1,3)).reshape(-1, 3)   # (PER*7,3), per-point step
                    P["H"][slice(P["n"], P["n"]+PER)] = hj.view(PER, 1)
                    rh = sel - sd["srcpos"].view(1, 3)                          # source->point (voxel=mm)
                    P["RHAT"][slice(P["n"], P["n"]+PER)] = rh / (rh.norm(dim=1, keepdim=True) + 1e-9)
                    xnf = xyz_to_norm(pts, vs)
                    phys = D1.sample_volume(sd["prop"], pts, vs)               # physical (PER*7,4)
                    ps = slice(P["n"], P["n"]+PER)
                    if use_pyr:
                        P["RAW"][ps] = sample_raw(None, xnf, sd["pyramid"]).half().view(PER,7,FEATDIM).cpu()
                    P["XN"][ps] = xnf.view(PER,7,3)
                    P["OPT"][ps] = OC.normalize_points(phys).view(PER,7,4)
                    P["SF"][ps] = src_features(pts, sd["srcpos"], sd["srcdir"]).view(PER,7,2)
                    P["LIGHT"][ps] = sd["light"].unsqueeze(0).expand(PER,-1)
                    ph = phys.view(PER,7,4)[:,0]                              # centre optical
                    mua, mus, g = ph[:,0:1], ph[:,1:2], ph[:,2:3]
                    P["MUA"][ps] = mua
                    P["D"][ps] = 1.0 / (3.0 * (mua + mus*(1-g)).clamp_min(1e-4))   # diffusion coeff
                    P["GT10"][ps] = gather_gates(sd["logflu"], sel)            # centre GT (band mask)
                    P["LFM"][ps] = sd["lfmax"]
                    P["n"] += PER
                if (n+1) % 400 == 0:
                    print(f"  precompute {n+1}/{len(scenes)} {time.time()-t0:.0f}s", flush=True)
        print(f"[{tag}] precompute {time.time()-t0:.0f}s total={total} pinn={P['n']}", flush=True)
        if a.save_buffers:
            torch.save({"RAW": RAW, "XN": XN.cpu(), "OPT": OPT.cpu(), "LIGHT": LIGHT.cpu(),
                        "SF": SF.cpu(), "GT10": GT10.cpu(), "LFM": LFM.cpu(), "CSFB": CSFB.cpu(),
                        "SP": SP.cpu(),
                        "P": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in P.items()},
                        "meta": {"total": total, "use_pyr": use_pyr, "npin": P["n"]}}, a.save_buffers)
            print(f"[{tag}] saved buffers -> {a.save_buffers}. exit.", flush=True); return

    # Feature ablations are applied by the model configuration rather than by
    # mutating cached training buffers.

    def gather_raw(arr, idx):
        """Batch-gather feature rows from a resident tensor, on whichever device it lives."""
        return arr[idx.to(arr.device)].to(dev, non_blocking=True).float()

    def rawb(idx):
        return gather_raw(RAW, idx) if use_pyr else None

    def pathb(idx):
        return gather_raw(PATH, idx) if PATHDIM else None

    n_gate_train = int(os.environ.get("INR_NGATE", N_STEP))   # <N_STEP drops the last gate(s) from
    GATE_W = None
    if a.gate_ramp > 0 and not a.cw:
        _gw = 1.0 + a.gate_ramp * torch.arange(n_gate_train, device=dev, dtype=torch.float32)
        GATE_W = _gw / _gw.mean()                             # renormalise to mean 1 (pure redistribution)
        print(f"[gate-ramp {a.gate_ramp}] per-gate data weights = "
              f"{[round(float(x),3) for x in GATE_W]}", flush=True)
    t_cw = torch.full((1, 1), T_CW, device=dev)               # CW: one fixed time input
    def batch_sample_t(b):                                    # the DATA loss (gate-9 ablation)
        """Draw a random gate per point (per-gate mode); OR, in --cw mode, use the time-INTEGRATED
        (steady-state) target with a single fixed time input. Returns (xyzt_norm, GT, DEC)."""
        if a.cw:
            gt = integ_t(GT10[b][:, :n_gate_train])          # (nb,1) CW target = log10 sum_k Phi_k
            xyzt = torch.cat([XN[b], t_cw.expand(b.numel(), 1)], dim=1)
            dec = LFM[b] - gt                                # ref = per-gate peak (~0.4 dec below CW peak)
            return xyzt, gt, dec, None                       # no per-gate index in CW mode
        k = torch.randint(0, n_gate_train, (b.numel(),), device=dev)
        gt = GT10[b].gather(1, k.view(-1, 1))                # (nb,1)
        t = tenc[k].view(-1, 1)
        xyzt = torch.cat([XN[b], t], dim=1)
        dec = LFM[b] - gt
        return xyzt, gt, dec, k

    def weights(dec, b):
        csf = CSFB[b] > 0.5
        w = torch.ones_like(dec)
        shallow = dec < DEC_SH; mid = (dec >= DEC_SH) & (dec < DEC_MID); unreach = dec >= DEC_DEEP
        wc = 1.0 + CSF_BOOST * ((dec - DEC_SH) / (DEC_MID - DEC_SH)).clamp(0, 1)
        w = torch.where(shallow, torch.full_like(w, 1.0 + SHALLOW_BOOST), w)
        w = torch.where(mid & csf, wc, w)
        w = torch.where(unreach, torch.zeros_like(w), w)
        return w

    def pinn_loss(bsz=2048):
        """Time-domain diffusion residual on the gate-INTEGRATED fluence:
             R_k = (Phi_{k+1}-Phi_{k-1})/(2 c dt) - D lap(Phi_k) + mu_a Phi_k
           (exact for a linear DE with time-independent D, mu_a; the centred gate
           difference comes from mean-value: gate mean ~ rate at gate centre).
           k in 1..8; shallow band (DEC<DEC_SH) excluded (DE invalid near source and the
           gate-centre approximation breaks inside gate 0); CSF excluded at sampling.
           Scale-free RELATIVE residual so it is valid across all 8 decades."""
        if a.lambda_pinn <= 0 or P.get("n", 0) == 0:
            return torch.zeros((), device=dev)
        if a.cw:
            # V10 STEADY-STATE residual on the time-INTEGRATED (CW) field: -D lap(Phi) + mu_a Phi = 0,
            # source-free away from the near-source shallow band. 7-point spatial stencil, ONE fixed
            # time input, NO dPhi/dt. Relative form (r = 10^(lg-lc)) to stay overflow-safe.
            m = torch.randint(0, P["n"], (bsz,), device=dev)
            gt_cw = integ_t(P["GT10"][m][:, :n_gate_train]).squeeze(1)     # CW target at collocation
            band = (P["LFM"][m].squeeze(1) - gt_cw) >= pinn_dec_min        # skip near-source shallow
            if not bool(band.any()):
                return torch.zeros((), device=dev)
            praw = gather_raw(P["RAW"], m).reshape(-1, FEATDIM) if use_pyr else None
            pxn = P["XN"][m].reshape(-1, 3); popt = P["OPT"][m].reshape(-1, 4); psf = P["SF"][m].reshape(-1, 2)
            pl = P["LIGHT"][m].unsqueeze(1).expand(-1, 7, -1).reshape(-1, LIGHT_DIM)
            tcw = t_cw.expand(pxn.shape[0], 1)
            ppa = P["PATH"][m].reshape(-1, PATHDIM).float() if PATHDIM else None
            lg = model.forward_feats(praw, torch.cat([pxn, tcw], 1), popt, pl, psf, ppa).view(bsz, 7)
            lc = lg[:, 0:1]
            r = torch.pow(10.0, (lg - lc).clamp(-LOGD_CLAMP, LOGD_CLAMP))  # (bsz,7); r[:,0]==1
            h2 = (P["H"][m].squeeze(1) ** 2) if "H" in P else (PINN_H ** 2)
            lap_c = (r[:, 1:].sum(1) - 6.0) / h2                           # lap(Phi)/Phi_c
            Dc = P["D"][m].squeeze(1); mua = P["MUA"][m].squeeze(1)
            num = -Dc * lap_c + mua                                        # steady residual / Phi_c
            den = (Dc * lap_c).abs() + mua + 1e-12
            return ((num / den) ** 2)[band].mean()
        m = torch.randint(0, P["n"], (bsz,), device=dev)
        k = torch.randint(1, n_gate_train - 1, (bsz,), device=dev)    # 1..8 (1..7 if gate 9 dropped)
        gt_c = P["GT10"][m].gather(1, k.view(-1, 1)).squeeze(1)
        band = (P["LFM"][m].squeeze(1) - gt_c) >= pinn_dec_min         # skip near-source shallow
        if not bool(band.any()):
            return torch.zeros((), device=dev)
        praw = gather_raw(P["RAW"], m).reshape(-1, FEATDIM) if use_pyr else None
        pxn = P["XN"][m].reshape(-1, 3); popt = P["OPT"][m].reshape(-1, 4); psf = P["SF"][m].reshape(-1, 2)
        pl = P["LIGHT"][m].unsqueeze(1).expand(-1, 7, -1).reshape(-1, LIGHT_DIM)
        tk = tenc[k].view(-1, 1, 1).expand(-1, 7, 1).reshape(-1, 1)
        ppa = P["PATH"][m].reshape(-1, PATHDIM).float() if PATHDIM else None
        lg = model.forward_feats(praw, torch.cat([pxn, tk], 1), popt, pl, psf, ppa).view(bsz, 7)
        craw = praw.view(bsz, 7, -1)[:, 0] if use_pyr else None
        cxn = pxn.view(bsz, 7, 3)[:, 0]; copt = popt.view(bsz, 7, 4)[:, 0]
        csf_ = psf.view(bsz, 7, 2)[:, 0]; cl = P["LIGHT"][m]
        cpa = ppa.view(bsz, 7, PATHDIM)[:, 0] if PATHDIM else None
        lgm = model.forward_feats(craw, torch.cat([cxn, tenc[k-1].view(-1,1)], 1), copt, cl, csf_, cpa).squeeze(1)
        lgp = model.forward_feats(craw, torch.cat([cxn, tenc[k+1].view(-1,1)], 1), copt, cl, csf_, cpa).squeeze(1)
        # NUMERICALLY STABLE form: the residual is RELATIVE, so divide it AND its scale by the
        # centre value Phi_c and work with RATIOS r = 10^(logPhi - logPhi_c). Mathematically
        # identical, but never forms 10^(large) -> no fp32 overflow. (The naive version above
        # NaN'd at ~ep60: 10^logPhi overflowed fp32 (max 3.4e38) and the Laplacian did inf-inf.)
        lc = lg[:, 0:1]
        r = torch.pow(10.0, (lg - lc).clamp(-LOGD_CLAMP, LOGD_CLAMP))  # (bsz,7); r[:,0]==1
        h2 = (P["H"][m].squeeze(1) ** 2) if "H" in P else (PINN_H ** 2)   # per-point step^2 (jittered FD)
        lap_c = (r[:, 1:].sum(1) - 6.0) / h2                           # lap(Phi)/Phi_c
        rm = torch.pow(10.0, (lgm - lc.squeeze(1)).clamp(-LOGD_CLAMP, LOGD_CLAMP))
        rp = torch.pow(10.0, (lgp - lc.squeeze(1)).clamp(-LOGD_CLAMP, LOGD_CLAMP))
        Dc = P["D"][m].squeeze(1); mua = P["MUA"][m].squeeze(1)
        dpdt_c = (rp - rm) / TWO_C_DT                                  # (1/c)dPhi/dt / Phi_c
        # PLAN A -- steady-state in the deep tail. The probe showed |dPhi/dt| is ~8x larger in the
        # 5e7-blind zone than in the resolved deep bands: the model uses a spurious time-derivative
        # to cancel a steady-state (monotone-decay) violation, which is how above-floor bumps /
        # checkerboard survive (v10 steady-state forbade them; the v11 time term is the escape). In
        # the deep CENSORED band the tail is quasi-static, so drop the time term there -> the large
        # spatial Laplacian of any bump is no longer cancellable and gets penalised, forcing decay.
        if DEEP_STEADY:
            deep = (P["LFM"][m].squeeze(1) - gt_c) >= DEC_DEEP
            dpdt_c = torch.where(deep, torch.zeros_like(dpdt_c), dpdt_c)
        num = dpdt_c - Dc * lap_c + mua
        den = dpdt_c.abs() + (Dc * lap_c).abs() + mua + 1e-12
        loss_pde = ((num / den) ** 2)[band].mean()
        # PLAN B -- explicit radial monotonicity: in the deep band fluence must DECREASE away from
        # the source (dPhi/dr <= 0). r[1..6] are neighbours at +-h along x,y,z, so the relative
        # gradient is (r[+]-r[-])/(2h); dotted with the stored source->point unit direction gives
        # dPhi/dr / Phi_c. relu penalises any INCREASE. rhat comes from the path-integral geometry.
        if RADIAL_MONO > 0 and "RHAT" in P:
            grad = torch.stack([(r[:,1]-r[:,2]), (r[:,3]-r[:,4]), (r[:,5]-r[:,6])], 1) / (2.0*torch.sqrt(h2).unsqueeze(1))
            dphidr = (grad * P["RHAT"][m]).sum(1)                      # dPhi/dr / Phi_c
            deepb = (P["LFM"][m].squeeze(1) - gt_c) >= DEC_MID         # apply from mid-deep outward
            sel_b = band & deepb
            if bool(sel_b.any()):
                loss_pde = loss_pde + RADIAL_MONO * (torch.relu(dphidr)[sel_b] ** 2).mean()
        return loss_pde

    tr = torch.nonzero(SP == 0).squeeze(1); va = torch.nonzero(SP == 1).squeeze(1)
    opt_ = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=EPOCHS, eta_min=1e-5)

    # ---- full-state checkpoint / resume (survive an OOM kill or any interruption) ----
    STATE = os.path.join(CKPT, f"state_{tag}.pt")
    start_ep, best, hist = 1, float("inf"), {"train": [], "val": []}
    if a.resume and os.path.isfile(STATE):
        st = torch.load(STATE, map_location=dev)
        model.load_state_dict(st["model"]); opt_.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_ep = st["epoch"] + 1; best = st["best"]; hist = st["hist"]
        print(f"[{tag}] RESUMED from epoch {st['epoch']} (best {best:.4f})", flush=True)

    def save_state(ep):
        # PID-unique tmp: a shared "*.tmp" lets two concurrent trainers (an auto-restart that
        # overlapped a still-live predecessor) clobber each other's temp file and leave state.pt
        # half-written and unreadable. The encoder lost its resume state to exactly this at ~epoch
        # 160. os.replace on the same filesystem is atomic; the TEMP NAME is what must not collide.
        tmp = f"{STATE}.tmp.{os.getpid()}"
        torch.save({"model": model.state_dict(), "opt": opt_.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best, "hist": hist}, tmp)
        os.replace(tmp, STATE)

    def ev(idx):
        """VAL objective CONSISTENT with training: MSE on the reachable set + one-sided
        hinge on the CENSORED set. (A plain MSE vs gt=floor would punish the model for
        predicting BELOW the floor -- which the censored likelihood explicitly allows and
        which is physically correct -- and would make model selection fight the hinge.)"""
        model.eval(); tot = nb = 0
        with torch.no_grad():
            for j in range(0, idx.numel(), BATCH):
                b = idx[j:j+BATCH]; xyzt, gt, dec, _ = batch_sample_t(b)
                p = model.forward_feats(rawb(b), xyzt, OPT[b], LIGHT[b], SF[b], pathb(b))
                cen = dec >= DEC_DEEP
                # Match validation to the selected two-sided or censored training objective.
                err = (p - gt) if a.floor_mode == "zero" else torch.where(
                    cen, torch.relu(p - gt + a.hinge_offset), p - gt)
                tot += float((err ** 2).mean()); nb += 1
        return tot / max(nb, 1)

    t1 = time.time()
    for ep in range(start_ep, EPOCHS + 1):
        model.train(); perm = tr[torch.randperm(tr.numel(), device=dev)]; tl = nb = 0
        for j in range(0, perm.numel(), BATCH):
            b = perm[j:j+BATCH]; xyzt, gt, dec, k = batch_sample_t(b)
            pred = model.forward_feats(rawb(b), xyzt, OPT[b], LIGHT[b], SF[b], pathb(b))
            w = weights(dec, b)
            if GATE_W is not None:
                w = w * GATE_W[k].view(-1, 1)                 # per-gate rebalancing (renormalised, mean 1)
            loss = (w * (pred - gt) ** 2).sum() / w.sum().clamp_min(1e-6)
            if a.lambda_lin > 0:
                sh = dec < lin_dec_max
                if sh.any():
                    # The clamp is what keeps this term finite, and +-2 was far too loose:
                    # (10^2-1)^2 = 9801, so a single saturating shallow point contributes ~2940
                    # after lambda_lin, and one in 10k such points moves the batch mean by 0.3 --
                    # exactly the size of the training spikes seen at ep 40/190/220/250 in EVERY
                    # run, including the ones with lambda_pinn=0 (so the PINN was never the cause).
                    # The shallow band is the high-SNR region; being a full decade off there is
                    # already a pathological output that should not dominate the gradient.
                    d = (pred - gt).clamp(-a.lin_clamp, a.lin_clamp)
                    loss = loss + a.lambda_lin * ((torch.pow(10.0, d) - 1.0)[sh] ** 2).mean()
            if a.lambda_floor > 0 and a.floor_mode == "zero":
                # DEPLOYMENT variant: the dark zone is declared to be fluence 0. gt is already the
                # floor there (the MC was clipped), so a SYMMETRIC MSE pins the prediction to the
                # floor: the model neither invents light (leak%) nor extrapolates below it. This
                # trades away Level-2 entirely -- deliberately -- for a sharp, well-defined
                # reachable boundary, which is all the downstream dose optimisation needs.
                cen = dec >= DEC_DEEP
                if cen.any():
                    loss = loss + a.lambda_floor * (((pred - gt)[cen]) ** 2).mean()
            elif a.lambda_floor > 0:                 # one-sided hinge on the censored (floor) set
                cen = dec >= DEC_DEEP                # gt == floor there (clipped) -> hinge vs gt
                if cen.any():
                    # threshold = floor - hinge_offset (gt IS the floor here). offset=0 is the honest
                    # detection-limit hinge; offset>0 is the production sharp-boundary lever.
                    over = torch.relu(pred - gt + a.hinge_offset)[cen]   # penalise pred ABOVE floor-offset
                    loss = loss + a.lambda_floor * (over ** 2).mean()
            if a.lambda_ceil > 0:                    # PHYSICAL CEILING: pred cannot exceed the peak
                ceil = torch.relu(pred - (LFM[b] + a.ceil_offset))
                loss = loss + a.lambda_ceil * (ceil ** 2).mean()
            if a.lambda_mono > 0:
                # dense radial monotonicity/decay in the deep, non-CSF dark zone, via autograd
                csfm = CSFB[b].squeeze(-1) > 0.5
                darkm = (dec.squeeze(-1) >= a.mono_dec_min) & (~csfm)
                if bool(darkm.any()):
                    xn = XN[b].detach().requires_grad_(True)
                    pm = model.forward_feats(rawb(b), torch.cat([xn, xyzt[:, 3:4]], 1),
                                             OPT[b], LIGHT[b], SF[b], pathb(b))
                    gxn, = torch.autograd.grad(pm.sum(), xn, create_graph=True)  # dlogPhi/dxn
                    _L = LIGHT_SRC if LIGHT_SRC is not None else LIGHT
                    cen_xn = CENTER[b] if CENTER is not None else (2.0 * _L[b][:, 0:3] - 1.0)
                    du = XN[b] - cen_xn                                          # centre->point, XN space
                    un = du / du.norm(dim=1, keepdim=True).clamp_min(1e-6)
                    # physical-unit radial derivative: divide by |d_phys|/|d_xn| (anisotropic MNI scale)
                    half = (MONO_VS.to(du) - 1.0) / 2.0
                    scale = (du * half).norm(dim=1) / du.norm(dim=1).clamp_min(1e-6)   # mm per XN-unit
                    dphidr = (gxn * un).sum(1) / scale                           # d log10Phi / d r_phys
                    if a.mono_rate > 0:
                        pth = pathb(b)
                        mu = pth[:, 2] / (pth[:, 5] * 100.0).clamp_min(1.0)      # mu_eff = tau_eff/r, /mm
                        viol = torch.relu(dphidr + a.mono_rate * mu / math.log(10.0))
                    else:
                        viol = torch.relu(dphidr)                                # sign only
                    loss = loss + a.lambda_mono * (viol[darkm] ** 2).mean()
            if a.lambda_pinn > 0:
                ramp = min(1.0, ep / max(PINN_WARMUP, 1))          # PINN warm-up
                loss = loss + a.lambda_pinn * ramp * pinn_loss()
            opt_.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # PINN gradient stabiliser
            opt_.step(); tl += loss.item(); nb += 1
        sched.step(); vl = ev(va)
        hist["train"].append(tl/nb); hist["val"].append(vl)
        CFG = {"mlp_width": a.mlp_width, "mlp_extra": a.mlp_extra,
               "mlp_residual": bool(a.mlp_residual),
               "num_freq": a.num_freq, "num_freq_t": a.num_freq_t, "activation": a.activation,
               "pe_freqs": pe_freqs, "pe_freqs_t": pe_freqs_t,
               # zero_feats is what build_model reads back; no_pe/zero_light are kept only so the
               # cfg records which CLI produced it (the specs alone do not say "all" vs "0:10").
               "zero_feats": list(zero_feats), "no_pe": bool(a.no_pe), "zero_light": a.zero_light,
               "drop_feats": list(drop_feats), "drop_light": a.drop_light,
               "path_dim": PATHDIM, "path_nseg": a.path_nseg, "src_ref": a.src_ref, "path_tri": a.path_trilinear,
               "lin_clamp": a.lin_clamp, "seed": a.seed,
               "floor_mode": a.floor_mode, "hinge_offset": a.hinge_offset,
               "lambda_lin": a.lambda_lin, "lambda_floor": a.lambda_floor, "lambda_pinn": a.lambda_pinn, "pinn_dec_min": pinn_dec_min, "lin_dec_max": lin_dec_max, "dec_sh": DEC_SH, "dec_mid": DEC_MID,
               "shallow_boost": SHALLOW_BOOST, "csf_boost": CSF_BOOST,
               "lambda_ceil": a.lambda_ceil, "ceil_offset": a.ceil_offset,
               "lambda_mono": a.lambda_mono, "mono_dec_min": a.mono_dec_min, "mono_rate": a.mono_rate, "mono_center": a.mono_center,
               "n_gate_train": n_gate_train,
               # NOTE: reach/bright_frac are BUFFER-BUILD parameters (sample_bright runs only in the
               # precompute branch). On a --load-mm run they record this process's env, NOT the
               # buffer's actual sampling -> they read 0.0/0.5 and are MEANINGLESS here. The buffer
               # path below is the real provenance of the sampling mix.
               "reach_dec": REACH_DEC, "reach_frac": REACH_FRAC,
               "bright_frac": BRIGHT_FRAC, "pinn_ncap": PINN_NCAP,
               "buffer": a.load_mm or a.load_buffers or "live",
               "cw": bool(a.cw), "dec_deep": DEC_DEEP, "no_time": bool(a.no_time),
               "use_pyramid": use_pyr, "n_step": N_STEP}
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "epoch": ep, "val": vl, "history": hist,
                        "cfg": CFG}, os.path.join(CKPT, f"inr_{tag}.pt"))
        # PERIODIC MODEL SNAPSHOTS -- best-val is NOT a valid selector for Level-1: the val objective
        # is scored on the SAME sparse sampled points the model is fit on (0.05% of voxels), so it is
        # blind to the near-source overshoot / sub-floor checkerboard that live in the unsupervised
        # GAPS. Measured on v13ceil: val kept improving (ep110 -> best ep134) while reachable R2hB
        # collapsed +0.35 -> -2.10 and overshoot grew 0.02% -> 0.25%. Keep snapshots so the hero can
        # be picked post-hoc on the real Level-1 instrument (eval_level1_reach.py, VAL only).
        if a.snap_every > 0 and (ep % a.snap_every == 0 or ep == EPOCHS):
            torch.save({"model": model.state_dict(), "epoch": ep, "val": vl, "cfg": CFG},
                       os.path.join(CKPT, f"snap_{tag}_ep{ep:03d}.pt"))
        if ep % a.ckpt_every == 0 or ep == EPOCHS:
            save_state(ep)                       # resumable: survives an OOM kill
        if ep % 10 == 0 or ep == 1:
            print(f"  [{tag}] ep {ep}/{EPOCHS} tr {tl/nb:.4f} val {vl:.4f} best {best:.4f} "
                  f"{time.time()-t1:.0f}s", flush=True)
    json.dump({"tag": tag, "best_val": best, "history": hist}, open(os.path.join(CKPT, f"metrics_{tag}.json"), "w"))
    print(f"[{tag}] DONE best_val={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
