"""The surrogate itself: load a checkpoint, query fluence.

Output is always **log10 of fluence**, in the same arbitrary units as the Monte-Carlo reference the
model was trained against (mcx normalised output, 5e7 photons). Ratios between points are the
meaningful quantity; the absolute offset is not calibrated to milliwatts.
"""
import os

import torch

from . import features as FT
from .inr_model import build_model
from .features import (N_STEP, T_ENC, path_features, sample_raw, sample_volume, src_features,
                       t_code, xyz_to_norm)
from . import optical_config as OC

# The finalised V18 hero3 recipe. Asserted on load rather than trusted: this project has twice
# published a number computed from a checkpoint that was not the one named, and a cfg assertion is
# the cheapest place to catch it.
HERO3 = dict(src_ref="entry", num_freq=4, num_freq_t=3, use_pyramid=True,
             drop_feats=["optical", "light:6:10"])


class Predictor:
    def __init__(self, ckpt_path, device, assert_hero3=True, verbose=True):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = cfg = ck["cfg"]
        if assert_hero3:
            for k, want in HERO3.items():
                got = cfg.get(k)
                if list(got) != list(want) if isinstance(want, list) else got != want:
                    raise SystemExit(
                        f"{os.path.basename(ckpt_path)} is not a V18 hero3 checkpoint: "
                        f"cfg[{k!r}] = {got!r}, expected {want!r}. Pass assert_hero3=False to load "
                        f"it anyway -- but then check src_ref yourself, because feeding an "
                        f"entry-trained model source-referenced features fails silently.")
        self.model = build_model(cfg).to(device).eval()
        sd = ck.get("model", ck.get("state_dict", ck))
        missing = self.model.load_state_dict(sd, strict=True)
        self.device = device
        self.src_ref = cfg.get("src_ref", "source")
        self.path_dim = cfg.get("path_dim", 0)
        self.seed = cfg.get("seed")
        if verbose:
            n = sum(p.numel() for p in self.model.parameters())
            in_dim = self.model.mlp[0].in_features
            print(f"[model] {os.path.basename(ckpt_path)}  seed={self.seed}  "
                  f"in_dim={in_dim}  params={n/1e6:.2f}M  src_ref={self.src_ref}")

    # ---------------------------------------------------------------------------------------- #
    @torch.no_grad()
    def _features(self, scene, xyz):
        """Everything except the time code, which is the only thing that varies per gate."""
        vs = scene.vol_shape
        xn = xyz_to_norm(xyz, vs)
        opt = OC.normalize_points(sample_volume(scene.prop, xyz, vs))
        sf = src_features(xyz, scene.srcpos, scene.srcdir)
        # cfg["path_tri"] MUST be forwarded. Leaving `trilinear` at its False default evaluates a
        # trilinearly-trained model with nearest-voxel gathers -- a silent train/eval mismatch.
        pf = (path_features(xyz, scene.srcpos, scene.prop, vs, self.cfg.get("path_nseg", 0),
                            self.cfg.get("path_tri", False)) if self.path_dim else None)
        raw = (sample_raw(xn, scene.pyramid).float()
               if self.cfg.get("use_pyramid", True) else None)
        if self.cfg.get("use_pyramid", True) and scene.pyramid is None:
            raise RuntimeError("this checkpoint needs the FM pyramid; construct the Scene with "
                               "cache_dir=... so it can be extracted")
        return xn, opt, sf, pf, raw

    @torch.no_grad()
    def at_points(self, scene, xyz, t_ns=None, chunk=200_000):
        """(N,3) voxel coords -> (N, G) log10 fluence.

        `t_ns=None` gives all 10 training gates. A float or list gives those times in nanoseconds,
        interpolated by the network -- the gate grid is not a constraint on where you may ask.
        """
        xyz = torch.as_tensor(xyz, dtype=torch.float32, device=self.device).reshape(-1, 3)
        codes = list(T_ENC) if t_ns is None else \
            [t_code(v) for v in ([t_ns] if isinstance(t_ns, (int, float)) else t_ns)]
        out = torch.empty(xyz.shape[0], len(codes), device=self.device)
        for i in range(0, xyz.shape[0], chunk):
            x = xyz[i:i + chunk]
            xn, opt, sf, pf, raw = self._features(scene, x)
            for k, tt in enumerate(codes):
                xt = torch.cat([xn, torch.full((xn.shape[0], 1), float(tt), device=self.device)], 1)
                out[i:i + chunk, k] = self.model.forward_feats(
                    raw, xt, opt, scene.light, sf, pf).squeeze(1)
        return out

    @torch.no_grad()
    def volume(self, scene, t_ns=None, chunk=200_000, fill=None):
        """Whole head -> (X, Y, Z, G) log10 fluence, air left at `fill`.

        Only tissue voxels are evaluated. Air is not merely uninteresting: it is outside the
        training distribution entirely, so values there would be extrapolation presented as result.
        `fill` defaults to the minimum predicted tissue value.
        """
        xyz = scene.tissue_coords()
        vals = self.at_points(scene, xyz, t_ns=t_ns, chunk=chunk)
        X, Y, Z = [int(v) for v in scene.vol_shape]
        G = vals.shape[1]
        vol = torch.full((X, Y, Z, G), float(vals.min()) if fill is None else float(fill),
                         device=self.device)
        idx = xyz.long()
        vol[idx[:, 0], idx[:, 1], idx[:, 2], :] = vals
        return vol


def load_ensemble(weights_dir, device, seeds=(0, 1, 2), **kw):
    """The three seeds. Report the spread, not one seed silently chosen.

    Which seed is 'best' cannot be decided on the data you then report -- picking by validation OR
    by test score both bias the number. Average, or state the spread.
    """
    return [Predictor(os.path.join(weights_dir, f"phomineuro_s{s}.pt"), device, **kw)
            for s in seeds]
