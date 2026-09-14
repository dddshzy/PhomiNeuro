#!/usr/bin/env python3
"""
v2 INR dataset: 810 nm, head-only, light-source-ANGLE conditioned.

Differences vs inr_dataset (v1):
  - scenes come from the orientation-standardized volumes (dataset_v2_head810),
    the v2 pyramids (extracted_pyramids_v2), and the angle-grid fluence
    (sim_v2_810/data/wl810, named fluence_{head}_F810_A{..}_B{..}_r5.mat).
  - domain is always 0 (head); wavelength is fixed (810) so it is constant.
  - the light vector is EXTENDED to 10-D:
        [srcpos_norm(3), srcdir(3), sinA, cosA, sinB, cosB]
    The explicit (A,B) sin/cos give a smooth, low-dim handle for the downstream
    differentiable illumination optimization over (A,B).

Geometry helpers (grid_sample, point sampling, fixed-range optical normalization)
are reused unchanged from inr_dataset.
"""
import os
import re
import sys
import glob
import json
import math
import torch
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
import inr_dataset as D1            # reuse helpers
import v2_manifest as M
import optical_config as OC

LIGHT_DIM = 10
FLU_FLOOR_DECADES = D1.FLU_FLOOR_DECADES
FLU_EPS = D1.FLU_EPS


def discover_scenes_v2():
    """One scene per (head, A, B). geom_key = the head pyramid (shared across angles)."""
    scenes = []
    for pyr in sorted(glob.glob(os.path.join(M.PYRAMID_V2, "*_pyramid.pt"))):
        base = os.path.basename(pyr)
        m = re.match(r"(.+)_copmri_withHermiteF(\d+)_pyramid\.pt$", base)
        if not m:
            continue
        head, wl = m.group(1), m.group(2)
        prop = M.std_mat(head)
        if not os.path.isfile(prop):
            continue
        for flu in sorted(glob.glob(os.path.join(M.SIM_V2_DATA, f"fluence_{head}_F{wl}_*.mat"))):
            fm = re.match(rf"fluence_{re.escape(head)}_F{wl}_(.+)\.mat$", os.path.basename(flu))
            if not fm:
                continue
            tag = fm.group(1)
            meta = os.path.join(M.SIM_V2_DATA, f"meta_{head}_F{wl}_{tag}.json")
            if os.path.isfile(meta):
                scenes.append(dict(head=head, sample=head, wl=wl, tag=tag, geom_key=pyr,
                                   pyramid=pyr, prop=prop, fluence=flu, meta=meta))
    return scenes


def group_by_geometry(scenes):
    return D1.group_by_geometry(scenes)


def _light_vec(meta, X, Y, Z):
    srcpos = torch.tensor(meta["srcpos"], dtype=torch.float32)
    srcdir = torch.tensor(meta["srcdir"], dtype=torch.float32)
    srcpos_norm = srcpos / (torch.tensor([X, Y, Z], dtype=torch.float32) - 1)
    A = math.radians(float(meta["A_deg"])); B = math.radians(float(meta["B_deg"]))
    ang = torch.tensor([math.sin(A), math.cos(A), math.sin(B), math.cos(B)], dtype=torch.float32)
    return torch.cat([srcpos_norm, srcdir, ang])      # (10,)


class SceneStoreV2:
    """Two-level cache (geometry once, illumination swapped) — head-only, 10-D light."""
    def __init__(self, device):
        self.device = device
        self._gk = None; self._geo = None; self._ik = None; self._il = None

    def _load_geometry(self, scene):
        prop = sio.loadmat(scene["prop"])[M.PROP_KEY].astype(np.float32)
        prop = np.transpose(prop, (3, 0, 1, 2))
        prop_t = torch.from_numpy(prop).unsqueeze(0).to(self.device)
        X, Y, Z = prop_t.shape[-3:]
        tissue_np = prop[1] > 0
        valid_idx = torch.nonzero(prop_t[0, 1] > 0, as_tuple=False)
        pyr = torch.load(scene["pyramid"], map_location="cpu")["pyramid"]
        pyramid = [p.to(self.device) for p in pyr]
        return dict(prop=prop_t, pyramid=pyramid, valid_idx=valid_idx, tissue_np=tissue_np,
                    vol_shape=torch.tensor([X, Y, Z], device=self.device))

    def _load_illumination(self, scene, geo):
        X, Y, Z = [int(v) for v in geo["vol_shape"].tolist()]
        flu = sio.loadmat(scene["fluence"])["fluence"].astype(np.float32)
        fmax = float(flu.max())
        floor = max(FLU_EPS, fmax * (10.0 ** (-FLU_FLOOR_DECADES)))
        flu = np.clip(flu, floor, None)
        logflu = np.log10(flu)
        logflu_t = torch.from_numpy(logflu).unsqueeze(0).unsqueeze(0).to(self.device)
        bright_np = geo["tissue_np"] & (logflu > np.log10(floor) + 1e-4)
        bright_idx = torch.from_numpy(np.argwhere(bright_np)).to(self.device)
        with open(scene["meta"]) as f:
            meta = json.load(f)
        light = _light_vec(meta, X, Y, Z).to(self.device)
        return dict(logflu=logflu_t, bright_idx=bright_idx, light=light)

    def get(self, scene):
        if scene["geom_key"] != self._gk:
            self._geo = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._geo = self._load_geometry(scene)
            self._gk = scene["geom_key"]; self._ik = None
        if scene["fluence"] != self._ik:
            self._il = self._load_illumination(scene, self._geo)
            self._ik = scene["fluence"]
        return {**self._geo, **self._il, "vol_shape": self._geo["vol_shape"]}


# reuse the v1 stratified sampler (identical semantics)
sample_points = D1.sample_points


if __name__ == "__main__":
    sc = discover_scenes_v2()
    g = group_by_geometry(sc)
    print(f"v2 scenes: {len(sc)} across {len(g)} head geometries (light_dim={LIGHT_DIM})")
    if sc:
        print("example:", {k: (os.path.basename(v) if isinstance(v, str) else v)
                           for k, v in sc[0].items()})
