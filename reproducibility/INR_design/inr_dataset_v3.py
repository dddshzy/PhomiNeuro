#!/usr/bin/env python3
"""
v3 INR dataset: base + rotation-aug + angle-perturbation-aug scenes, with
source-relative per-point physics features (r, cos theta).

Scenes
  - base : 16 scb heads (PYRAMID_V2 / DATASET_V2 / SIM_V2_DATA), tag like A45_Bm60_r5
  - rot  : 8 rotated heads "<h>rot" (PYRAMID_V3 / DATASET_V3_AUG / SIM_V3_ROT)
  - pert : perturbed angles on base heads (PYRAMID_V2 / DATASET_V2 / SIM_V3_PERT),
           tag like pert0123_r5 ; meta carries the true A,B / srcpos / srcdir

Split (held-out sets identical in spirit to v2; aug only enters TRAIN):
  - val_head : base scenes of scb11
  - val_ang  : base scenes whose angle tag is in the held-out 15% (head != scb11)
  - train    : everything else (base train + ALL rot + ALL pert)

Source-relative features per point (physics-informed; encode the exp(-mu r)/r form):
  r_norm = |x - srcpos| / 128 ;  cos_theta = (x - srcpos)·srcdir / |x - srcpos|
"""
import os, re, sys, glob, json, math
import numpy as np
import scipy.io as sio
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
import inr_dataset as D1
import inr_dataset_v2 as D2
import v2_manifest as M
import optical_config as OC

LIGHT_DIM = 10
SRCFEAT_DIM = 2
R_SCALE = 128.0
FLU_FLOOR_DECADES = D1.FLU_FLOOR_DECADES
FLU_EPS = D1.FLU_EPS
SPLIT_SEED = 0
ANGLE_HOLDOUT_FRAC = 0.15


def _held_tags(base_tags):
    import random
    tags = sorted(set(base_tags))
    rng = random.Random(SPLIT_SEED); rng.shuffle(tags)
    return set(tags[:max(1, round(ANGLE_HOLDOUT_FRAC * len(tags)))])


def discover_scenes_v3():
    scenes = []
    # base
    for pyr in sorted(glob.glob(os.path.join(M.PYRAMID_V2, "*_pyramid.pt"))):
        m = re.match(r"(.+)_copmri_withHermiteF(\d+)_pyramid\.pt$", os.path.basename(pyr))
        if not m:
            continue
        head, wl = m.group(1), m.group(2)
        prop = M.std_mat(head)
        if not os.path.isfile(prop):
            continue
        for flu in sorted(glob.glob(os.path.join(M.SIM_V2_DATA, f"fluence_{head}_F{wl}_*.mat"))):
            tag = re.match(rf"fluence_{re.escape(head)}_F{wl}_(.+)\.mat$", os.path.basename(flu)).group(1)
            meta = os.path.join(M.SIM_V2_DATA, f"meta_{head}_F{wl}_{tag}.json")
            if os.path.isfile(meta):
                scenes.append(dict(kind="base", head=head, wl=wl, tag=tag, geom_key=pyr,
                                   pyramid=pyr, prop=prop, fluence=flu, meta=meta))
    # rotation-aug
    for pyr in sorted(glob.glob(os.path.join(M.PYRAMID_V3, "*_pyramid.pt"))):
        m = re.match(r"(.+)_copmri_withHermiteF(\d+)_pyramid\.pt$", os.path.basename(pyr))
        if not m:
            continue
        head, wl = m.group(1), m.group(2)
        base_head = re.sub(r"rot\d+$", "", head)   # 'bw03rot1' -> 'bw03'
        if base_head in M.HELDOUT_HEADS:        # never leak a val/test head via aug
            continue
        prop = M.std_mat_v3(head)
        if not os.path.isfile(prop):
            continue
        for flu in sorted(glob.glob(os.path.join(M.SIM_V3_ROT, f"fluence_{head}_F{wl}_*.mat"))):
            tag = re.match(rf"fluence_{re.escape(head)}_F{wl}_(.+)\.mat$", os.path.basename(flu)).group(1)
            meta = os.path.join(M.SIM_V3_ROT, f"meta_{head}_F{wl}_{tag}.json")
            if os.path.isfile(meta):
                scenes.append(dict(kind="rot", head=head, wl=wl, tag=tag, geom_key=pyr,
                                   pyramid=pyr, prop=prop, fluence=flu, meta=meta))
    # angle-perturbation-aug (base head geometry, perturbed angle)
    for flu in sorted(glob.glob(os.path.join(M.SIM_V3_PERT, f"fluence_*_F{M.WL}_*.mat"))):
        b = os.path.basename(flu)
        head = b.split(f"_F{M.WL}_")[0].replace("fluence_", "")
        if head in M.HELDOUT_HEADS:             # never leak a held-out head via pert
            continue
        tag = b.split(f"_F{M.WL}_")[1].replace(".mat", "")
        meta = os.path.join(M.SIM_V3_PERT, f"meta_{head}_F{M.WL}_{tag}.json")
        pyr = os.path.join(M.PYRAMID_V2, f"{head}_copmri_withHermiteF{M.WL}_pyramid.pt")
        if os.path.isfile(meta) and os.path.isfile(pyr):
            scenes.append(dict(kind="pert", head=head, wl=M.WL, tag=tag, geom_key=pyr,
                               pyramid=pyr, prop=M.std_mat(head), fluence=flu, meta=meta))
    return scenes


def split_labels(scenes):
    base_tags = [s["tag"] for s in scenes if s["kind"] == "base"]
    held = _held_tags(base_tags)
    labels = []
    for s in scenes:
        if s["kind"] == "base" and s["head"] in M.TEST_HEADS:
            labels.append("test_head")               # LOCKED subject-level test
        elif s["kind"] == "base" and s["head"] in M.VAL_HEADS:
            labels.append("val_head")                # subject-level validation (selection)
        elif s["kind"] == "base" and s["tag"] in held and s["head"] not in M.HELDOUT_HEADS:
            labels.append("val_ang")                 # seen-subject unseen-angle (diagnostic)
        else:
            labels.append("train")
    return labels, held


def group_by_geometry(scenes):
    return D1.group_by_geometry(scenes)


def src_features(xyz, srcpos, srcdir):
    """(N,3) voxel coords -> (N,2) [r_norm, cos_theta] relative to the source."""
    u = xyz - srcpos.view(1, 3)
    r = u.norm(dim=1, keepdim=True).clamp_min(1e-3)
    cos = (u * srcdir.view(1, 3)).sum(1, keepdim=True) / r
    return torch.cat([r / R_SCALE, cos], dim=1)


class SceneStoreV3:
    """Geometry cached once; illumination swapped. Light=10-D; exposes srcpos/srcdir."""
    def __init__(self, device):
        self.device = device; self._gk = None; self._geo = None; self._ik = None; self._il = None

    def _load_geometry(self, scene):
        prop = sio.loadmat(scene["prop"])[M.PROP_KEY].astype(np.float32)
        prop = np.transpose(prop, (3, 0, 1, 2))
        prop_t = torch.from_numpy(prop).unsqueeze(0).to(self.device)
        X, Y, Z = prop_t.shape[-3:]
        tissue_np = prop[1] > 0
        valid_idx = torch.nonzero(prop_t[0, 1] > 0, as_tuple=False)
        # CSF mask from the unique F810 signature (mua~0.0026, mus~0.091): the only tissue
        # with mua<=0.005 AND 0<mus<=0.5 (fat has mua<0.005 but mus~9.9, so excluded).
        csf = (prop_t[0, 0] <= 0.005) & (prop_t[0, 1] > 1e-6) & (prop_t[0, 1] <= 0.5)  # (X,Y,Z) bool
        pyr = torch.load(scene["pyramid"], map_location="cpu")["pyramid"]
        pyramid = [p.to(self.device) for p in pyr]
        return dict(prop=prop_t, pyramid=pyramid, valid_idx=valid_idx, tissue_np=tissue_np,
                    csf=csf, vol_shape=torch.tensor([X, Y, Z], device=self.device))

    def _load_illumination(self, scene, geo):
        X, Y, Z = [int(v) for v in geo["vol_shape"].tolist()]
        flu = sio.loadmat(scene["fluence"])["fluence"].astype(np.float32)
        # incidence point = fluence-max voxel (for V10 distance-stratified PINN loss)
        peakpos = torch.tensor(np.unravel_index(int(flu.argmax()), flu.shape),
                               dtype=torch.float32, device=self.device)
        fmax = float(flu.max()); floor = max(FLU_EPS, fmax * (10.0 ** (-FLU_FLOOR_DECADES)))
        flu = np.clip(flu, floor, None); logflu = np.log10(flu)
        logflu_t = torch.from_numpy(logflu).unsqueeze(0).unsqueeze(0).to(self.device)
        bright_np = geo["tissue_np"] & (logflu > np.log10(floor) + 1e-4)
        bright_idx = torch.from_numpy(np.argwhere(bright_np)).to(self.device)
        meta = json.load(open(scene["meta"]))
        srcpos = torch.tensor(meta["srcpos"], dtype=torch.float32)
        srcdir = torch.tensor(meta["srcdir"], dtype=torch.float32)
        srcpos_norm = srcpos / (torch.tensor([X, Y, Z], dtype=torch.float32) - 1)
        A = math.radians(float(meta["A_deg"])); B = math.radians(float(meta["B_deg"]))
        ang = torch.tensor([math.sin(A), math.cos(A), math.sin(B), math.cos(B)])
        light = torch.cat([srcpos_norm, srcdir, ang]).to(self.device)
        return dict(logflu=logflu_t, bright_idx=bright_idx, light=light,
                    srcpos=srcpos.to(self.device), srcdir=srcdir.to(self.device),
                    fmax=fmax, floor=floor, peakpos=peakpos)

    def get(self, scene):
        if scene["geom_key"] != self._gk:
            self._geo = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._geo = self._load_geometry(scene); self._gk = scene["geom_key"]; self._ik = None
        if scene["fluence"] != self._ik:
            self._il = self._load_illumination(scene, self._geo); self._ik = scene["fluence"]
        return {**self._geo, **self._il, "vol_shape": self._geo["vol_shape"]}


sample_points = D1.sample_points


if __name__ == "__main__":
    sc = discover_scenes_v3()
    labels, held = split_labels(sc)
    from collections import Counter
    kc = Counter(s["kind"] for s in sc); lc = Counter(labels)
    print(f"v3 scenes: {len(sc)} | kinds={dict(kc)} | splits={dict(lc)}")
    print(f"held angles={len(held)}")
