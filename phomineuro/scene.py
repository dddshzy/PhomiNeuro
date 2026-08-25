"""A scene = one head model + one illumination site.

Everything the surrogate needs at inference, and nothing it does not. In particular NO Monte-Carlo
fluence: in the production tree the scene loader reads the MCX output too, but only to build
training targets and evaluation masks. Prediction never touches it.
"""
import glob
import json
import os

import numpy as np
import scipy.io as sio
import torch

from .features import entry_of, light_vector

PROP_KEY = "vol_prop_eye_aseg"


class Scene:
    """Load a head + electrode. `src_ref` must come from the checkpoint cfg, never a default."""

    def __init__(self, head, electrode, root, device, src_ref="entry", pyramid=None,
                 cache_dir=None, vista3d_ckpt=None, weights_dir=None, verbose=True):
        self.head, self.electrode, self.device = head, electrode, device
        self.src_ref = src_ref

        head_path = os.path.join(root, "demo_heads", f"{head}_v11mni_F810.mat")
        if not os.path.isfile(head_path):
            avail = sorted(os.path.basename(p).split("_")[0]
                           for p in glob.glob(os.path.join(root, "demo_heads", "*_v11mni_F810.mat")))
            raise FileNotFoundError(f"no head {head!r}; shipped heads: {avail}")
        meta_path = os.path.join(root, "demo_heads", "meta",
                                 f"meta_{head}_F810_{electrode}_r5_t10.json")
        if not os.path.isfile(meta_path):
            avail = sorted(os.path.basename(p).split("_")[3]
                           for p in glob.glob(os.path.join(root, "demo_heads", "meta",
                                                           f"meta_{head}_F810_*_r5_t10.json")))
            raise FileNotFoundError(f"no site {electrode!r} for {head}; shipped sites: {avail}")

        prop = np.transpose(sio.loadmat(head_path)[PROP_KEY].astype(np.float32), (3, 0, 1, 2))
        self.prop_phys = prop                                   # (4,X,Y,Z) mu_a, mu_s, g, n
        self.prop = torch.from_numpy(prop).unsqueeze(0).to(device)
        X, Y, Z = self.prop.shape[-3:]
        self.vol_shape = torch.tensor([X, Y, Z], device=device)
        self.tissue = torch.from_numpy(prop[1] > 0).to(device)  # mu_s > 0 == not air

        meta = json.load(open(meta_path))
        self.mcx_srcpos = torch.tensor(meta["srcpos"], dtype=torch.float32, device=device)
        self.srcdir = torch.tensor(meta["srcdir"], dtype=torch.float32, device=device)

        # THE ENTRY-POINT CONVENTION. The mcx source is parked 15 voxels outside the scalp, so every
        # source-referenced ray starts with ~15 mm of air -- and air passes the CSF test used by the
        # path features, which is how 53-63% of L_csf came to be air. Referencing the entry point
        # instead also makes the surrogate exactly invariant to sliding the source along its own
        # axis, which a collimated beam must be. Feeding an entry-trained model source-referenced
        # features is silent: the numbers come out wrong, nothing raises.
        if src_ref == "entry":
            self.srcpos = entry_of(self.mcx_srcpos, self.srcdir, self.tissue, self.vol_shape)
        elif src_ref == "source":
            self.srcpos = self.mcx_srcpos
        else:
            raise ValueError(f"unknown src_ref {src_ref!r}")
        self.light = light_vector(self.srcpos, self.srcdir, self.vol_shape.tolist()).to(device)

        if pyramid is not None:
            self.pyramid = [p.to(device) for p in pyramid]
        elif cache_dir is not None:
            from .encoder import get_pyramid
            pyr = get_pyramid(head, prop, cache_dir, vista3d_ckpt, weights_dir, device, verbose)
            self.pyramid = [p.to(device) for p in pyr]
        else:
            self.pyramid = None
        if verbose:
            off = float((self.srcpos - self.mcx_srcpos).norm())
            print(f"[scene] {head}/{electrode}  volume {X}x{Y}x{Z}  src_ref={src_ref}  "
                  f"srcpos={[round(float(v), 1) for v in self.srcpos]} "
                  f"({off:.1f} voxels from the mcx source)")

    def tissue_coords(self):
        """(N,3) float voxel coords of every non-air voxel."""
        return torch.nonzero(self.tissue, as_tuple=False).float()

    def mc_reference(self, root):
        """The Monte-Carlo field for this scene, if it was shipped. (X,Y,Z,10) linear fluence."""
        p = os.path.join(root, "demo_heads", "mc_reference",
                         f"fluence_{self.head}_F810_{self.electrode}_r5_t10.mat")
        if not os.path.isfile(p):
            return None
        return sio.loadmat(p)["fluence"].astype(np.float32)
