#!/usr/bin/env python3
"""Pre-normalise every phantom into a memory-mappable .npy for the encoder's MAE training.

The MAE dataset draws a 64^3 crop per sample, but `OpticalPropertyDataset._load_data` reached that
crop by running `sio.loadmat` on the whole phantom -- decompressing a 275 MB (224x256x300x4) array
and then normalising it -- for EVERY sample. Measured: 0.34 s to load + 0.12 s to transpose and
normalise = 0.46 s per sample, against 0.002 s for a memmapped crop. With 4 workers that caps
throughput at ~8.7 samples/s while the GPU can consume ~16/s, so encoder training was I/O-bound,
not compute-bound. This is also why adding GPUs would not have helped: DDP multiplies the I/O and,
at fixed per-GPU batch, it QUARTERS the number of gradient steps -- the opposite of what an
under-trained encoder needs.

Writes {head}.npy in (C, X, Y, Z) layout, ALREADY normalised with optical_config, so the Dataset
only has to memmap and slice.

Usage: python prep_encoder_npy.py [--out .../dataset_v11_npy] [--heads sh001 ...]
"""
import os, sys, argparse
import numpy as np, scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "data_expansion"))
import optical_config as OC
import repro_config as RC

SRC = RC.MNI_DATASET_DIR
PROP_KEY = "vol_prop_eye_aseg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RC.WORK_DIR, "dataset_v11_npy"))
    ap.add_argument("--heads", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import v11_split as SP
    heads = a.heads or SP.HEADS
    print(f"pre-normalising {len(heads)} phantoms -> {a.out}", flush=True)
    for i, h in enumerate(heads):
        dst = os.path.join(a.out, f"{h}.npy")
        src = os.path.join(SRC, f"{h}_v11mni_F810.mat")
        # Keyed on the phantom, like the MCX cache: a stale .npy after a phantom fix is exactly
        # the failure that let the scalp bug survive a full re-simulation.
        if os.path.isfile(dst) and not a.force and os.path.getmtime(dst) >= os.path.getmtime(src):
            continue
        vol = sio.loadmat(src)[PROP_KEY].astype(np.float32)      # (X,Y,Z,4)
        vol = np.transpose(vol, (3, 0, 1, 2))                    # (4,X,Y,Z)
        nrm = OC.normalize(vol, channels=OC.CHANNELS, channel_dim=0).astype(np.float32)
        np.save(dst, nrm)
        if i % 25 == 0:
            print(f"  [{i}/{len(heads)}] {h}  {nrm.shape}  {nrm.nbytes/1e6:.0f} MB", flush=True)
    tot = sum(os.path.getsize(os.path.join(a.out, f"{h}.npy")) for h in heads
              if os.path.isfile(os.path.join(a.out, f"{h}.npy")))
    print(f"done. {len(heads)} volumes, {tot/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
