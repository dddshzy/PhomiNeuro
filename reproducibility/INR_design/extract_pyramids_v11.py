#!/usr/bin/env python3
"""
Extract spatial FM feature pyramids for the 225 rigid-MNI-normalized v11 heads,
using the v11-finetuned encoder (FM_tune_v11). Pyramids are xyz-ONLY / static (the
head is fixed over the sim window), shared across all illumination angles AND time
gates -- t never enters the encoder.

Reads dataset_v11_mni/{head}_v11mni_F810.mat -> extracted_pyramids_v11/{head}_..._pyramid.pt.
Shard across GPUs with FM_NSHARD / FM_SHARD. Uses the v11 encoder via FM_CKPT_DIR.
"""
import os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
import repro_config as RC
os.environ.setdefault("FM_CKPT_DIR", str(RC.FM_ADAPTER_DIR))
import extract_pyramids as EP          # reads FM_CKPT_DIR at import -> v11 encoder
import v11_split as SP

DATASET = RC.MNI_DATASET_DIR
OUT = os.environ.get("FM_PYR_OUT", RC.PYRAMID_DIR)
# The encoder is the pyramid's SOURCE. When the encoder is retrained, every pyramid is
# stale -- writing v12 pyramids over the v11 directory would destroy the only control we
# have, and (worse) a partially-overwritten directory would train on a mixture. Keep them
# apart: FM_CKPT_DIR selects the encoder, FM_PYR_OUT selects where its pyramids land.


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    model = EP.build_model(device)
    # FM_HEADS lets extra heads (e.g. the OASIS external-validation subjects, which are not in
    # SP.HEADS) get a pyramid without touching the default 225-head behaviour.
    extra = [h for h in os.environ.get("FM_HEADS", "").split() if h]
    files = [f"{h}_v11mni_F810.mat" for h in (extra or SP.HEADS)]  # default: all 225
    nshard = int(os.environ.get("FM_NSHARD", 1)); shard = int(os.environ.get("FM_SHARD", 0))
    skip = os.environ.get("FM_SKIP_EXISTING", "1") == "1"
    files = files[shard::nshard]
    print(f"[v11 pyramids shard {shard}/{nshard}] {len(files)} heads -> {OUT}", flush=True)
    with torch.no_grad():
        for fn in files:
            out = os.path.join(OUT, fn.replace(".mat", "_pyramid.pt"))
            if skip and os.path.isfile(out):
                print(f"  [skip] {fn}"); continue
            vol = EP.load_and_normalize(os.path.join(DATASET, fn))
            pyr = EP.extract_pyramid(model, vol, device)
            torch.save({"pyramid": pyr, "shapes": [t.shape for t in pyr],
                        "volume_shape": tuple(vol.shape[-3:])}, out)
            print(f"  [ok] {fn}", flush=True)
    print(f"[v11 pyramids shard {shard}] done", flush=True)


if __name__ == "__main__":
    main()
