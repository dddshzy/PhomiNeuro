#!/usr/bin/env python3
"""
V18 hero3 VERSION (2026-08-10). Same measurement, same stride, same electrodes, same tissue rows;
only the checkpoint changes. Two things were checked before trusting this path with hero3, because
it does NOT go through build_predictor_v14:

  * T.build_model forwards cfg["drop_feats"], so hero3's dropped optical block and sin/cos angles
    are honoured and the network is rebuilt at its true in_dim (1536, not 1578).
  * eval_v11_level2.predict forwards cfg["path_tri"]. Its own comment records that it was the THIRD
    copy of that call to have forgotten it; a model trained with trilinear path sampling and scored
    with nearest-voxel gathers is a silent mis-scoring, and hero3 has path_tri=True.

The geometry is model-independent, so n_vox per head must come out bit-identical to the V17 run --
that is asserted at the end rather than assumed.

  python batch_energy_deposition_v18h3.py --gpu 0
Per-(head, electrode) absorbed-energy DEPOSITION fractions from the V17 surrogate.

For each scene we take the surrogate's linear time-integrated fluence over every tissue voxel
(mua>0), multiply by that voxel's absorption coefficient mua (deposition rate = fluence * mua =
absorbed power density), and sum by tissue:
    gm_frac  = sum(dep over GM voxels)   / sum(dep over ALL tissue voxels)
    wm_frac  = sum(dep over WM voxels)   / sum(dep over ALL tissue voxels)
    other_frac = 1 - gm_frac - wm_frac   (scalp, skull, CSF, muscle, ...)
i.e. where the deposited optical energy goes. The surrogate output is per-gate log10 fluence, so
linear fluence = (10**output).sum(gates) (same convention as batch_surface_fluence).

mua map and tissue labels both come from the phantom `vol_prop_eye_aseg` (col 0 = mua; labels via
mni_normalize.optical_to_label, GM=2 WM=1 CSF=3). Tissue voxels are stride-subsampled (default 4)
for speed — the ratio is unbiased under uniform subsampling (verified: stride-8 matches full to 1e-3).

Output CSV (resume-safe): dataset_OASIS/pup_mni/cohort_energy_deposition.csv
Usage: python batch_energy_deposition.py --gpu 0 [--stride 4]
"""
import os, sys, csv, argparse, time
import numpy as np, torch, scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "data_expansion"))
import repro_config as RC
import train_inr_v11 as T
from eval_v11_level2 import predict
import mni_normalize as MN

GM, WM, CSF = 2, 1, 3      # phantom _P0F rows (CSF added 0803 for the Mantel-test tissue triple)
MNI_DIR = str(RC.MNI_DATASET_DIR)
EEG19 = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "C3", "C4", "Cz",
         "P3", "P4", "Pz", "T3", "T4", "T5", "T6", "O1", "O2"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RC.INR_CHECKPOINT_DIR / "inr_v18hero3_s0_base.pt"))
    ap.add_argument("--electrodes", nargs="+", default=EEG19)
    ap.add_argument("--out", default=str(
        RC.PUP_MNI_DIR / "cohort_energy_deposition_csf_v18h3.csv"))
    ap.add_argument("--heads-file", default=str(
        RC.OASIS_METADATA_DIR / "selected_84_groups.csv"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--no-assert-hero3", dest="assert_hero3", action="store_false",
                    default=True)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    grp = {r["subject"].lower(): r["group"] for r in csv.DictReader(open(a.heads_file))}

    ck = torch.load(a.ckpt, map_location="cpu"); cfg = ck["cfg"]
    if a.assert_hero3:
        want = dict(num_freq=4, num_freq_t=3, src_ref="entry", path_tri=True)
        bad = [f"{k}={cfg.get(k)!r} (expected {v!r})" for k, v in want.items() if cfg.get(k) != v]
        if list(cfg.get("drop_feats") or []) != ["optical", "light:6:10"]:
            bad.append(f"drop_feats={cfg.get('drop_feats')!r}")
        if os.path.basename(str(cfg.get("buffer", ""))) != "bufmm_v16":
            bad.append(f"buffer={os.path.basename(str(cfg.get('buffer','')))!r} (want bufmm_v16, K=3000)")
        if bad:
            raise SystemExit("[v18h3] checkpoint is not the finalised hero3:\n  - " + "\n  - ".join(bad))
        print(f"[v18h3] cfg VERIFIED hero3: num_freq=4 num_freq_t=3 src_ref=entry path_tri=True "
              f"drop_feats={cfg['drop_feats']} buffer=bufmm_v16 (K=3000)", flush=True)
    T.SRC_REF = cfg["src_ref"]
    m = T.build_model(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    store = T.SceneStore(dev)
    scmap = {(s["head"], s["tag"]): s for s in T.discover_scenes()}
    print(f"model {os.path.basename(a.ckpt)} dev={dev} stride={a.stride} | {len(grp)} heads x "
          f"{len(a.electrodes)} electrodes", flush=True)

    cols = ["subject", "group", "electrode", "csf_frac", "gm_frac", "wm_frac", "other_frac", "n_vox"]
    done = set()
    if os.path.isfile(a.out):
        for r in csv.DictReader(open(a.out)): done.add((r["subject"], r["electrode"]))
    new = not os.path.isfile(a.out)
    fh = open(a.out, "a", newline=""); w = csv.writer(fh)
    if new: w.writerow(cols)
    t0 = time.time(); n = 0

    def surr_linear(pts):
        pr = predict(m, cfg, sd, torch.tensor(pts, dtype=torch.float32, device=dev), dev).cpu().numpy()
        assert pr.ndim == 2 and pr.shape[1] == T.N_STEP, f"predict returned {pr.shape}, want (N, 10)"
        return (10.0 ** pr).sum(1)          # time-integrate the 10 gates, then weight by mu_a

    for head in grp:
        if all((head, el) in done for el in a.electrodes): continue
        try:
            vp = sio.loadmat(os.path.join(MNI_DIR, f"{head}_v11mni_F810.mat"))["vol_prop_eye_aseg"]
            mua = vp[..., 0]; lab = MN.optical_to_label(vp); tis = mua > 0
            coords = np.argwhere(tis).astype(np.float32)[::a.stride]
            muav = mua[tis][::a.stride]; labv = lab[tis][::a.stride]
        except Exception as e:
            print(f"  {head}: head-load FAIL {e}", flush=True); continue
        for el in a.electrodes:
            if (head, el) in done: continue
            sc = scmap.get((head, el))
            if sc is None: print(f"  {head}/{el}: no scene", flush=True); continue
            try:
                sd = store.get(sc)
                dep = surr_linear(coords) * muav
                tot = dep.sum()
                if not np.isfinite(tot) or tot <= 0:
                    print(f"  {head}/{el}: bad tot {tot}", flush=True); continue
                csf = dep[labv == CSF].sum(); gm = dep[labv == GM].sum(); wm = dep[labv == WM].sum()
                w.writerow([head, grp[head], el, f"{csf/tot:.6f}", f"{gm/tot:.6f}", f"{wm/tot:.6f}",
                            f"{1-(csf+gm+wm)/tot:.6f}", len(coords)]); fh.flush(); n += 1
                if n % 40 == 0:
                    print(f"  [{n}] {head}/{el} csf={csf/tot:.3f} gm={gm/tot:.3f} wm={wm/tot:.3f} "
                          f"({(time.time()-t0)/60:.1f}min)", flush=True)
            except Exception as e:
                print(f"  {head}/{el}: FAIL {type(e).__name__} {e}", flush=True)
    fh.close()
    print(f"done {n} scenes in {(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
