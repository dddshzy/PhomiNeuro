#!/usr/bin/env python3
"""TIME-RESOLVED coordinate baseline (rff / siren) — the coord-family counterpart of the time-resolved
grid baselines, so the V15 benchmark has a per-point (non-grid, non-FM) reference.

Apples-to-apples with ours: identical physical inputs (optical 4 + light 10 + srcfeat 2 + path 6) and
the SAME time ingestion as INRv7 -- the normalized gate code is appended to the conditioning vector --
but NO FM pyramid, which is the contribution being isolated. Trained off the same v14samp buffer with
the same decade-stratified weighting, so only architecture differs.

Per step a random gate k is drawn per point (mirrors train_inr_v11.batch_sample_t), so the model sees
all 10 gates without a 10x cost.

  python train_baseline_coord_tr.py --gpu 0 --arch rff --mm inr_checkpoints_v11/bufmm_v14samp --seed 1
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data_expansion")):
    sys.path.insert(0, p)
import v2_manifest as M
import train_inr_v11 as T
from baselines import make_coord_model, count_params

DEC_SH, DEC_MID, DEC_DEEP = 2.5, 6.5, 8.0
SHALLOW_BOOST, CSF_BOOST = 3.0, 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arch", choices=["rff", "siren"], required=True)
    ap.add_argument("--mm", default="inr_checkpoints_v11/bufmm_v14samp")
    ap.add_argument("--tag", default="v15tr")
    ap.add_argument("--seed", type=int, default=1)
    # INPUT ALIGNMENT with hero3. hero3 takes NO optical block and only 6 of the 10 light columns
    # (srcpos_norm + srcdir; the sin/cos angles are a bijective re-encoding of srcdir). Without these
    # the coord baselines carry MORE information than the model they are compared against, and the
    # temporal side carries LESS (raw scalar t vs a 7-dim band-limited PE) -- the pairing was never
    # matched in either direction. `--drop-optical --drop-light ang --num-freq-t 3` matches it.
    ap.add_argument("--drop-optical", action="store_true",
                    help="ALIGNMENT: remove the 4 optical columns entirely (in_features -4).")
    ap.add_argument("--drop-light", choices=["pos", "dir", "ang", "dirang"], default=None,
                    help="ALIGNMENT: remove that part of the 10-element illumination vector.")
    ap.add_argument("--num-freq-t", type=int, default=None,
                    help="V16: band-limit the gate-time positional encoding to this many bands "
                         "(unset = V15's raw scalar t). Mirrors ours' --num-freq-t so the F_T "
                         "comparison isolates F_T rather than the architecture.")
    ap.add_argument("--src-ref", choices=["source", "entry"], default="source",
                    help="V17 parity: 'entry' loads the SF_entry/LIGHT_entry/PATH_entry side files "
                         "built by make_entryfeat.py, i.e. the same entry-point reference our model "
                         "uses. Keeps the coordinate-INR family apples-to-apples with V17; recorded "
                         "in cfg['src_ref'] so the scorer's guard can enforce it.")
    ap.add_argument("--plain-loss", action="store_true",
                    help="PURE data-driven loss (uniform MSE over supervised voxels) instead of our "
                         "decade-stratified weighting. For a fair benchmark each model may train under "
                         "its OWN recommended objective (SIREN/RFF papers use a plain reconstruction "
                         "MSE) while all are scored by the SAME metric. Still masks the censored set "
                         "(dec>=DEC_DEEP), where the MC target is a detection-limit floor, not a value.")
    ap.add_argument("--floor-hinge", type=float, default=0.0,
                    help="weight of a censored-data (floor) hinge on dec >= DEC_DEEP, the dark zone. "
                         "0 = off; both loss branches then omit the censored region. "
                         "Use 1.0 to match our model's lambda_floor for a benchmark where every "
                         "family has been asked to handle the dark zone.")
    ap.add_argument("--hinge-offset", type=float, default=0.5,
                    help="shift the hinge knee by this many decades, as in our model. 0.5 is the "
                         "production value: it is the knee where deep leak falls without the "
                         "checkerboard returning.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    tag = f"{a.tag}_coord_{a.arch}" + (f"_ft{a.num_freq_t}" if a.num_freq_t else "") + f"_s{a.seed}"

    t0 = time.time()
    mm = a.mm if os.path.isabs(a.mm) else os.path.join(HERE, a.mm)
    z = torch.load(os.path.join(mm, "meta.pt"), map_location="cpu")
    XN = z["XN"].to(dev); OPT = z["OPT"].to(dev); LIGHT = z["LIGHT"].to(dev)
    SF = z["SF"].to(dev); GT10 = z["GT10"].to(dev); LFM = z["LFM"].to(dev)
    CSFB = z["CSFB"].to(dev); SP = z["SP"].to(dev)
    sfx = ""
    if a.src_ref == "entry":
        sfx = "_entry"
        for _f in ("SF_entry.npy", "LIGHT_entry.npy"):
            if not os.path.exists(os.path.join(mm, _f)):
                raise SystemExit(f"--src-ref entry needs {os.path.join(mm,_f)} -- run make_entryfeat.py")
        SF = torch.from_numpy(np.load(os.path.join(mm, "SF_entry.npy"))).to(dev)
        LIGHT = torch.from_numpy(np.load(os.path.join(mm, "LIGHT_entry.npy"))).to(dev)
        print(f"[{tag}] --src-ref entry: SF/LIGHT from *_entry.npy", flush=True)
    pth = os.path.join(mm, f"PATH{sfx}.npy")
    PATH = torch.from_numpy(np.load(pth)).to(dev) if os.path.exists(pth) else None
    path_dim = PATH.shape[1] if PATH is not None else 0
    tenc = T.T_ENC.to(dev)
    print(f"[{tag}] buffers {time.time()-t0:.0f}s total={GT10.shape[0]} path_dim={path_dim} "
          f"gates={GT10.shape[1]}", flush=True)

    LIGHT_SLICE = {"pos": (0, 3), "dir": (3, 6), "ang": (6, 10), "dirang": (3, 10)}
    drop_feats = []
    if a.drop_optical:
        drop_feats.append("optical")
    if a.drop_light:
        drop_feats.append("light:{}:{}".format(*LIGHT_SLICE[a.drop_light]))
    drop_feats = tuple(drop_feats)
    model = make_coord_model(a.arch, path_dim=path_dim, time_dim=1,
                             num_freq_t=a.num_freq_t, drop_feats=drop_feats).to(dev)
    _first = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
    print(f"[{tag}] arch={a.arch} params={count_params(model):,} "
          f"(time_dim=1, num_freq_t={a.num_freq_t}, drop_feats={drop_feats}, "
          f"in_features={_first.in_features})", flush=True)
    opt_ = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=a.epochs, eta_min=1e-5)

    tr = torch.nonzero(SP == 0).squeeze(1); vh = torch.nonzero(SP == 1).squeeze(1)

    def sample_gate(b, gen=None):
        """random gate per point -> (gt, t, dec), matching train_inr_v11.batch_sample_t"""
        k = torch.randint(0, GT10.shape[1], (b.numel(),), device=dev, generator=gen)
        gt = GT10[b].gather(1, k.view(-1, 1))
        t = tenc[k].view(-1, 1)
        return gt, t, LFM[b] - gt

    def weights(dec, b):
        csf = CSFB[b] > 0.5
        w = torch.ones_like(dec)
        shallow = dec < DEC_SH; mid = (dec >= DEC_SH) & (dec < DEC_MID); unreach = dec >= DEC_DEEP
        wc = 1.0 + CSF_BOOST * ((dec - DEC_SH) / (DEC_MID - DEC_SH)).clamp(0, 1)
        w = torch.where(shallow, torch.full_like(w, 1.0 + SHALLOW_BOOST), w)
        w = torch.where(mid & csf, wc, w)
        return torch.where(unreach, torch.zeros_like(w), w)

    def ev(idx):
        """Validation objective consistent with the selected training objective.
        Measured consequence -- 3 of 9 runs picked their "best" checkpoint at ep1-4
        (v16trplain_rff_s2 @ep1, v17trplain_rff_s1 @ep1, v17tr_rff_s1 @ep2) and rose monotonically
        after, i.e. those baselines were reported essentially untrained. It is not a learning-rate
        problem and lowering LR cannot fix it. train_inr_v11.py:947 documents the identical bug
        ("a plain MSE val kept selecting epoch 1") and its ev() is the reference for this fix.

        With --floor-hinge the censored term is one-sided, matching the training hinge: predicting
        BELOW the floor is what the censored likelihood allows and is physically correct, so it must
        not be punished. Without it the censored set is excluded, exactly as training excludes it.
        """
        model.eval(); tot = nb = 0
        g = torch.Generator(device=dev); g.manual_seed(1234)      # fixed gates -> comparable val
        with torch.no_grad():
            for j in range(0, idx.numel(), a.batch):
                b = idx[j:j + a.batch]
                gt, t, dec = sample_gate(b, g)
                p = model.forward_feats(None, XN[b], OPT[b], LIGHT[b], SF[b],
                                        path=None if PATH is None else PATH[b], t=t)
                cen = dec >= DEC_DEEP
                if a.floor_hinge > 0:
                    err = torch.where(cen, torch.relu(p - gt + a.hinge_offset), p - gt)
                    tot += float((err ** 2).mean())
                else:
                    m = (~cen).float()
                    tot += float((m * (p - gt) ** 2).sum() / m.sum().clamp_min(1e-6))
                nb += 1
        return tot / max(nb, 1)

    hist = {"val_head": []}; best = float("inf"); t1 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); perm = tr[torch.randperm(tr.numel(), device=dev)]; tl, nb = 0.0, 0
        for j in range(0, perm.numel(), a.batch):
            b = perm[j:j + a.batch]
            gt, t, dec = sample_gate(b)
            pred = model.forward_feats(None, XN[b], OPT[b], LIGHT[b], SF[b],
                                       path=None if PATH is None else PATH[b], t=t)
            if a.plain_loss:
                m = (dec < DEC_DEEP).float()                 # mask censored floor; else uniform MSE
                loss = (m * (pred - gt) ** 2).sum() / m.sum().clamp_min(1e-6)
            else:
                w = weights(dec, b)
                loss = (w * (pred - gt) ** 2).sum() / w.sum().clamp_min(1e-6)
            if a.floor_hinge > 0:
                # CENSORED-DATA hinge, same form and same offset as our model's. Sub-floor voxels are
                # left-censored ("true value <= floor"), so only over-prediction is an error; a
                # two-sided MSE against gt=floor would punish the physically correct answer. Without
                # this term the coordinate family gets NO dark-zone supervision at all, which makes
                # "the baselines leak into the dark zone" an untestable claim -- they were never
                # asked not to.
                cen = dec >= DEC_DEEP
                if bool(cen.any()):
                    loss = loss + a.floor_hinge * (
                        torch.relu(pred[cen] - gt[cen] + a.hinge_offset) ** 2).mean()
            opt_.zero_grad(); loss.backward(); opt_.step(); tl += loss.item(); nb += 1
        sched.step(); vl = ev(vh); hist["val_head"].append(vl)
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "epoch": ep, "val_head": vl,
                        "arch": a.arch, "family": "coord_tr", "params": count_params(model),
                        "path_dim": path_dim, "time_dim": 1, "seed": a.seed,
                        "num_freq_t": a.num_freq_t, "drop_feats": list(drop_feats),
                        "cfg": {"path_dim": path_dim, "drop_feats": list(drop_feats), "time_dim": 1, "seed": a.seed, "src_ref": a.src_ref,
                                "plain_loss": a.plain_loss,
                                "floor_hinge": a.floor_hinge, "hinge_offset": a.hinge_offset,
                                "num_freq_t": a.num_freq_t,
                                "buffer": mm, "dec_sh": DEC_SH, "dec_mid": DEC_MID}},
                       os.path.join(M.INR_CKPT_V3, f"inr_{tag}.pt"))
        if ep % 10 == 0 or ep == 1:
            print(f"  [{tag}] ep {ep}/{a.epochs} tr {tl/nb:.4f} vh {vl:.4f} best {best:.4f} "
                  f"{time.time()-t1:.0f}s", flush=True)
    json.dump({"arch": a.arch, "best_val_head": best, "history": hist},
              open(os.path.join(M.INR_CKPT_V3, f"metrics_{tag}.json"), "w"))
    print(f"[{tag}] DONE best={best:.4f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
