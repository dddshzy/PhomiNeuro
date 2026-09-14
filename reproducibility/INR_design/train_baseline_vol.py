#!/usr/bin/env python3
"""
Train a VOXEL (U-Net / ResUNet / FNO-3D) or OPERATOR (DeepONet) fluence baseline on
the fixed-grid 128^3 cache.  Shared with our model: same target (floored log10 Phi),
same decade-stratified weighted loss (hero weighting; no PINN/linear), same VAL-head
selection.  Voxel input = [mu_a/0.25, mu_s/45, source-blob, cos(theta)] (vol_common).

  python train_baseline_vol.py --gpu 0 --arch unet --cache "$PHOMINEURO_VOLUME_CACHE_DIR"
  FMINR_V10=1 python train_baseline_vol.py --gpu 1 --arch segresnet --cache ...
  FMINR_V10=1 python train_baseline_vol.py --gpu 2 --arch fno       --cache ...
  FMINR_V10=1 python train_baseline_vol.py --gpu 3 --arch deeponet  --cache ...
"""
import os, sys, json, time, math, argparse
import numpy as np, torch
# Use filesystem-backed tensor sharing to avoid dependence on shared-memory capacity.
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
import v2_manifest as M
import repro_config as RC
from baselines import make_voxel_model, DeepONet, count_params
from vol_common import source_channels, light_channels

DEC_SH, DEC_MID, DEC_DEEP = 2.5, 6.5, 8.0
SHALLOW_BOOST, CSF_BOOST = 3.0, 3.0

# Tolerance on the censored-set test, and it is load-bearing, not cosmetic.
#
# The target is floored at exactly 8 decades below the scene peak, so every censored voxel should
# satisfy dec == DEC_DEEP exactly. It does not, because make_vol_cache stores tgt as float16
# (`.half()`), whose spacing near the floor value (~ -6.5) is 0.0039. Rounding therefore scatters
# the floor across dec in [7.9982, 8.0020] -- one float16 ULP wide, straddling the threshold.
# Measured over 60 training scenes: 33 of them (55%) have dec.max() < 8.0, so an exact `>= 8.0`
# test excluded NOTHING and the loss fitted the MC detection limit as if it were data -- 92% of the
# tissue voxels in those scenes, diluting the gradient on real signal about twelvefold, while the
# other 45% of scenes excluded it correctly. One objective, two behaviours, decided by rounding.
#
# 0.01 is chosen to be wider than the float16 ULP (0.0039) and far narrower than the gap to genuine
# near-floor signal: in the measured distributions the band [7.9, 7.99) holds only 0.02-1.2% of
# tissue voxels, so the tolerance cannot swallow a meaningful amount of real data.
# The buffer path used by the coord baselines and by our model is NOT affected -- it keeps full
# precision and its dec lands on 8.000000 exactly.
DEC_TOL = 0.01


def weight_vol(inp2, tgt, fmax):
    """Per-voxel loss weight (numpy, g,g,g): hero decade stratification, tissue-masked."""
    mua = inp2[0].astype(np.float32) * 0.25; mus = inp2[1].astype(np.float32) * 45.0
    tissue = mus > 0
    dec = math.log10(fmax) - tgt.astype(np.float32)
    csf = (mua <= 0.005) & (mus > 1e-6) & (mus <= 0.5)
    w = np.ones_like(dec)
    shallow = dec < DEC_SH; mid = (dec >= DEC_SH) & (dec < DEC_MID)
    unreach = dec >= DEC_DEEP - DEC_TOL
    wc = 1.0 + CSF_BOOST * np.clip((dec - DEC_SH) / (DEC_MID - DEC_SH), 0, 1)
    w = np.where(shallow, 1.0 + SHALLOW_BOOST, w)
    w = np.where(mid & csf, wc, w)
    w = np.where(unreach, 0.0, w)
    return (w * tissue).astype(np.float32)


def weight_vol_torch(inp2, tgt, fmax):
    """GPU/batched port of weight_vol. inp2 (B,2,g,g,g), tgt (B,10,g,g,g), fmax (B,). Computed in the
    training step on the (idle) GPU instead of per-sample on the CPU DataLoader: weight_vol was ~387ms
    (87%) of the per-sample cost and is recomputed identically every epoch, starving the GPU. Single-
    gate quantities (tissue/csf, from the gate-less inp2) get a gate axis via unsqueeze(1) to broadcast
    against the (B,10,g,g,g) decade tensor -- same broadcasting the numpy version relied on."""
    mua = inp2[:, 0] * 0.25; mus = inp2[:, 1] * 45.0                          # (B,g,g,g)
    tissue = (mus > 0).unsqueeze(1)                                            # (B,1,g,g,g)
    dec = torch.log10(fmax).view(-1, 1, 1, 1, 1) - tgt                         # (B,10,g,g,g)
    csf = ((mua <= 0.005) & (mus > 1e-6) & (mus <= 0.5)).unsqueeze(1)          # (B,1,g,g,g)
    shallow = dec < DEC_SH; mid = (dec >= DEC_SH) & (dec < DEC_MID)
    unreach = dec >= DEC_DEEP - DEC_TOL                                        # see DEC_TOL
    wc = 1.0 + CSF_BOOST * ((dec - DEC_SH) / (DEC_MID - DEC_SH)).clamp(0, 1)
    w = torch.ones_like(dec)
    w = torch.where(shallow, torch.full_like(w, 1.0 + SHALLOW_BOOST), w)
    w = torch.where(mid & csf, wc, w)
    w = torch.where(unreach, torch.zeros_like(w), w)
    # The censored set is returned alongside the weight because the floor hinge needs exactly the
    # voxels the weighted MSE drops. Deriving it again in the training step would mean repeating the
    # threshold, and the two copies would drift.
    return w * tissue, unreach & tissue


class VolCache(Dataset):
    def __init__(self, cache, split, kinds):
        man = json.load(open(os.path.join(cache, "manifest.json")))
        self.grid = man["grid"]; self.cache = cache
        self.items = [it for it in man["items"] if it["split"] == split and it["kind"] in kinds]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        z = torch.load(os.path.join(self.cache, self.items[i]["file"]))
        inp2 = z["inp2"].float(); tgt = z["tgt"].float()
        # weight is now built on the GPU in the training step (weight_vol_torch); the DataLoader only
        # ships the scalar fmax it needs, not the recomputed (10,g,g,g) weight volume.
        fmax = torch.tensor(float(z["fmax"]), dtype=torch.float32)
        return (inp2, tgt, fmax, z["srcpos"].float(), z["srcdir"].float(), z["vol_shape"].float())


SRC_REF = "source"          # set from --src-ref; read by make_inp4 (the single blob-placement site)
_MARGIN = 15                # illumination.SRC_R_MARGIN


def make_inp4(inp2, srcpos, srcdir, vs, grid, with_light=False):
    """(B,4,g,g,g) or (B,14,g,g,g): cached mu_a/mu_s + on-the-fly source channels (+ light broadcast).

    With SRC_REF="entry" the blob is placed at the beam ENTRY point instead of the mcx source point.
    srcdir points into the head, so entry = srcpos + 15*srcdir. The sub-voxel snap that the point
    feature path applies is deliberately skipped: it moves the point by <=1 full-resolution voxel,
    i.e. ~0.5 cells on the grid these models actually see, which is far below the blob's own width."""
    B = inp2.shape[0]
    if SRC_REF == "entry":
        srcpos = srcpos + _MARGIN * srcdir
    src = torch.stack([source_channels(srcpos[b], srcdir[b], vs[b], grid) for b in range(B)], 0)
    parts = [inp2, src]
    if with_light:
        parts.append(torch.stack([light_channels(srcpos[b], srcdir[b], vs[b], grid) for b in range(B)], 0))
    return torch.cat(parts, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arch", choices=["unet", "segresnet", "fno", "deeponet", "dynunet", "unetr"], required=True)
    ap.add_argument("--floor-hinge", type=float, default=0.0,
                    help="weight of the one-sided censored (floor) hinge on the dark zone, matching "
                         "the coord baselines' --floor-hinge and our model's lambda_floor. "
                         "0 disables dark-zone supervision.")
    ap.add_argument("--hinge-offset", type=float, default=0.5,
                    help="shift the hinge knee by this many decades, as in our model.")
    ap.add_argument("--src-ref", choices=["source", "entry"], default="source",
                    help="V17 PARITY PROBE. 'entry' moves the source blob from the mcx source point "
                         "(SRC_R_MARGIN=15 voxels outside the scalp) to the beam entry point, the same "
                         "reference V17's point features use. The volumetric family already has the "
                         "information (blob position + direction channel), so this is a representation "
                         "test, not an information change: if the score does not move, that is evidence "
                         "the family is insensitive to the convention and the other volumetric runs do "
                         "not need repeating. Sub-voxel snapping is deliberately NOT applied -- these "
                         "models work on a downsampled grid where <1 full-res voxel is meaningless.")
    ap.add_argument("--cache", default=RC.VOLUME_CACHE_DIR)
    ap.add_argument("--timeres", action="store_true",
                    help="TIME-RESOLVED: 10-gate target/output (out_c=10) from the --v14tr cache, "
                         "so grid baselines can be benchmarked against the time-resolved V14 hero.")
    ap.add_argument("--v14cw", action="store_true",
                    help="V14 CW benchmark: add the light(10) broadcast channels (in_c 4->14).")
    ap.add_argument("--kinds", nargs="+", default=["base", "rot"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    # Optional gradient clipping for numerical stability.
    ap.add_argument("--clip-grad", type=float, default=0.0,
                    help="max grad norm; 0 = off (the setting every existing checkpoint used)")
    # CNN baselines default to fp16; bf16 is available for transformer attention.
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--points", type=int, default=8192)        # DeepONet pts/volume
    ap.add_argument("--tag", default="v10")
    ap.add_argument("--seed", type=int, default=None, help="reproducible repeat runs (multi-seed benchmark)")
    a = ap.parse_args()
    global SRC_REF
    SRC_REF = a.src_ref
    if a.seed is not None:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    name = f"bench_{a.arch}"; tag = f"{a.tag}_{name}"; os.makedirs(M.INR_CKPT_V3, exist_ok=True)

    tr = VolCache(a.cache, "train", a.kinds); va = VolCache(a.cache, "val_head", ["base"])
    grid = tr.grid
    trl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=2, pin_memory=True)
    print(f"[{tag}] grid={grid} train={len(tr)} val={len(va)} arch={a.arch}", flush=True)

    is_op = a.arch == "deeponet"
    in_c = 14 if (a.v14cw or a.timeres) else 4
    out_c = 10 if a.timeres else 1
    if is_op and a.timeres:
        raise SystemExit("--timeres is not implemented for the operator (deeponet) branch yet")
    model = (DeepONet(in_c=in_c) if is_op else
             make_voxel_model(a.arch, in_c=in_c, out_c=out_c)).to(dev)
    print(f"[{tag}] params={count_params(model):,} in_c={in_c} out_c={out_c}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)
    _amp_dt = torch.bfloat16 if a.amp_dtype == "bf16" else torch.float16
    # GradScaler exists to keep fp16 gradients off the denormal floor; bf16 has the range already,
    # so it is disabled there. That also matters for the clip below: unscale_ on a disabled scaler
    # raises, so the two settings have to be read together.
    scaler = torch.amp.GradScaler('cuda', enabled=(a.amp_dtype == "fp16"))

    def wmse_grid(pred, tgt, w):
        """CW: pred (B,1,g,g,g) vs tgt/w (B,g,g,g).  Time-resolved: all three are (B,10,g,g,g)
        (weight_vol is elementwise, so it already returns a per-gate weight)."""
        p = pred.squeeze(1) if pred.dim() == tgt.dim() + 1 else pred
        d = (p - tgt) ** 2
        return (w * d).sum() / w.sum().clamp_min(1e-6)

    def hinge_term(p, tgt, cen):
        """One-sided penalty on the censored (floor) set, identical in form and offset to the coord
        baselines' --floor-hinge and to our model's lambda_floor.

        Sub-floor voxels are LEFT-censored: the MC target records "the true value is at or below the
        detection limit", so predicting below it is the physically correct answer and a two-sided
        MSE against gt=floor would punish it. Only over-prediction is an error. Without this term
        the grid family gets no dark-zone supervision at all, which makes "the baselines leak into
        the dark zone" untestable -- they were never asked not to."""
        if a.floor_hinge <= 0 or not bool(cen.any()):
            return 0.0
        return a.floor_hinge * (torch.relu(p - tgt + a.hinge_offset)[cen] ** 2).mean()

    def step_voxel(inp4, tgt, w, cen, train):
        with torch.amp.autocast('cuda', dtype=_amp_dt):
            pred = model(inp4)
            p = pred.squeeze(1) if pred.dim() == tgt.dim() + 1 else pred
            loss = wmse_grid(pred, tgt, w) + hinge_term(p, tgt, cen)
        return loss

    def step_op(inp4, tgt, w, cen, train):
        B = inp4.shape[0]
        code = model.branch_code(inp4)                        # (B,K)
        loss = 0.0
        for b in range(B):
            wb = w[b].reshape(-1); cb = cen[b].reshape(-1)
            # Sample from the supervised set UNION the censored set. Sampling `w>0` alone -- what
            # this did before the hinge existed -- can never draw a censored voxel, so the hinge
            # would silently be a no-op for the operator family while the voxel family enforced it.
            pos = torch.nonzero(wb > 0).squeeze(1)
            pos_c = torch.nonzero(cb).squeeze(1)
            if pos.numel() + pos_c.numel() < 16:
                pos = torch.arange(wb.numel(), device=dev); pos_c = pos[:0]
            n_c = 0 if pos_c.numel() == 0 else min(a.points // 2, a.points)
            n_w = a.points - n_c if pos.numel() > 0 else 0
            n_c = a.points - n_w
            parts = []
            if n_w > 0:
                parts.append(pos[torch.randint(0, pos.numel(), (n_w,), device=dev)])
            if n_c > 0 and pos_c.numel() > 0:
                parts.append(pos_c[torch.randint(0, pos_c.numel(), (n_c,), device=dev)])
            sel = torch.cat(parts)
            zz = sel % grid; yy = (sel // grid) % grid; xx = sel // (grid * grid)
            xn = torch.stack([xx, yy, zz], 1).float() / (grid - 1) * 2 - 1
            pr = model(code[b:b + 1], xn).squeeze(0)          # (P,)
            tb = tgt[b].reshape(-1)[sel]; wbp = wb[sel]
            loss = loss + (wbp * (pr - tb) ** 2).sum() / wbp.sum().clamp_min(1e-6)
            loss = loss + hinge_term(pr, tb, cb[sel])
        return loss / B

    step = step_op if is_op else step_voxel
    hist = {"val": []}; best = float("inf"); t0 = time.time(); start_ep = 1
    # FULL-STATE resume: if a run is interrupted (shm crash, OOM, node hiccup) it must continue from the
    # last epoch, not restart from scratch. State (model+opt+sched+epoch+best+hist) is written EVERY
    # epoch to a PID-independent path; os.replace makes it atomic. A stale state from a prior config is
    # ignored via the arch/epochs guard.
    STATE = os.path.join(M.INR_CKPT_V3, f"state_{tag}.pt")

    def save_state(ep):
        tmp = f"{STATE}.tmp.{os.getpid()}"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "hist": hist,
                    "arch": a.arch, "epochs": a.epochs}, tmp)
        os.replace(tmp, STATE)

    if os.path.isfile(STATE):
        st = torch.load(STATE, map_location=dev)
        if st.get("arch") == a.arch and st.get("epochs") == a.epochs:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
            start_ep = st["epoch"] + 1; best = st["best"]; hist = st["hist"]
            print(f"[{tag}] RESUMED from epoch {st['epoch']} (best {best:.4f})", flush=True)
    for ep in range(start_ep, a.epochs + 1):
        model.train(); tl, nb = 0.0, 0
        for inp2, tgt, fmax, sp, sd_, vs in trl:
            inp2, tgt = inp2.to(dev), tgt.to(dev)
            w, cen = weight_vol_torch(inp2, tgt, fmax.to(dev))     # was CPU/DataLoader; now GPU
            inp4 = make_inp4(inp2, sp.to(dev), sd_.to(dev), vs.to(dev), grid, with_light=(a.v14cw or a.timeres))
            opt.zero_grad()
            loss = step(inp4, tgt, w, cen, True)
            scaler.scale(loss).backward()
            if a.clip_grad > 0:
                # unscale_ FIRST when the scaler is live: with fp16 AMP the gradients are still
                # scaled here, so clipping a scaled norm would clip by an arbitrary, loss-scale-
                # dependent threshold. Under bf16 the scaler is disabled and unscale_ would raise.
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
            scaler.step(opt); scaler.update()
            tl += float(loss); nb += 1
        sched.step()
        model.eval(); vl, vnb = 0.0, 0
        with torch.no_grad():
            for inp2, tgt, fmax, sp, sd_, vs in val:
                inp2, tgt = inp2.to(dev), tgt.to(dev)
                w, cen = weight_vol_torch(inp2, tgt, fmax.to(dev))
                inp4 = make_inp4(inp2, sp.to(dev), sd_.to(dev), vs.to(dev), grid, with_light=(a.v14cw or a.timeres))
                vl += float(step(inp4, tgt, w, cen, False)); vnb += 1
        vl /= max(vnb, 1); hist["val"].append(vl)
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "epoch": ep, "val_head": vl,
                        "arch": a.arch, "family": "operator" if is_op else "voxel",
                        "grid": grid, "params": count_params(model), "in_c": in_c, "v14cw": bool(a.v14cw),
                        "out_c": out_c, "timeres": bool(a.timeres),
                        "cfg": {"dec_sh": DEC_SH, "dec_mid": DEC_MID, "kinds": a.kinds,
                                "in_c": in_c, "v14cw": bool(a.v14cw), "out_c": out_c,
                                "timeres": bool(a.timeres), "src_ref": a.src_ref, "seed": a.seed,
                                "dec_tol": DEC_TOL, "floor_hinge": a.floor_hinge,
                                "lr": a.lr, "clip_grad": a.clip_grad, "amp_dtype": a.amp_dtype,
                                "hinge_offset": a.hinge_offset}},
                       os.path.join(M.INR_CKPT_V3, f"inr_{tag}.pt"))
        save_state(ep)                                  # resume point, every epoch
        if ep % 5 == 0 or ep == 1:
            print(f"  [{tag}] ep {ep}/{a.epochs} tr {tl/max(nb,1):.4f} val {vl:.4f} "
                  f"best {best:.4f} {time.time()-t0:.0f}s", flush=True)
    json.dump({"arch": a.arch, "best_val": best, "history": hist},
              open(os.path.join(M.INR_CKPT_V3, f"metrics_{tag}.json"), "w"))
    if os.path.isfile(STATE):
        os.remove(STATE)                                # done: drop the resume state
    print(f"[{tag}] DONE best_val={best:.4f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
