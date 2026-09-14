#!/usr/bin/env python3
"""Per-gate DIAGNOSTIC panel — why does R2 vary across gates, and why are gates 1-3 our weak spot?

R2 = 1 - SSE/SS_tot is a RATIO, so a gate's R2 moves for two independent reasons:
  (a) the model's error changes            -> numerator
  (b) the GT's own spread changes          -> denominator (measured: std dips to a minimum at
      gates 3-4 and rises at both ends, so identical error looks worse mid-sequence)
Reporting R2 alone cannot separate these. This script emits scale-free and error-structure metrics
that can, plus the depth/tissue breakdown that localises WHERE the error sits.

Metric groups (all per gate):
  variance-context : gt_std, sse, r2, r2_check          -- decomposes R2 into numerator/denominator
  scale-free       : rho (Pearson), spearman, ccc       -- correlation, immune to the variance dip
  amplitude        : rmse, mae, medae, p90ae            -- raw error magnitude
  systematic       : bias, slope, intercept             -- offset vs gain error (regress pred on gt)
  structure        : gradR2, hpRMSE                     -- gradient / high-pass field fidelity
  depth-resolved   : rmse_dec0_2, rmse_dec2_4, ...      -- error vs depth band (which band hurts)
  tissue-resolved  : rmse_csf, rmse_brain, rmse_other   -- error vs tissue class
  set-agreement    : dice, leak, miss                   -- reachable-set geometry

  EVAL_SPLIT=test python diag_pergate.py --gpu 0 --models v15=... tr_dynunet=... --out diag.json
"""
import os, sys, math, json, argparse, zlib
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "data_expansion"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, p)
import train_inr_v11 as T
import inr_dataset as D1
from eval_v11_level2 import load, SIM7
from eval_benchmark_v14cw import build_predictor_v14
from eval_benchmark_full import resolve
from eval_bench_pergate import predict_gates
from eval_comprehensive import gamma2d
from scipy.ndimage import gaussian_filter

FLOOR = 8.0


def diag(p, g, dec, csf, mua_mus):
    """p,g: (N,) log10 rel. to this gate's peak, clamped at -FLOOR. dec = -g (depth below peak)."""
    out = {}
    reach = g > -FLOOR
    pr, gr = p[reach], g[reach]
    n = pr.size
    out["n_reach"] = int(n)
    if n < 50:
        return out
    # --- variance context: R2's denominator is the GT spread, which is NOT constant across gates
    ss_tot = float(((gr - gr.mean()) ** 2).sum())
    sse = float(((pr - gr) ** 2).sum())
    out["gt_std"] = float(gr.std())
    out["r2"] = float(1 - sse / ss_tot) if ss_tot > 0 else np.nan
    out["sse_per_voxel"] = sse / n
    # --- scale-free agreement: unaffected by the GT-variance dip
    out["rho"] = float(np.corrcoef(pr, gr)[0, 1])
    rk_p = np.argsort(np.argsort(pr)); rk_g = np.argsort(np.argsort(gr))
    out["spearman"] = float(np.corrcoef(rk_p, rk_g)[0, 1])
    vp, vg = pr.var(), gr.var(); mp, mg = pr.mean(), gr.mean()
    cov = float(((pr - mp) * (gr - mg)).mean())
    out["ccc"] = float(2 * cov / (vp + vg + (mp - mg) ** 2))     # Lin's concordance
    # --- amplitude
    e = pr - gr
    out["rmse"] = float(np.sqrt((e ** 2).mean()))
    out["mae"] = float(np.abs(e).mean())
    out["medae"] = float(np.median(np.abs(e)))
    out["p90ae"] = float(np.percentile(np.abs(e), 90))
    # --- systematic: split offset error from GAIN error (slope!=1 means compressed/expanded range)
    out["bias"] = float(e.mean())
    A = np.vstack([gr, np.ones_like(gr)]).T
    sl, ic = np.linalg.lstsq(A, pr, rcond=None)[0]
    out["slope"] = float(sl); out["intercept"] = float(ic)
    # variance ratio: <1 = the model UNDER-disperses (over-smooths), >1 = over-disperses
    out["var_ratio"] = float(vp / vg) if vg > 0 else np.nan
    # --- depth-resolved error: which decade band carries the error
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8)]:
        m = (dec[reach] >= lo) & (dec[reach] < hi)
        out[f"rmse_dec{lo}_{hi}"] = float(np.sqrt((e[m] ** 2).mean())) if m.sum() > 30 else np.nan
        out[f"frac_dec{lo}_{hi}"] = float(m.mean())
    # --- tissue-resolved
    c = csf[reach] > 0.5
    out["rmse_csf"] = float(np.sqrt((e[c] ** 2).mean())) if c.sum() > 30 else np.nan
    out["rmse_noncsf"] = float(np.sqrt((e[~c] ** 2).mean())) if (~c).sum() > 30 else np.nan
    # --- reachable-set geometry
    A_ = p > -FLOOR; B_ = g > -FLOOR
    ni = int((A_ & B_).sum())
    out["dice"] = 2 * ni / max(int(A_.sum()) + int(B_.sum()), 1)
    out["leak"] = int((A_ & ~B_).sum()) / max(int(B_.sum()), 1)
    out["miss"] = int((B_ & ~A_).sum()) / max(int(B_.sum()), 1)
    # see eval_bench_pergate.gate_metrics: DICE/leak degenerate when the model floods the dark zone
    # (saturated) or when the reachable set is too small to be a stable denominator (leak_illcond).
    out["saturated"] = float((p[~B_] > -FLOOR + 0.5).mean() > 0.98) if (~B_).sum() >= 50 else 0.0
    out["leak_illcond"] = float(int(B_.sum()) < 2000)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--elecs", nargs="+", default=["Cz", "C3"])
    ap.add_argument("--points", type=int, default=200000)
    ap.add_argument("--heads", nargs="+", default=None)
    ap.add_argument("--cohort", choices=["dev", "heldout", "all"], default="dev",
                    help="which part of the test split to score. 'dev' = the 20 bw/scb/sh heads that "
                         "ablation and architecture selection already used many times (a DEVELOPMENT "
                         "test set); 'heldout' = the 84 OASIS heads, which have never taken part in "
                         "any selection and carry the generalisation claim. The OASIS exclusion used "
                         "to be hard-coded here, which made the held-out benchmark impossible to run.")
    ap.add_argument("--gamma-anchor", choices=["srcpos", "gtpeak"], default="srcpos",
                    help="which x-slice gamma is evaluated on. srcpos (default, legacy) follows\nthe model's own source convention and is therefore NOT comparable across --src-ref settings;\ngtpeak anchors on the ground-truth gate-0 peak and is identical for every model.")
    ap.add_argument("--dump-slice", default=None,
                    help="directory in which to keep the 2-D source-plane slice that gamma already\ncomputes for every (model, seed, scene). Costs a write and no extra forward pass, and it is what\nfield-comparison figures need -- without it the only way to plot a baseline is to re-run the whole\nbenchmark. Use with --gamma-anchor gtpeak so every model shares one slice per scene.")
    ap.add_argument("--per-scene", default=None,
                    help="also write one row per (model, seed, head, electrode, gate) to this CSV. "
                         "The per-scene values are already computed -- the script only averages them "
                         "at the end -- so this adds no computation, and it is what the later "
                         "subject-level statistics need.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    P = {}
    for spec in a.models:
        nm, ck = spec.split("=", 1)
        preds = [build_predictor_v14(f, dev)[0] for f in resolve(ck)]
        P[nm] = preds
        print(f"  {nm}: {len(preds)} seed(s), family={preds[0].family}", flush=True)

    store = T.SceneStore(dev)
    _is_oasis = lambda h: h.startswith(("oa", "oas"))
    _keep = {"dev": lambda h: not _is_oasis(h), "heldout": _is_oasis, "all": lambda h: True}[a.cohort]
    te = [s for s in T.discover_scenes() if T.label_of(s["head"]) == os.environ.get("EVAL_SPLIT", "test")
          and s["tag"] in a.elecs and _keep(s["head"])]
    if a.heads:
        te = [s for s in te if s["head"] in a.heads]
    print(f"[diag] cohort={a.cohort} scenes={len(te)} heads={len({x['head'] for x in te})}\n", flush=True)

    # ---- incremental state / resume ---------------------------------------------------------
    # A held-out pass is 1596 scenes; at ~1.8 s of wall and up to 4x that in CPU per scene-model, a
    # multi-seed spec exceeds the login node's 4 CPU-hour RLIMIT and is SIGKILLed with exit 0 and no
    # traceback. Every (model, scene) result is therefore appended to a JSONL sidecar as it is
    # produced and reloaded on restart; the CSV named by --per-scene is still written whole at the
    # end, so the output contract is unchanged. JSONL rather than CSV for the sidecar because the
    # column set is only known after the first scene, and a partial file must not need a header.
    rows, done, seen_rows = [], set(), set()
    jl = (a.per_scene + ".jsonl") if a.per_scene else None
    if jl and os.path.exists(jl):
        ndup = 0
        with open(jl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue                       # truncated final line from a kill mid-write
                if r["model"] not in P:
                    continue
                # TWO DIFFERENT KEYS, and conflating them destroys data. `done` decides which SCENES
                # to skip, so it is keyed on (model, scene) -- a scene is finished as a whole.
                # The sidecar, however, holds ONE ROW PER GATE, so deduplicating `rows` on the
                # scene key throws away nine rows in ten. That is not hypothetical: the first
                # version of this dedup did exactly that, and three ViT panels came out of a
                # resumed run holding 504 rows with only gate 0 instead of 5040 across ten gates
                # ("4536 duplicate rows dropped" in the log). Row identity is (model, scene, GATE).
                sk = (r["model"], r["head"], r["electrode"])
                rk = (sk, r.get("gate"))
                if rk in seen_rows:
                    ndup += 1
                    continue
                rows.append(r); seen_rows.add(rk); done.add(sk)
        print(f"[diag] resuming: {len(done)} (model,scene) results already on disk"
              + (f" ({ndup} duplicate rows dropped)" if ndup else ""), flush=True)
    jlf = open(jl, "a") if jl else None

    @torch.no_grad()
    def gate_gamma(pred, sd, f7, dump_key=None):
        """Per-gate 2-D gamma (3%/2px) on a fixed slice -- the medical-physics acceptance criterion.
        Needs a dense 2-D field, so it is evaluated on a slice, not the sampled cloud.

        WHICH SLICE, AND WHY IT MATTERS. Anchoring on srcpos makes the slice MODEL-DEPENDENT: with
        --src-ref entry srcpos is the beam entry point, with source it is 15 voxels outside the
        scalp, and for lateral electrodes those are different anatomy. Measured on oas30026, the
        slice holds 370 tissue voxels under source against 4,082 under entry at T3 (11x), 447 vs
        4,720 at T4, and 8 of 19 electrodes differ by more than 25%. Under source a lateral slice
        merely grazes the scalp -- a thin, near-surface, easy sliver -- while under entry it cuts
        through the head. Comparing gamma across the two conventions therefore compares different
        problems, which silently invalidates every cross-family gamma number.

        --gamma-anchor gtpeak fixes the slice by the DATA (the x of the ground-truth gate-0 peak),
        so it is identical for every model on a scene. The default stays `srcpos` so that tables
        already computed are not silently mixed with re-runs; switch deliberately and re-run the
        whole comparison together.
        """
        vs = sd["vol_shape"]
        if a.gamma_anchor == "gtpeak":
            sx = int(np.unravel_index(np.argmax(f7[..., 0]), f7[..., 0].shape)[0])
        else:
            sp = sd["srcpos"].cpu().numpy().astype(int)
            sx = int(np.clip(sp[0], 0, int(vs[0]) - 1))
        msl = (sd["prop"][0, 1] > 0)[sx].cpu().numpy()
        if msl.sum() < 50:
            return {}
        ys, zs = np.where(msl)
        xyz = torch.tensor(np.stack([np.full_like(ys, sx), ys, zs], 1),
                           dtype=torch.float32, device=dev)
        pred.reset()
        pg = predict_gates(pred, sd, xyz).cpu().numpy()          # (M,10) log10 absolute
        out = {}
        for k in range(T.N_STEP):
            pk = float(f7[..., k].max())
            if pk <= 0:
                continue
            gsl = f7[sx][ys, zs, k]                              # linear GT on the slice
            G = np.zeros(msl.shape); P_ = np.zeros(msl.shape)
            G[ys, zs] = gsl
            P_[ys, zs] = np.clip(10.0 ** pg[:, k], 0, None)
            out[k] = float(gamma2d(G, P_, msl))

        # ---- optional: keep the 2-D field instead of throwing it away ---------------------------
        # This slice is already computed for gamma on every (model, seed, scene); saving it costs a
        # write and nothing else, and it is the field every comparison figure needs. Regenerating it
        # later would mean re-running the whole benchmark, so it is saved now.
        # Under --gamma-anchor gtpeak the slice is fixed by the DATA, so every model on a scene lands
        # on the same anatomy and the ground truth can be stored once and shared. sx is in every
        # filename so a convention mismatch is visible rather than silent.
        if dump_key is not None:
            dd, mdl, sidx, head, elec = dump_key
            os.makedirs(dd, exist_ok=True)
            gtf = os.path.join(dd, f"gt_{head}_{elec}_x{sx}.npz")
            if not os.path.exists(gtf):
                np.savez_compressed(
                    gtf, gt=f7[sx][ys, zs, :T.N_STEP].astype(np.float32),
                    ys=ys.astype(np.int16), zs=zs.astype(np.int16),
                    shape=np.asarray(msl.shape, np.int32), sx=np.int32(sx),
                    gate_max=np.asarray([float(f7[..., k].max()) for k in range(T.N_STEP)], np.float32))
            np.savez_compressed(
                os.path.join(dd, f"pred_{mdl}_s{sidx}_{head}_{elec}_x{sx}.npz"),
                pred=pg.astype(np.float16),      # log10 absolute, same convention as the maps
                sx=np.int32(sx), anchor=a.gamma_anchor)
        return out
    for si, s in enumerate(te):
        sd = store.get(s); vs = sd["vol_shape"]
        f7 = load(os.path.join(SIM7, f"fluence_{s['head']}_F810_{s['tag']}_r5_t10.mat"))
        idx = sd["valid_idx"].float()
        # deterministic point set: seeded per scene, so EVERY model -- including models scored in
        # separate runs/groups -- is measured on the exact same voxels. An unseeded randperm
        # made group-to-group comparisons carry an extra sampling difference.
        _g = torch.Generator(device=idx.device); _g.manual_seed(zlib.crc32((s['head'] + s['tag']).encode()) % (2**31))
        ii = idx[torch.randperm(idx.shape[0], device=idx.device, generator=_g)[:min(a.points, idx.shape[0])]]
        jj = ii.round().long().cpu().numpy()
        gt_lin = f7[jj[:, 0], jj[:, 1], jj[:, 2], :T.N_STEP]
        csf = sd["csf"][ii.round().long()[:, 0], ii.round().long()[:, 1],
                        ii.round().long()[:, 2]].cpu().numpy().astype(float)
        for nm, preds in P.items():
            if (nm, s["head"], s["tag"]) in done:
                continue                           # already present in the partial output
            for sidx, pred in enumerate(preds):
                gam = gate_gamma(pred, sd, f7,
                                 dump_key=None if not a.dump_slice else
                                 (a.dump_slice, nm, sidx, s["head"], s["tag"]))
                pred.reset()
                pg = predict_gates(pred, sd, ii).cpu().numpy()
                for k in range(T.N_STEP):
                    pk = float(f7[..., k].max())
                    if pk <= 0:
                        continue
                    lpk = math.log10(pk)
                    g = np.maximum(np.log10(np.clip(gt_lin[:, k], pk * 1e-12, None)) - lpk, -FLOOR)
                    p = np.maximum(pg[:, k] - lpk, -FLOOR)
                    dd = diag(p, g, -g, csf, None)
                    r = dict(model=nm, seed_idx=sidx, family=pred.family,
                             src_ref=pred.cfg.get("src_ref", "source"),
                             head=s["head"], electrode=s["tag"], gate=k,
                             # gate centre in ns. NOT T_ENC -- that is the NORMALISED time code
                             # the network consumes (-0.9 .. 0.9), not physical time. The 10
                             # gates span a 2 ns window in 0.2 ns steps, so gate k is centred at
                             # 0.1 + 0.2k ns, matching the t (ns) column of the V15 tables.
                             t_ns=0.1 + 0.2 * k)
                    r.update({mk: (float(v) if v is not None and np.isfinite(v) else None)
                              for mk, v in dd.items()})
                    gv = gam.get(k)
                    r["gamma"] = float(gv) if gv is not None and np.isfinite(gv) else None
                    rows.append(r)
                    if jlf is not None:
                        jlf.write(json.dumps(r) + "\n")
            if jlf is not None:
                jlf.flush()                        # one fsync per (model,scene), not per gate
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(te)}", flush=True)
    if jlf is not None:
        jlf.close()

    if a.per_scene:
        import csv as _csv
        os.makedirs(os.path.dirname(os.path.abspath(a.per_scene)), exist_ok=True)
        cols = sorted({c for r in rows for c in r})
        head_cols = ["model", "seed_idx", "family", "src_ref", "head", "electrode", "gate", "t_ns"]
        cols = head_cols + [c for c in cols if c not in head_cols]
        with open(a.per_scene, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"\nper-scene rows -> {a.per_scene}  ({len(rows)} rows)", flush=True)

    # Aggregate from the per-scene rows rather than from a running accumulator, so a run that was
    # killed and resumed produces exactly the same summary as one that never stopped -- the rows are
    # the state. None (non-finite) becomes nan, which nanmean/nanstd skip, matching the previous
    # behaviour of simply not appending those values.
    acc = {nm: {k: {} for k in range(T.N_STEP)} for nm in P}
    for r in rows:
        d = acc[r["model"]][r["gate"]]
        for mk, v in r.items():
            if mk in ("model", "seed_idx", "family", "src_ref", "head", "electrode", "gate", "t_ns"):
                continue
            d.setdefault(mk, []).append(np.nan if v is None else float(v))
    for nm in acc:                                 # drop metrics that are nan everywhere
        for k in acc[nm]:
            acc[nm][k] = {mk: v for mk, v in acc[nm][k].items() if not np.all(np.isnan(v))}

    out = {}
    for nm in P:
        out[nm] = {"n_seeds": len(P[nm]), "family": P[nm][0].family}
        for k in range(T.N_STEP):
            d = {mk: float(np.nanmedian(v) if mk in ("leak", "miss") else np.nanmean(v))
                 for mk, v in acc[nm][k].items() if v}
            d.update({mk + "_std": float(np.nanstd(v)) for mk, v in acc[nm][k].items() if v})
            out[nm][str(k)] = d
    groups = [("variance context", ["gt_std", "sse_per_voxel", "r2"]),
              ("scale-free", ["rho", "spearman", "ccc"]),
              ("amplitude", ["rmse", "mae", "medae", "p90ae"]),
              ("systematic", ["bias", "slope", "var_ratio"]),
              ("depth-resolved RMSE", ["rmse_dec0_2", "rmse_dec2_4", "rmse_dec4_6", "rmse_dec6_8"]),
              ("tissue", ["rmse_csf", "rmse_noncsf"]),
              ("set geometry", ["dice", "leak", "miss", "saturated", "leak_illcond"]),
              ("acceptance", ["gamma"])]
    for nm in P:
        for title, keys in groups:
            print("\n" + "=" * 100)
            print(f"### {nm} — {title}")
            print(f"{'gate':>5}{'t(ns)':>7}" + "".join(f"{k:>14}" for k in keys))
            print("-" * 100)
            for k in range(T.N_STEP):
                row = f"{k:>5}{0.1 + 0.2 * k:>7.1f}"
                for mk in keys:
                    v = out[nm][str(k)].get(mk)
                    row += (f"{v:>14.4f}" if v is not None and np.isfinite(v) else f"{'-':>14}")
                print(row)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
