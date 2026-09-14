#!/usr/bin/env python3
"""Per-gate scoring on a support fixed by the 2 ns TIME-INTEGRATED field, shared by every gate.

WHY THE SUPPORT MOVES TO THE TIME INTEGRAL.
The legacy convention keeps, at each gate, the voxels within D decades of THAT gate's own peak. Every
gate is then scored on a different set: measured on OASIS/E6 the per-gate reachable set is 0.4% of
tissue at gate 0 and ~8% at gate 4. A per-gate curve built that way compares numbers computed on
different supports, which is precisely the comparison a temporal claim must not make. Here the
support is defined once,

    integ = sum_k Phi_k        support = tissue AND integ > max(integ) * 10^-D      (D = 7)

and reused at every gate -- 15.2% of tissue. Temporal shape becomes comparable because the support
no longer moves.

DO NOT CALL THIS THE STEADY-STATE OR CW FIELD. For a linear medium the CW response equals the time
integral of the impulse response, so the identification is right in principle, but this integral is
TRUNCATED at 2 ns. Aggregated over all voxels the missing tail is only 0.14% of the deposited energy,
which is what makes the shorthand tempting; PER VOXEL it is not small. Extrapolating the last two
gates geometrically, the tail beyond 2 ns is a median 5.4% of the integrated value over the support
and 29.2% over the deepest 10% of it, and 7.2% of supported voxels (23.0% of the deepest tenth) have
their peak gate at 8 or 9, i.e. are still near or before their maximum when the window closes. Deep
tissue is exactly where this work aims, so the honest name is "2 ns time-integrated fluence". The
same bias shifts the support boundary by about 0.15 decades (an effective D of ~6.85); it is
identical for every model, so comparisons are unaffected, but the boundary should be quoted as such.

WHY D = 7 AND NOT 8. At D = 8 a quarter of the support is still floored at gate 5 (25.9% against
7.2%) and 3.1% at gate 9, so late-gate R2 mixes amplitude accuracy with censoring. At D = 7 the late
gates are essentially floor-free (0.24% at gate 9) and read as pure amplitude. This also matches the
convention this project already adopted for time-integrated fields.

THE ZEROS ARE PHYSICS, NOT A DEFECT. Inside the support a gate's fluence is frequently EXACTLY zero
-- 96.1% of supported voxels at gate 0, 82.3% at gate 1, falling to 0.24% at gate 9 (measured, 8
scenes) -- because the light has not yet arrived. Those voxels are set to that gate's own floor, the
same clamp the prediction may receive.

READ THE EARLY-GATE R2 ACCORDINGLY. With 96% of the support at a constant floor, gate-0 R2 is
dominated by whether the model puts the floor where the ground truth has it; only the remaining few
percent probe amplitude. Early-gate R2 is therefore mostly an ARRIVAL-TIME score and late-gate R2
mostly an amplitude score. The reported frac_floor column exists so a reader can tell which regime a
gate is in.

TWO READINGS OF THE PREDICTION are reported side by side, because the clamp is a modelling choice
rather than a fact: r2 clamps the prediction at the floor (consistent with left-censored targets),
r2_raw does not (predicting far below the floor then costs). Our own model is the one most exposed
to that choice, since its floor hinge pushes predictions down, so both must be shown.

    INR_SPLIT=v16_split python eval_steadymask.py --gpu 0 --elecs Fp1 F7 Cz T4 Pz O1 \
        --models ours=... dynunet=... --out steady.json
"""
import os, sys, math, json, argparse, zlib
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "data_expansion"),
          os.path.join(os.path.dirname(HERE), "evaluation")):
    sys.path.insert(0, p)
import train_inr_v11 as T
from eval_v11_level2 import load, SIM7
from eval_benchmark_v14cw import build_predictor_v14
from eval_bench_pergate import predict_gates

FLOOR = 7.0        # D for the time-integrated support; see --floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--elecs", nargs="+", default=["Fp1", "F7", "Cz", "T4", "Pz", "O1"])
    ap.add_argument("--points", type=int, default=50000)
    ap.add_argument("--floor", type=float, default=FLOOR,
                    help="decades below the 2 ns time-integrated peak that define the support, and "
                         "the per-gate clamp. D=7 rather than 8: at D=8 a quarter of the support is "
                         "still floored at gate 5 (25.9%% vs 7.2%%), so late-gate R2 mixes amplitude "
                         "accuracy with censoring; at D=7 the late gates are essentially floor-free "
                         "and read as pure amplitude. Matches the project convention for "
                         "time-integrated fields.")
    ap.add_argument("--gate-floor", type=float, default=None,
                    help="decades of dynamic range kept WITHIN each gate, independent of --floor. "
                         "--floor picks the voxels (from the 2 ns time integral); this picks how far "
                         "below that gate's own peak a value is still resolved. They answer different "
                         "questions and need not be equal. Defaults to --floor.")
    ap.add_argument("--cohort", choices=["heldout", "dev"], default="heldout")
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only (pilot runs)")
    ap.add_argument("--partial", default=None,
                    help="per-scene jsonl written as the run proceeds and re-read on restart. This "
                         "node SIGKILLs a process at 4 CPU-hours and a 504-scene sweep exceeds that, "
                         "so without resume the runner would restart from scratch forever.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu); dev = torch.device(f"cuda:{a.gpu}")

    P = {}
    for spec in a.models:
        nm, ck = spec.split("=", 1)
        P[nm] = build_predictor_v14(ck, dev)[0]

    # Cohort selection must match diag_pergate.py exactly, or the tables are computed on different
    # scene sets. Two filters are needed, not one: the OASIS/non-OASIS split picks the cohort, and
    # T.label_of(head) == EVAL_SPLIT picks the held-out partition WITHIN it. Dropping the second
    # silently widened the held-out sweep from 84 heads to all 209 OASIS heads in the dataset
    # (504 -> 1254 scenes), including heads used for training.
    split = os.environ.get("EVAL_SPLIT", "test")
    is_ho = lambda h: h.startswith(("oa", "oas"))
    keep = is_ho if a.cohort == "heldout" else (lambda h: not is_ho(h))
    scenes = [s for s in T.discover_scenes()
              if T.label_of(s["head"]) == split and s["tag"] in a.elecs and keep(s["head"])]
    if a.limit:
        scenes = scenes[:a.limit]
    print(f"[steady] cohort={a.cohort} split={split} scenes={len(scenes)} "
          f"heads={len({x['head'] for x in scenes})} support=integ D={a.floor} "
          f"gate_D={a.gate_floor if a.gate_floor is not None else a.floor} models={list(P)}", flush=True)

    D = a.floor                       # support: 2 ns time-integrated peak minus D
    Dg = a.gate_floor if a.gate_floor is not None else D    # per-gate dynamic range
    # Resume from valid partial records when available.
    done = set(); rows = []; ndup = 0
    if a.partial and os.path.exists(a.partial):
        for ln in open(a.partial):
            try: r = json.loads(ln)
            except Exception: continue
            k = (r["head"], r["tag"], r["model"])
            # DEDUPLICATE, for the same reason as diag_pergate and eval_energy_bycat: `done` was a
            # set but `rows` was a list, so two processes appending to one partial file (which does
            # happen -- login nodes share Lustre but not a process table) silently double-count
            # scenes in every aggregate and inflate the row count the supervisors test for.
            if k in done:
                ndup += 1
                continue
            rows.append(r); done.add(k)
        print(f"[resume] {len(rows)} scene-model rows from {os.path.basename(a.partial)}"
              + (f" ({ndup} duplicates dropped)" if ndup else ""), flush=True)
    pf = open(a.partial, "a") if a.partial else None
    KEYS = ("r2", "r2_common", "r2_raw", "rmse", "mae", "medae", "p90ae", "bias", "slope",
            "intercept", "var_ratio", "rho", "spearman", "ccc", "gt_std", "dice", "miss", "tnr",
            "frac_floor", "n_sampled", "support_frac", "rmse_csf", "rmse_noncsf",
            "peakband_logerr", "peakband_bias", "peakband_n",
            "rmse_dec0_2", "rmse_dec2_4", "rmse_dec4_6", "rmse_dec6_7", "rmse_dec6_8",
            "frac_dec0_2", "frac_dec2_4", "frac_dec4_6", "frac_dec6_7", "frac_dec6_8")
    acc = {nm: {k: {m: [] for m in KEYS} for k in range(T.N_STEP)} for nm in P}
    store = T.SceneStore(dev)
    for si, s in enumerate(scenes):
        sd = store.get(s)
        f7 = load(os.path.join(SIM7, f"fluence_{s['head']}_F810_{s['tag']}_r5_t10.mat"))[..., :T.N_STEP]
        idx = sd["valid_idx"]
        jj = idx.round().long().cpu().numpy()
        gt = np.ascontiguousarray(f7[jj[:, 0], jj[:, 1], jj[:, 2], :])          # (N,10) linear

        # ---- support: the 2 ns time integral, one set shared by all ten gates ----
        steady = gt.sum(axis=1)
        reach = steady > float(steady.max()) * 10.0 ** -D
        if reach.sum() < 200:
            continue
        rid = np.nonzero(reach)[0]
        supp_frac = float(rid.size) / float(reach.size)      # support / tissue, before sampling
        g = np.random.default_rng(zlib.crc32((s["head"] + s["tag"]).encode()) % (2 ** 31))
        sel = rid if rid.size <= a.points else g.choice(rid, a.points, replace=False)
        xs = idx[torch.from_numpy(sel).to(idx.device)].float()
        gsel = gt[sel]                                                          # (M,10)
        csf_np = sd["csf"].reshape(-1)[
            (jj[:, 0] * int(sd["vol_shape"][1]) + jj[:, 1]) * int(sd["vol_shape"][2]) + jj[:, 2]
        ].cpu().numpy()[sel] > 0.5 if "csf" in sd else np.zeros(len(sel), bool)

        for nm, pred in P.items():
            if (s["head"], s["tag"], nm) in done:
                continue
            pred.reset()
            with torch.no_grad():
                pg = predict_gates(pred, sd, xs).cpu().numpy()                  # (M,10) log10
            for k in range(T.N_STEP):
                pk = float(f7[..., k].max())
                if pk <= 0:
                    continue
                lpk = math.log10(pk)
                # SELF floor: clamp each side at ITS OWN gate peak minus D, instead of putting the
                # prediction on the ground truth's floor. Both peaks are taken over the SAME sampled
                # points; using the full-volume GT peak against a sampled prediction peak would make
                # the two sides incomparable by construction. A model whose gate peak sits below the
                # ground truth's then gets a lower floor, so its deep predictions are no longer lifted
                # onto the GT floor -- the reading stops crediting a model for a truncation it did not
                # earn, at the price of no longer being a common scale.
                # One record per scene and gate is shared by the accumulator and resume file.
                #
                # EVERY metric below is computed on the SELF-floored pair (Gs, Ps). That is the whole
                # point of the convention: the prediction is clamped by its own gate peak, which is
                # available at deployment, instead of by the ground truth's, which is not. The change
                # therefore does NOT stop at R2 -- RMSE, medAE, the decade-band errors, DICE and the
                # regression diagnostics all read clamped values and all move with it. Metrics that
                # never touch a floor (gamma, absorbed energy, peak error, top-K relative error) are
                # computed elsewhere and are deliberately not duplicated here.
                gsl_k, pgk, csf_k = gsel[:, k], pg[:, k], csf_np
                gs = np.log10(np.clip(gsl_k, pk * 10.0 ** -30, None))
                gfl, pfl = gs.max() - Dg, pgk.max() - Dg      # each side's OWN floor
                Gs = np.maximum(gs, gfl)
                Ps = np.maximum(pgk, pfl)
                sst = float(((Gs - Gs.mean()) ** 2).sum())
                if sst <= 0:
                    continue
                e = Ps - Gs
                dec = Gs.max() - Gs                                 # depth below this gate's GT peak
                A_, B_ = Ps > pfl + 1e-9, Gs > gfl + 1e-9
                ni = int((A_ & B_).sum())
                rk_p = np.argsort(np.argsort(Ps)); rk_g = np.argsort(np.argsort(Gs))
                vp, vg = float(Ps.var()), float(Gs.var())
                mp, mg = float(Ps.mean()), float(Gs.mean())
                sl, ic = np.linalg.lstsq(np.vstack([Gs, np.ones_like(Gs)]).T, Ps, rcond=None)[0]
                rec = {"r2": 1.0 - float((e ** 2).sum()) / sst,
                       "rmse": float(np.sqrt((e ** 2).mean())),
                       "mae": float(np.abs(e).mean()),
                       "medae": float(np.median(np.abs(e))),
                       "p90ae": float(np.percentile(np.abs(e), 90)),
                       "bias": float(e.mean()),
                       "slope": float(sl), "intercept": float(ic),
                       "var_ratio": float(vp / vg) if vg > 0 else float("nan"),
                       "rho": float(np.corrcoef(Ps, Gs)[0, 1]),
                       "spearman": float(np.corrcoef(rk_p, rk_g)[0, 1]),
                       "ccc": float(2 * float(((Ps - mp) * (Gs - mg)).mean())
                                    / (vp + vg + (mp - mg) ** 2)),
                       "gt_std": float(Gs.std()),
                       "dice": 2 * ni / max(int(A_.sum()) + int(B_.sum()), 1),
                       "miss": int((B_ & ~A_).sum()) / max(int(B_.sum()), 1),
                       "frac_floor": float((~B_).mean()),
                       "n_sampled": int(Gs.size),          # points actually scored (<= --points)
                       "support_frac": supp_frac}          # support voxels / tissue voxels
                # the two alternative readings, kept for the robustness note only
                Gc = np.maximum(gs - gs.max(), -Dg)
                rec["r2_common"] = 1.0 - float(((np.maximum(pgk - gs.max(), -Dg) - Gc) ** 2).sum()) \
                    / max(float(((Gc - Gc.mean()) ** 2).sum()), 1e-12)
                rec["r2_raw"] = 1.0 - float(((pgk - gs.max() - Gc) ** 2).sum()) \
                    / max(float(((Gc - Gc.mean()) ** 2).sum()), 1e-12)
                if (~B_).sum() >= 50:
                    rec["tnr"] = float((Ps[~B_] <= pfl + 1e-9).mean())
                if csf_k.any() and (~csf_k).any():
                    if csf_k.sum() > 30:
                        rec["rmse_csf"] = float(np.sqrt((e[csf_k] ** 2).mean()))
                    if (~csf_k).sum() > 30:
                        rec["rmse_noncsf"] = float(np.sqrt((e[~csf_k] ** 2).mean()))
                # ROBUST PEAK FIDELITY. The existing peak_logerr in eval_v17_targeted reads the
                # SINGLE brightest ground-truth voxel, which sits at the beam entry -- the steepest
                # point in the field and a genuine singularity of the forward problem. One scene
                # (sh001/Cz) misses it by 6.1 decades while the 380-scene mean is 0.17, so that
                # statistic carries enormous per-scene variance and cannot support a claim on its
                # own. Taking the MEDIAN over the brightest 0.2% of the support is insensitive to
                # that one voxel while still describing the region the peak lives in.
                # Both signed and unsigned are kept: the magnitude ranks models, the sign says
                # whether the bright region is under- or over-predicted, which |.| destroys.
                nb = max(30, int(round(0.002 * Gs.size)))
                if Gs.size > nb:
                    top = np.argpartition(Gs, -nb)[-nb:]
                    rec["peakband_logerr"] = float(np.median(np.abs(e[top])))
                    rec["peakband_bias"] = float(np.median(e[top]))
                    rec["peakband_n"] = int(nb)
                # decade bands stop at D, so the last band is 6-7 rather than the legacy 6-8
                for lo, hi in ((0, 2), (2, 4), (4, 6), (6, int(round(Dg)))):
                    m = (dec >= lo) & (dec < hi)
                    rec[f"frac_dec{lo}_{hi}"] = float(m.mean())
                    if m.sum() > 30:
                        rec[f"rmse_dec{lo}_{hi}"] = float(np.sqrt((e[m] ** 2).mean()))
                for m_, v_ in rec.items():
                    acc[nm][k][m_].append(v_)
                if pf is not None:
                    pf.write(json.dumps({"head": s["head"], "tag": s["tag"], "model": nm,
                                         "gate": k, **rec}) + "\n")
        if pf is not None:
            pf.flush()
        if (si + 1) % 20 == 0:
            print(f"  {si+1}/{len(scenes)}", flush=True)

    for r in rows:                      # resumed rows join the freshly measured ones
        if r["model"] not in acc:
            continue
        for m in KEYS:
            if m in r:
                acc[r["model"]][int(r["gate"])][m].append(r[m])
    if pf is not None:
        pf.close()
    # var_ratio is a RATIO (var_pred / var_gt) and its denominator can be tiny on an individual
    # scene, so a mean over scenes is dominated by that scene -- the same failure that once reported
    # an energy ratio of 524.8 when the true value was 0.47. Bounded quantities (dice, tnr, miss,
    # frac_*) are unaffected and keep the mean.
    RATIO = {"var_ratio", "slope"}
    out = {}
    for nm in P:
        out[nm] = {"per_gate": {
            k: {m: float(np.median(v) if m in RATIO else np.mean(v)) for m, v in d.items() if v}
            for k, d in acc[nm].items()}}
    for nm in out:
        pgd = out[nm]["per_gate"]
        gm = lambda m: float(np.mean([pgd[k][m] for k in pgd if m in pgd[k]]))
        print(f"\n### {nm}")
        print(f"  {'gate':>5}{'R2(self)':>10}{'R2(com)':>9}{'RMSE':>8}{'medAE':>8}"
              f"{'bias':>8}{'slope':>8}{'DICE':>8}{'TNR':>8}{'floor%':>8}")
        for k in sorted(pgd):
            d = pgd[k]
            print(f"  {k:>5}{d.get('r2',np.nan):10.4f}{d.get('r2_common',np.nan):9.4f}"
                  f"{d.get('rmse',np.nan):8.3f}{d.get('medae',np.nan):8.3f}"
                  f"{d.get('bias',np.nan):8.3f}{d.get('slope',np.nan):8.3f}"
                  f"{d.get('dice',np.nan):8.3f}{d.get('tnr',np.nan):8.3f}"
                  f"{d.get('frac_floor',np.nan)*100:8.1f}")
        print(f"  {'MEAN':>5}{gm('r2'):10.4f}{gm('r2_common'):9.4f}{gm('rmse'):8.3f}"
              f"{gm('medae'):8.3f}{gm('bias'):8.3f}{gm('slope'):8.3f}{gm('dice'):8.3f}"
              f"{gm('tnr'):8.3f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
