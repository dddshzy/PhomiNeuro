#!/usr/bin/env python3
"""Per-gate ABSORBED-ENERGY conservation + R2, broken down by head SOURCE (bw / scb / sh / oasis).

Energy ratio (per gate k, per scene) = sum_voxels(10^pred_k * mu_a) / sum_voxels(GT_k * mu_a),
summed over ALL tissue voxels (not a sample) -- so it is the physically meaningful statement
"does the surrogate deposit the same total dose as the Monte-Carlo reference". 1.0 = conserved,
<1 = under-deposits, >1 = over-deposits. This is orthogonal to R2: a model can have a good R2 in
the reachable set yet lose most of the absorbed energy (or invent it) in the tail.

Predictions are made on EVERY tissue voxel, in chunks. Ratios are aggregated across scenes by
MEDIAN (a ratio with a small denominator on one scene otherwise dominates a mean -- that defect
produced the 524.8 artefact in the V15 tables).

  INR_SPLIT=v16_split EVAL_SPLIT=test python eval_energy_bycat.py --gpu 0 \
      --models ours=inr_checkpoints_v11/inr_v16ls_lin03dmid70_base.pt \
      --oasis oas30057 oas30187 --out out.json
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


def category(head):
    if head.startswith(("oa", "oas")):
        return "oasis"
    if head.startswith("bw"):
        return "bw"
    if head.startswith("scb"):
        return "scb"
    return "sh"


def r2(p, g):
    ss = ((g - g.mean()) ** 2).sum()
    return float(1 - ((p - g) ** 2).sum() / ss) if ss > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--elecs", nargs="+", default=["Cz"])
    ap.add_argument("--oasis", nargs="+", default=None, help="restrict oasis to these heads")
    ap.add_argument("--cats", nargs="+", default=None,
                    help="restrict to these head categories (bw scb sh oasis). The ablation table "
                         "consumes bw+scb+sh only, so running oasis too costs 5.2x for nothing.")
    ap.add_argument("--tissue", choices=["all", "brain"], default="all",
                    help="voxels the energy sums run over. 'brain' keeps only GM and WM.\n"
                         "WHY IT MATTERS: measured on the held-out cohort, gate 0 deposits 0.0% of "
                         "its energy in brain -- the light has not arrived yet, and the tPSF "
                         "time-to-peak is 0.1 ns in scalp against 0.4-0.55 ns in GM. Gate 0 "
                         "nonetheless carries 79% of the ALL-TISSUE absorbed energy, so an "
                         "all-tissue dose figure is 97% a statement about scalp, skull, fat and "
                         "muscle. For transcranial photobiomodulation that energy is loss, not "
                         "dose. Restricted to brain, gates 1-3 carry 74% of the deposited energy.")
    ap.add_argument("--floor", type=float, default=8.0, help="D for the R2 reachable set")
    ap.add_argument("--points", type=int, default=200000, help="sampled points for R2 (energy uses ALL voxels)")
    ap.add_argument("--chunk", type=int, default=200000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--partial", default=None,
                    help="append one JSON line per (model,scene) here and RESUME from it. "
                         "Without this a kill loses everything, because --out is written once at the end.")
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu); dev = torch.device(f"cuda:{a.gpu}")
    split = os.environ.get("EVAL_SPLIT", "test")

    P = {}
    for spec in a.models:
        nm, paths = spec.split("=", 1)
        preds = [build_predictor_v14(p, dev)[0] for p in paths.split(",")]
        P[nm] = preds
        print(f"  {nm}: {len(preds)} seed(s), family={preds[0].family}", flush=True)

    scenes = [s for s in T.discover_scenes() if T.label_of(s["head"]) == split and s["tag"] in a.elecs]
    if a.oasis is not None:
        keep = set(a.oasis)
        scenes = [s for s in scenes if category(s["head"]) != "oasis" or s["head"] in keep]
    if a.cats is not None:
        scenes = [s for s in scenes if category(s["head"]) in set(a.cats)]
    cats = sorted(set(category(s["head"]) for s in scenes))
    print(f"[energy] split={split} scenes={len(scenes)} cats={cats}", flush=True)

    # ---- resume ----------------------------------------------------------------------------
    # This loop costs ~11 s of CPU per scene, so a full 1976-scene pass needs ~6 CPU-hours -- more
    # than the 4-hour RLIMIT_CPU on the login node, which kills the process with exit 0 and no
    # traceback. Writing only at the end meant every such kill discarded hours of finished scenes.
    # Each (model, scene) result is now appended as it is produced and reloaded on restart, so a
    # kill costs at most the scene in flight. The same file doubles as the per-scene long table.
    done, rows = set(), []
    if a.partial and os.path.exists(a.partial):
        ndup = 0
        with open(a.partial) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue                      # truncated last line from a kill mid-write
                if r["model"] not in P:
                    continue
                k = (r["model"], r["head"], r["tag"])
                # DEDUPLICATE. `done` is a set but `rows` was a list, so a scene appearing twice in
                # the partial file was counted twice in every aggregate below -- the median, the
                # percentiles and the R2 mean are all weighted by row multiplicity. That is not
                # hypothetical: two supervisors on different login nodes appended to the same
                # partial (login nodes do not share a process table, so neither could see the
                # other), giving energy_v18hero3_s0.jsonl 1223 rows for 504 scenes and s1 815 for
                # 504, and n_scenes was reported as 521. The duplicate rows are byte-identical, so
                # nothing is lost by keeping the first; what is fixed is the weighting.
                if k in done:
                    ndup += 1
                    continue
                rows.append(r); done.add(k)
        print(f"[energy] resuming: {len(done)} (model,scene) results already on disk"
              + (f" ({ndup} duplicate rows dropped)" if ndup else ""), flush=True)

    store = T.SceneStore(dev)
    for si, s in enumerate(scenes):
        c = category(s["head"])
        todo = [nm for nm in P if (nm, s["head"], s["tag"]) not in done]
        if not todo:
            continue                              # already present in the partial output
        sd = store.get(s); vs = sd["vol_shape"]
        f7 = load(os.path.join(SIM7, f"fluence_{s['head']}_F810_{s['tag']}_r5_t10.mat"))   # (X,Y,Z,10)
        mua = sd["prop"][0, 0]                                       # (X,Y,Z) absorption
        idx_all = sd["valid_idx"]                                    # every tissue voxel
        if a.tissue == "brain":
            # Classify on (mu_a, mu_s) TOGETHER, never mu_a alone: in the F810 nine-tissue table GM
            # and skin/scalp share mu_a = 0.028 exactly and are separated only by mu_s (7.3 vs 6.4).
            # Selecting on absorption would silently pull in a quarter of the head.
            _j = idx_all.round().long()
            _ua = sd["prop"][0, 0][_j[:, 0], _j[:, 1], _j[:, 2]]
            _us = sd["prop"][0, 1][_j[:, 0], _j[:, 1], _j[:, 2]]
            keep = torch.zeros_like(_ua, dtype=torch.bool)
            for _a0, _s0 in ((0.028, 7.3), (0.092, 38.0)):           # GM, WM
                keep |= (_ua - _a0).abs().lt(1e-5) & (_us - _s0).abs().lt(1e-3)
            if int(keep.sum()) == 0:
                print(f"  [skip] {s['head']}/{s['tag']}: no GM/WM voxel matched", flush=True)
                continue
            idx_all = idx_all[keep]
        jj = idx_all.round().long().cpu().numpy()
        mua_v = mua[jj[:, 0], jj[:, 1], jj[:, 2]].float()            # (Nvox,)
        gt_all = torch.from_numpy(np.ascontiguousarray(
            f7[jj[:, 0], jj[:, 1], jj[:, 2], :T.N_STEP])).to(dev).float()      # (Nvox,10) LINEAR
        # denominator: total absorbed energy per gate from the MC reference
        den = (gt_all * mua_v[:, None]).sum(0)                                  # (10,)
        # R2 sample: the crc32-seeded subset, same convention as eval_bench_pergate
        n = min(a.points, idx_all.shape[0])
        _g = torch.Generator(device=idx_all.device)
        _g.manual_seed(zlib.crc32((s["head"] + s["tag"]).encode()) % (2 ** 31))
        sub = torch.randperm(idx_all.shape[0], device=idx_all.device, generator=_g)[:n]

        for nm in todo:
            preds = P[nm]
            num = torch.zeros(T.N_STEP, device=dev)
            r2_g = {k: [] for k in range(T.N_STEP)}
            for pred in preds:
                pred.reset()
                # ---- energy over ALL tissue voxels, chunked ----
                nsum = torch.zeros(T.N_STEP, device=dev)
                for i in range(0, idx_all.shape[0], a.chunk):
                    xyz = idx_all[i:i + a.chunk].float()
                    pg = predict_gates(pred, sd, xyz)                            # (m,10) log10
                    nsum += (torch.pow(10.0, pg) * mua_v[i:i + a.chunk, None]).sum(0)
                num += nsum / len(preds)
                # ---- R2 on the sampled subset, per gate, reachable set only ----
                pg_s = predict_gates(pred, sd, idx_all[sub].float()).cpu().numpy()   # (n,10)
                gt_s = gt_all[sub].cpu().numpy()
                for k in range(T.N_STEP):
                    pk = float(f7[..., k].max())
                    if pk <= 0:
                        continue
                    lpk = math.log10(pk)
                    g = np.maximum(np.log10(np.clip(gt_s[:, k], pk * 1e-12, None)) - lpk, -a.floor)
                    p = np.maximum(pg_s[:, k] - lpk, -a.floor)                   # common pred-floor
                    m = g > -a.floor
                    if m.sum() >= 50:
                        r2_g[k].append(r2(p[m], g[m]))
            # e_num / e_den are stored ALONGSIDE the ratio, not instead of it. A cumulative dose
            # curve needs sum_{g<=t} E_pred / sum_{g<=t} E_gt, which cannot be rebuilt from ratios
            # alone -- the first attempt at that figure had to recover the denominators through a
            # separate ground-truth pass because only the quotient had been kept.
            row = dict(model=nm, head=s["head"], tag=s["tag"], cat=c, tissue=a.tissue,
                       energy=[float(num[k] / den[k]) if float(den[k]) > 0 else None
                               for k in range(T.N_STEP)],
                       e_num=[float(num[k]) for k in range(T.N_STEP)],
                       e_den=[float(den[k]) for k in range(T.N_STEP)],
                       r2=[r2_g[k] for k in range(T.N_STEP)])
            rows.append(row)
            if a.partial:
                with open(a.partial, "a") as fh:      # append+flush per scene: a kill loses one scene
                    fh.write(json.dumps(row) + "\n")
        if (si + 1) % 5 == 0:
            print(f"  {si+1}/{len(scenes)}", flush=True)

    # Aggregate from the per-scene rows, so a resumed run and an uninterrupted one give the same
    # numbers -- the rows are the state, the in-memory accumulator no longer is.
    acc = {nm: {c: {k: {"energy": [], "R2": []} for k in range(T.N_STEP)} for c in cats} for nm in P}
    for r in rows:
        if r["cat"] not in cats:
            continue
        d = acc[r["model"]][r["cat"]]
        for k in range(T.N_STEP):
            if r["energy"][k] is not None:
                d[k]["energy"].append(r["energy"][k])
            d[k]["R2"].extend(r["r2"][k])

    out = {}
    for nm in P:
        out[nm] = {}
        for c in cats:
            out[nm][c] = {"n_scenes": len(acc[nm][c][0]["energy"]), "per_gate": {}}
            for k in range(T.N_STEP):
                e = acc[nm][c][k]["energy"]; r = acc[nm][c][k]["R2"]
                out[nm][c]["per_gate"][k] = {
                    "energy_median": float(np.median(e)) if e else float("nan"),
                    "energy_p25": float(np.percentile(e, 25)) if e else float("nan"),
                    "energy_p75": float(np.percentile(e, 75)) if e else float("nan"),
                    "R2": float(np.mean(r)) if r else float("nan")}
    for nm in out:
        for c in out[nm]:
            print(f"\n### {nm} / {c}  ({out[nm][c]['n_scenes']} scenes)")
            print(f"  {'gate':>4}{'R2':>9}{'E_ratio':>10}{'[p25':>9}{'p75]':>9}")
            for k in range(T.N_STEP):
                d = out[nm][c]["per_gate"][k]
                print(f"  {k:>4}{d['R2']:>9.4f}{d['energy_median']:>10.4f}{d['energy_p25']:>9.4f}{d['energy_p75']:>9.4f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
