#!/usr/bin/env python3
"""Batch per-electrode tissue-layer thickness over the 84-head held-out OASIS AD/HC cohort.

For each head: ensure the phantom->MNI transform exists (recover+persist if missing, via
pup_to_mni.get_transform -- MSE self-registration, phantom untouched), transport GRACE to MNI
and measure the 19-electrode layer thicknesses (electrode_layer_thickness.measure). Writes a
long per-(head,electrode) table + a per-electrode cohort summary, and a per-head QC line.
Flags (frontal sinus / sphenoid) are preserved, not corrected.
"""
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "viz"))
import repro_config as RC
import mni_normalize as MN
import pup_to_mni as P2M
import electrode_layer_thickness as ELT

META = os.path.join(RC.OASIS_METADATA_DIR, "selected_84_groups.csv")
OUTD = RC.PUP_MNI_DIR
LAYERS = ("skin", "muscle", "fat", "skull", "cancellous", "cortical")

def heads():
    return [(r["subject"].lower(), r["group"]) for r in csv.DictReader(open(META))]

def main():
    hs = heads()
    long_rows, per_head, failures = [], [], []
    perE = {n: {k: [] for k in LAYERS} for n in MN.NAMES}
    t0 = time.time()
    for i, (sid, grp) in enumerate(hs):
        try:
            _, status = P2M.get_transform(sid)          # recover+persist .tfm if missing
            grace, rows = ELT.measure(sid)
        except Exception as e:
            failures.append((sid, repr(e)[:140])); print(f"[{i+1}/84] FAIL {sid}: {e}", flush=True)
            continue
        skulls = [th["skull"] for _, th in rows if not th["flag"]]
        medskull = float(np.median(skulls)) if skulls else float("nan")
        nflag = sum(1 for _, th in rows if th["flag"])
        qc = "" if (2.0 <= medskull <= 8.5 and nflag <= 6) else "REVIEW"
        per_head.append((sid, grp, status, medskull, nflag, qc))
        print(f"[{i+1}/84] {sid} {grp:7s} {status:20s} medskull={medskull:4.1f} flags={nflag} {qc}",
              flush=True)
        for name, th in rows:
            long_rows.append([sid, grp, name] + [f"{th[k]:.2f}" for k in LAYERS] + [th["flag"]])
            if not th["flag"]:
                for k in LAYERS: perE[name][k].append(th[k])

    # ---- long per-(head, electrode) table ----
    with open(os.path.join(OUTD, "cohort_electrode_thickness.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "group", "electrode", "skin_mm", "muscle_mm", "fat_mm",
                    "skull_mm", "cancellous_mm", "cortical_mm", "flag"])
        w.writerows(long_rows)
    # ---- per-electrode cohort summary (reliable sites only) ----
    with open(os.path.join(OUTD, "cohort_electrode_thickness_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["electrode", "n_reliable"] +
                   [f"{k}_{s}" for k in LAYERS for s in ("mean", "sd")])
        for name in MN.NAMES:
            n = len(perE[name]["skull"])
            row = [name, n]
            for k in LAYERS:
                v = np.array(perE[name][k])
                row += ["" if n == 0 else f"{v.mean():.2f}",
                        "" if n < 2 else f"{v.std(ddof=1):.2f}"]
            w.writerow(row)
    # ---- per-head QC ----
    with open(os.path.join(OUTD, "cohort_electrode_thickness_qc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "group", "transform", "median_skull_mm", "n_flagged", "qc"])
        for r in per_head:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", r[4], r[5]])

    # ---- console cohort summary ----
    print("\n===== cohort per-electrode mean thickness (mm, reliable sites only) =====")
    print(f"{'elec':5s}{'n':>4s}{'skin':>7s}{'muscle':>7s}{'fat':>6s}{'skull':>7s}{'canc':>6s}{'cort':>6s}")
    allk = {k: [] for k in LAYERS}
    for name in MN.NAMES:
        n = len(perE[name]["skull"])
        m = {k: (np.mean(perE[name][k]) if n else float('nan')) for k in LAYERS}
        for k in LAYERS: allk[k] += perE[name][k]
        print(f"{name:5s}{n:>4d}{m['skin']:7.1f}{m['muscle']:7.1f}{m['fat']:6.1f}"
              f"{m['skull']:7.1f}{m['cancellous']:6.1f}{m['cortical']:6.1f}")
    print("-" * 44)
    print(f"{'ALL':5s}{'':>4s}{np.mean(allk['skin']):7.1f}{np.mean(allk['muscle']):7.1f}"
          f"{np.mean(allk['fat']):6.1f}{np.mean(allk['skull']):7.1f}"
          f"{np.mean(allk['cancellous']):6.1f}{np.mean(allk['cortical']):6.1f}")
    recov = sum(1 for r in per_head if r[2].startswith("recovered"))
    review = [r[0] for r in per_head if r[5] == "REVIEW"]
    print(f"\nheads done: {len(per_head)}/84  (transforms recovered: {recov}, loaded: "
          f"{len(per_head)-recov})  failures: {len(failures)} {failures}")
    print(f"QC REVIEW heads ({len(review)}): {review}")
    print(f"elapsed {(time.time()-t0)/60:.1f} min")
    print(f"wrote {OUTD}/cohort_electrode_thickness{{,_summary,_qc}}.csv")

if __name__ == "__main__":
    main()
