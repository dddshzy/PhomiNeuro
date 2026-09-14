"""Derive the no-CSF-split deposition table for V18 hero3 from the CSF-split one already computed.

`fix_feature_correlation` (viz/fig_v17_update0801_0803.py:236) reads cohort_energy_deposition.csv
-- gm_frac / wm_frac / other_frac, where "other" means everything that is not GM or WM and therefore
INCLUDES CSF. `fig_parenchyma_anova` reads the CSF-split table instead. Only the split table was run
under hero3, but the two are related by an exact identity, so the second table needs no new inference:

    gm_frac    identical
    wm_frac    identical
    other_frac = csf_frac + other_frac(split)

Measured on the V17 pair (both tables produced by the same run, 1596 rows): gm and wm are BIT-identical
(max|delta| = 0.000e+00) and the "other" identity holds to 1.0e-06, which is the CSV's own six-decimal
rounding. That measurement is re-run here as a self-test with --selftest before anything is written,
so the identity is verified on this machine rather than quoted from a note.

    python derive_dep_nocsf_v18h3.py --selftest      # V17 round-trip only, writes nothing
    python derive_dep_nocsf_v18h3.py                 # self-test, then write the hero3 table
"""
import argparse, csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import repro_config as RC
PUP = str(RC.PUP_MNI_DIR)
COLS = ["subject", "group", "electrode", "gm_frac", "wm_frac", "other_frac", "n_vox"]
TOL = 2e-6          # the six-decimal CSV rounding of csf_frac + other_frac


def derive(rows):
    """CSF-split rows -> no-split rows. Formatting matches the producer (6 decimals on fractions)."""
    out = []
    for r in rows:
        out.append({"subject": r["subject"], "group": r["group"], "electrode": r["electrode"],
                    "gm_frac": r["gm_frac"], "wm_frac": r["wm_frac"],
                    "other_frac": f"{float(r['csf_frac']) + float(r['other_frac']):.6f}",
                    "n_vox": r["n_vox"]})
    return out


def key(r):
    return (r["subject"].lower(), r["electrode"])


def selftest():
    """Derive V17's no-split table from V17's split table and compare with the real one."""
    split = list(csv.DictReader(open(os.path.join(PUP, "cohort_energy_deposition_csf.csv"))))
    truth = {key(r): r for r in csv.DictReader(open(os.path.join(PUP, "cohort_energy_deposition.csv")))}
    got = {key(r): r for r in derive(split)}
    if set(got) != set(truth):
        raise SystemExit(f"[selftest] key sets differ: derived {len(got)}, real {len(truth)}")
    worst = {}
    for k in got:
        for c in ("gm_frac", "wm_frac", "other_frac"):
            d = abs(float(got[k][c]) - float(truth[k][c]))
            worst[c] = max(worst.get(c, 0.0), d)
        if got[k]["n_vox"] != truth[k]["n_vox"]:
            raise SystemExit(f"[selftest] n_vox differs at {k}")
    print(f"[selftest] V17 round-trip on {len(got)} rows: "
          + "  ".join(f"{c} max|d|={v:.3e}" for c, v in worst.items()))
    bad = [c for c, v in worst.items() if v > TOL]
    if bad:
        raise SystemExit(f"[selftest] FAILED, over tolerance {TOL:g}: {bad}")
    print(f"[selftest] PASS (tolerance {TOL:g} = the CSV's own rounding)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(PUP, "cohort_energy_deposition_csf_v18h3.csv"))
    ap.add_argument("--out", default=os.path.join(PUP, "cohort_energy_deposition_v18h3.csv"))
    ap.add_argument("--selftest", action="store_true", help="run the V17 round-trip and stop")
    a = ap.parse_args()

    selftest()
    if a.selftest:
        return
    rows = list(csv.DictReader(open(a.src)))
    if len(rows) != 1596:
        raise SystemExit(f"[derive] {a.src} has {len(rows)} rows, want 1596 (84 heads x 19 electrodes)")
    out = derive(rows)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"[derive] wrote {a.out}  ({len(out)} rows)")


if __name__ == "__main__":
    sys.exit(main())
