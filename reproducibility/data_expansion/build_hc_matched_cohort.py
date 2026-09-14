#!/usr/bin/env python3
"""Assemble the final matched-HC roster: merge the 14 matched candidates (select_hc_pet.py) with
the PUP->MNI transport QC (logs_hc_pup_to_mni.txt) into hc_matched_cohort.csv.

The roster is the demographically age/sex-matched, amyloid-negative (RSF Centiloid < 30),
same-session (gap=0) healthy controls whose PUP transported cleanly (cortex inside phantom brain).
"""
import os, csv, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import repro_config as RC
MET = str(RC.OASIS_METADATA_DIR)
LOG = str(RC.OASIS_ANALYSIS_DIR / "logs_hc_pup_to_mni.txt")

# transport QC:  [pup] oas30026  loaded  inside 99.9%  ctxSUVR 1.21  (181 labels)
qc = {}
pat = re.compile(r"\[pup\]\s+(\w+)\s+(\S+)\s+inside\s+([\d.]+)%\s+ctxSUVR\s+([\d.]+)\s+\((\d+)\s*labels\)")
for line in open(LOG):
    m = pat.search(line)
    if m:
        qc[m.group(1)] = dict(transform=m.group(2), inside=m.group(3),
                              ctx_suvr=m.group(4), n_labels=m.group(5))

cand = [r for r in csv.DictReader(open(os.path.join(MET, "hc_pet_candidates.csv")))
        if r["role"] == "matched"]

cols = ["subject", "sex", "age", "educ", "apoe", "mmse", "cdr", "centiloid_rsf", "amyloid_status",
        "tracer", "pet_session", "gap_days", "wmparc_inside_pct", "cortical_suvr_p50", "n_labels",
        "matched_ad", "matched_ad_age"]
out = os.path.join(MET, "hc_matched_cohort.csv")
n = 0
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader()
    for r in sorted(cand, key=lambda x: (x["sex"], float(x["age"]))):
        q = qc.get(r["subject"], {})
        w.writerow(dict(subject=r["subject"], sex=r["sex"], age=r["age"], educ=r["educ"],
                        apoe=r["apoe"], mmse=r["mmse"], cdr=r["cdr"], centiloid_rsf=r["centiloid_rsf"],
                        amyloid_status="negative", tracer=r["tracer"], pet_session=r["pet_session"],
                        gap_days=r["gap_days"], wmparc_inside_pct=q.get("inside", ""),
                        cortical_suvr_p50=q.get("ctx_suvr", ""), n_labels=q.get("n_labels", ""),
                        matched_ad=r["matched_ad"], matched_ad_age=r["matched_ad_age"]))
        n += 1
print(f"wrote {out}  ({n} matched HC)")
