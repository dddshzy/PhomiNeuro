"""
Generate multi-illumination MCX ground truth for the EXISTING continuous heads.

For each head (sample, wl) and each source in the structured STARTER_GRID, run the
existing MCX forward sim (mcx_fluence_simv2.run_simulation), which writes
  fluence_{sample}_F{wl}_th{θ}_ph{φ}_r{r}.mat  + meta_...json
into sim_t1/data/wl{wl}/ — exactly the layout discover_scenes reads.  The property
volumes + pyramids already exist, so these new illuminations become extra training
scenes for the SAME geometry (the surrogate learns to generalize over the light
source).  Idempotent: skips illuminations already on disk.  cfg_vol/cfg_prop are
cached per (sample,wl), so the 6 sources of a geometry share one cfg build.

Run:  python -m data_expansion.generate_head_illuminations
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pmcx_sim"))
import repro_config as RC
from data_expansion.illumination import STARTER_GRID, illum_tag

SIM_DIR = RC.SIM_DATA_DIR
DATASET_DIR = RC.DATASET_DIR

# "精简起步集": the heads + 2 wavelengths shared by all of them.
HEADS = ["ADT1", "mni_ext_2020", "p1v1", "p2v8", "p4v1"]
WAVELENGTHS = ["810", "1064"]
NPHOTON = 5e6                      # match the existing th75 GT quality


def main():
    from mcx_fluence_simv2 import run_simulation

    todo = []
    for sample in HEADS:
        for wl in WAVELENGTHS:
            if not os.path.isfile(
                    f"{DATASET_DIR}/{sample}_copmri_withHermiteF{wl}.mat"):
                continue
            for c in STARTER_GRID:
                tag = illum_tag(c["theta"], c["phi"], c["radius"])
                flu = f"{SIM_DIR}/wl{wl}/fluence_{sample}_F{wl}_{tag}.mat"
                if not os.path.isfile(flu):
                    todo.append((sample, wl, c))

    print(f"[gen] {len(todo)} illumination sims to run "
          f"({len(HEADS)} heads x {len(WAVELENGTHS)} wl x {len(STARTER_GRID)} sources)",
          flush=True)
    for i, (sample, wl, c) in enumerate(todo, 1):
        tag = illum_tag(c["theta"], c["phi"], c["radius"])
        print(f"\n[gen] {i}/{len(todo)}  {sample} F{wl} {tag}", flush=True)
        run_simulation(
            wavelength=wl, theta_deg=c["theta"], phi_deg=c["phi"],
            radius_mm=c["radius"], sample_id=sample, nphoton=NPHOTON,
            output_dir=f"{SIM_DIR}/wl{wl}", save_mat=True,
        )
    print(f"\n[gen] DONE — {len(todo)} new illuminations generated", flush=True)


if __name__ == "__main__":
    main()
