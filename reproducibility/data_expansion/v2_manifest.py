"""
Central manifest for the FM-INR v2 (810 nm, angle-conditioned, head-only) build.

One place for: the head list, the held-out test head, the wavelength, all v2
output paths, and the per-head native->canonical orientation transform.

Orientation: validated axis roles for every head are
    axis0 = L-R (smallest extent), axis1 = A-P, axis2 = S-I.
Only the SIGNS vary. AP_SIGN[h]=+1 means the head's anterior is already +Y;
SI_SIGN[h]=+1 means superior is already +Z (see viz/_orient_fingerprint.py).
We map each head into the canonical frame
    +X = right (don't-care sign), +Y = anterior (forward), +Z = superior
with a PROPER rotation (det=+1, no mirroring) by pairing any needed A-P / S-I
flip with an L-R flip (L-R sign is immaterial to the symmetric B sweep).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from augment import RigidTransform                       # signed axis permutation
import anat_angles as AA                                 # angle grid + srcdir geometry
import repro_config as RC

# ----------------------------------------------------------------------------
# Heads
# ----------------------------------------------------------------------------
# NOTE: the 2 continuous atlas heads (ADT1, MNI) were DROPPED from v2 because their
# scalp/skull optical layers are spatially swapped (skull-valued shell outermost;
# verified vs the anatomically-correct discrete scb heads). v2 trains on the 16
# discrete, anatomically-correct scb heads only.
ATLAS_HEADS = ["ADT1", "mni_ext_2020"]                  # excluded from v2 (swapped layers)
SCB_HEADS = [f"scb{i:02d}" for i in range(1, 17)]        # scb01..scb16 (5-class, 1mm)
BW_HEADS = [f"bw{i:02d}" for i in range(1, 21)]          # bw01..bw20 BrainWeb (9-tissue, 1mm)
HEADS = SCB_HEADS + BW_HEADS                             # 36 base head geometries
# ── Subject-level 3-way split (stratified scb + bw) ──────────────────────────
# VAL  = used for ALL model/hyperparameter/recipe selection + early-stopping.
# TEST = LOCKED: evaluated exactly ONCE for final reporting, never used for any
#        decision, and EXCLUDED from BOTH INR training AND FM-encoder finetuning
#        (so a fresh v9 encoder must exclude val+test for a pristine test set).
# bw19 (anatomical outlier, thickest skull+vasculature) kept in TRAIN.
VAL_HEADS  = ["scb08", "scb11", "bw17", "bw18", "bw20"]  # 2 scb + 3 bw (former dev heads)
TEST_HEADS = ["scb15", "scb16", "bw14", "bw15", "bw16"]  # 2 scb + 3 bw (locked, pristine)
HELDOUT_HEADS = VAL_HEADS + TEST_HEADS                   # excluded from train + aug + encoder
HELDOUT_HEAD = VAL_HEADS[0]                              # back-compat scalar
TRAIN_HEADS = [h for h in HEADS if h not in HELDOUT_HEADS]
WL = "810"

# ----------------------------------------------------------------------------
# Paths (all v2 artefacts isolated; v1 untouched)
# ----------------------------------------------------------------------------
ROOT          = RC.WORK_DIR
SRC_DATASET   = RC.DATASET_DIR
DATASET_V2    = RC.STANDARD_DATASET_DIR
PMCX_ROOT     = RC.SIM_ROOT
SIM_V2_ROOT   = os.path.join(PMCX_ROOT, "sim_v2_810")
SIM_V2_PRETEST = os.path.join(SIM_V2_ROOT, "pretest")
SIM_V2_DATA   = os.path.join(SIM_V2_ROOT, "data", "wl810")
CACHE_V2      = os.path.join(SIM_V2_ROOT, "cache")                  # per-head cfg cache
PYRAMID_V2    = os.path.join(ROOT, "INR_design", "extracted_pyramids_v2")
INR_CKPT_V2   = os.path.join(ROOT, "INR_design", "inr_checkpoints_v2")
VIZ_V2        = os.path.join(ROOT, "viz", "out", "v2")

# ---- v3 augmentation (rotation aug + angle-perturbation aug) ----
DATASET_V3_AUG = os.path.join(ROOT, "dataset_v3_aug")               # rotated head volumes
SIM_V3_ROT     = os.path.join(SIM_V2_ROOT, "data_rot", "wl810")     # rotated-head fluences
SIM_V3_PERT    = os.path.join(SIM_V2_ROOT, "data_pert", "wl810")    # perturbed-angle fluences
PYRAMID_V3     = os.path.join(ROOT, "INR_design", "extracted_pyramids_v3")  # rotated-head pyramids
INR_CKPT_V3    = os.path.join(ROOT, "INR_design", "inr_checkpoints_v3")
VIZ_V3         = os.path.join(ROOT, "viz", "out", "v3")


# ---- v10 (discrete-media rebuild + rot-aug encoder + PINN-stratification ablation) ----
# Fresh, isolated paths so the buggy v2/v3 1:1-media artefacts are never reused. Set
# env FMINR_V10=1 to redirect all v2/v3-keyed scripts onto these (see override block below).
SIM_V10_ROOT     = os.path.join(PMCX_ROOT, "sim_v10_810")
SIM_V10_DATA     = os.path.join(SIM_V10_ROOT, "data", "wl810")
SIM_V10_ROT      = os.path.join(SIM_V10_ROOT, "data_rot", "wl810")
SIM_V10_PERT     = os.path.join(SIM_V10_ROOT, "data_pert", "wl810")
CACHE_V10        = os.path.join(SIM_V10_ROOT, "cache")
DATASET_V10_AUG      = os.path.join(ROOT, "dataset_v10_aug")          # rotated head volumes
DATASET_V10_FM_TRAIN = os.path.join(ROOT, "dataset_v10_fm_train")    # 26 train-head volumes
PYRAMID_V10      = os.path.join(ROOT, "INR_design", "extracted_pyramids_v10")      # base heads
PYRAMID_V10_ROT  = os.path.join(ROOT, "INR_design", "extracted_pyramids_v10_rot")  # rot heads
INR_CKPT_V10     = os.path.join(ROOT, "INR_design", "inr_checkpoints_v10")
FM_CKPT_V10      = os.path.join(ROOT, "FM_tune_v10", "checkpoints")
VIZ_V10          = os.path.join(ROOT, "viz", "out", "v10")


def std_mat_v3(head):
    """Standardized (rotated) augmented head volume path."""
    return os.path.join(DATASET_V3_AUG, f"{head}_copmri_withHermiteF{WL}.mat")

PROP_KEY = "vol_prop_eye_aseg"


def src_mat(head):
    return os.path.join(SRC_DATASET, f"{head}_copmri_withHermiteF{WL}.mat")


def std_mat(head):
    return os.path.join(DATASET_V2, f"{head}_copmri_withHermiteF{WL}.mat")


# ----------------------------------------------------------------------------
# Per-head orientation signs (from viz/_orient_fingerprint.py).
# +1 = that anatomical direction already matches canonical (+Y ant / +Z sup).
# FILLED from the fingerprint run; LR sign is don't-care (not listed).
# ----------------------------------------------------------------------------
# Native->canonical orientation (verified visually + via anatomical AP/SI signs):
#   atlas heads (ADT1, mni_ext_2020) are ALREADY canonical -> identity.
#   scb heads store the head ROTATED 90 deg about the L-R axis (their A-P and S-I
#   array axes are swapped vs atlas). The fix is a 90-deg CLOCKWISE rotation about
#   the L-R (X) axis: (y,z) -> (z,-y), i.e. perm (0,2,1), sign (1,1,-1).
#   Verified: this yields anterior=+Y, superior=+Z, L-R smallest extent for all scb
#   (viz/_pick_rotation.py).
IDENTITY = RigidTransform((0, 1, 2), (1, 1, 1))
SCB_ROT_CW90_X = RigidTransform((0, 2, 1), (1, 1, -1))     # 90 deg clockwise about X
# BrainWeb native axis order = (S-I, A-P, L-R) (verified: brain-vs-neck-muscle COM
# separation -> axis0=S-I; bilateral flip-symmetry -> axis2=L-R; skin/fat anterior
# skew -> +axis1=anterior). Map to canonical (X=L-R,Y=A-P,Z=S-I): perm (2,1,0) with
# +Y=anterior,+Z=superior already positive; the odd perm is paired with an L-R flip
# (don't-care sign) to stay a PROPER rotation (det=+1). Visually verified vs scb.
BW_ORIENT = RigidTransform((2, 1, 0), (-1, 1, 1))


def _orient(h):
    if h in ATLAS_HEADS:
        return IDENTITY
    if h.startswith("bw"):
        return BW_ORIENT
    return SCB_ROT_CW90_X


ORIENT = {h: _orient(h) for h in HEADS}


def transform_for(head):
    return ORIENT[head]


# Canonical anatomical axes after standardization (shared by all heads).
F_ANT, U_SUP, R_RIGHT = AA.F_ANT, AA.U_SUP, AA.R_RIGHT
angle_grid = AA.angle_grid
angles_to_srcdir = AA.angles_to_srcdir
angle_tag = AA.angle_tag


def is_heldout(head):
    return head in HELDOUT_HEADS


# ----------------------------------------------------------------------------
# V10 redirect: with FMINR_V10=1, every v2/v3-keyed script (run_grid_v2,
# run_perturb_v3, augment_rotate_v3, extract_pyramids_v2/v3, inr_dataset_v3,
# sim_core_v2.get_cfg) reads/writes the isolated V10 paths instead -- no script
# edits needed. Module-level functions (std_mat_v3) read these globals at call
# time, so the reassignment takes effect everywhere.
# ----------------------------------------------------------------------------
if os.environ.get("FMINR_V10") == "1":
    SIM_V2_DATA    = SIM_V10_DATA
    CACHE_V2       = CACHE_V10
    SIM_V3_ROT     = SIM_V10_ROT
    SIM_V3_PERT    = SIM_V10_PERT
    DATASET_V3_AUG = DATASET_V10_AUG
    PYRAMID_V2     = PYRAMID_V10
    PYRAMID_V3     = PYRAMID_V10_ROT
    INR_CKPT_V3    = INR_CKPT_V10
    VIZ_V3         = VIZ_V10
    print("[v2_manifest] FMINR_V10=1 -> redirected to V10 discrete-media paths")


if __name__ == "__main__":
    print(f"v2 heads ({len(HEADS)}): {HEADS}")
    print(f"held-out: {HELDOUT_HEAD} | wl={WL}")
    print(f"angle grid: {len(angle_grid())} (A,B) pairs/head")
    print(f"=> {len(HEADS)*len(angle_grid())} total sims (formal)")
    for h in HEADS:
        t = transform_for(h)
        print(f"  {h:14s} -> perm{t.perm} sign{t.sign} proper={t.is_proper_rotation}")
