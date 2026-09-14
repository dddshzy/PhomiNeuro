"""
Per-tissue optical-property LUT for the 5-layer head model — the AUTHORITATIVE
"F"-series parameters (Q.F.), transcribed faithfully from
    pmcx_sim/optical_parameter_complete_version_v2.m

Self-consistency check: F810 WM = [mu_a=0.092, mu_s=38, g=0.87, n=1.37], which is
exactly the (mu_a,mu_s) maximum of the existing continuous head dataset — i.e. the
current heads were built from this very table. So scatterBrains heads mapped here
sit on the SAME optical scale as the existing heads.

The .m raw table p0 has 9 rows (per wavelength):
    0 background/air | 1 WM | 2 GM | 3 CSF | 4 SKULL | 5 vessel | 6 fat | 7 muscle | 8 skin
(mu_s values written as `mu_s'/(1-g)` in the .m are already RAW mu_s here.)

para5 (the .m's 5-layer + background + fat output) is rebuilt with the .m's tissue
volume-fraction weights, then mapped to scatterBrains labels:
    label 0 background | 1 scalp | 2 skull | 3 CSF | 4 GM | 5 WM
(scalp = volume-weighted blend of skin + muscle + fat, exactly as in the .m).

Columns: (mu_a [mm^-1], mu_s [mm^-1], g, n).
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optical_config as OC

_d = 1.0 - 0.89   # the recurring "/(1-0.89)" reduced->raw scattering factor in the .m

# raw 9-row tables (rows: bg, WM, GM, CSF, SKULL, vessel, fat, muscle, skin)
_P0 = {
    "F670": [
        [0,      0,         1.0,   1.0],
        [0.07,   40.1,      0.85,  1.37],   # WM
        [0.02,   8.4,       0.90,  1.37],   # GM
        [0.0004, 0.01/_d,   0.89,  1.37],   # CSF
        [0.0208, 1.19/_d,   0.89,  1.37],   # SKULL
        [0.13,   31.5,      0.989, 1.37],   # vessel
        [0.00045,1.21/_d,   0.89,  1.37],   # fat
        [0.054,  7.65,      0.93,  1.37],   # muscle
        [0.056,  2.42/_d,   0.89,  1.37],   # skin/scalp
    ],
    "F810": [
        [0,      0,         1.0,   1.0],
        [0.092,  38,        0.87,  1.37],
        [0.028,  7.3,       0.89,  1.37],
        [0.0026, 0.01/_d,   0.89,  1.37],
        [0.011,  1.92/_d,   0.89,  1.37],
        [0.11,   28,        0.99,  1.37],
        [0.00054,1.09/_d,   0.89,  1.37],
        [0.028,  0.704/_d,  0.89,  1.37],
        [0.045,  2.18/_d,   0.89,  1.37],
    ],
    "F850": [
        [0,      0,         1.0,   1.0],
        [0.1,    35,        0.87,  1.37],
        [0.033,  7.0,       0.90,  1.37],
        [0.0042, 0.01/_d,   0.89,  1.37],
        [0.011,  1.87/_d,   0.89,  1.37],
        [0.15,   27.5,      0.99,  1.37],
        [0.00071,1.064/_d,  0.89,  1.37],
        [0.03,   0.667/_d,  0.89,  1.37],
        [0.04,   2.043/_d,  0.89,  1.37],
    ],
    "F980": [
        [0,      0,         1.0,   1.0],
        [0.11,   31,        0.88,  1.37],
        [0.052,  6,         0.91,  1.37],
        [0.048,  0.01/_d,   0.89,  1.37],
        [0.022,  1.73/_d,   0.89,  1.37],
        [0.2,    22,        0.987, 1.37],
        [0.0014, 0.99/_d,   0.89,  1.37],
        [0.049,  0.58/_d,   0.89,  1.37],
        [0.028,  1.85/_d,   0.89,  1.37],
    ],
    "F1064": [
        [0,      0,         1.0,   1.0],
        [0.105,  30,        0.88,  1.37],
        [0.053,  5.9,       0.91,  1.37],
        [0.0144, 0.01/_d,   0.89,  1.37],
        [0.019,  1.61/_d,   0.89,  1.37],
        [0.13,   20,        0.985, 1.37],
        [0.0054, 0.945/_d,  0.89,  1.37],
        [0.056,  0.55/_d,   0.89,  1.37],
        [0.017,  2.03/_d,   0.89,  1.37],
    ],
}

WAVELENGTHS = tuple(_P0.keys())                 # F670, F810, F850, F980, F1064

# .m tissue volume-fraction weights (tissueratio8), used to blend scalp
_RATIO8 = {
    5: 0.0149 + 0.0088,          # around-fat + fat
    6: 0.0272 + 0.1317 / 2.0,    # muscle
    7: 0.1317 / 2.0,             # skin
}

# scatterBrains label -> p0 row index (the .m's para5 mapping, inlined)
#   1 scalp  -> blend(skin=8, muscle=7, fat=6)
#   2 skull  -> p0[4];  3 CSF -> p0[3];  4 GM -> p0[2];  5 WM -> p0[1]
_LABEL_P0ROW = {2: 4, 3: 3, 4: 2, 5: 1}         # skull, csf, GM, WM (label->row)


def para5_row(wavelength: str, label: int) -> np.ndarray:
    """(mu_a, mu_s, g, n) for a scatterBrains tissue label (0..5) at a wavelength."""
    p0 = np.array(_P0[wavelength], dtype=np.float64)
    if label == 0:
        return p0[0].copy()                     # background/air
    if label == 1:                              # scalp = volume-weighted skin/muscle/fat
        w = _RATIO8
        num = p0[8] * w[7] + p0[7] * w[6] + p0[6] * w[5]
        return num / (w[5] + w[6] + w[7])
    return p0[_LABEL_P0ROW[label]].copy()


def tissue_lut(wavelength: str) -> dict:
    """{label(0..5): (mu_a,mu_s,g,n)} for the 5-layer scatterBrains model."""
    return {lab: para5_row(wavelength, lab) for lab in range(6)}


def labels_to_property_volume(label_vol: np.ndarray, wavelength: str) -> np.ndarray:
    """(X,Y,Z) uint label volume (0..5) -> (X,Y,Z,4) physical optical volume."""
    lut = tissue_lut(wavelength)
    out = np.zeros((*label_vol.shape, 4), dtype=np.float32)
    for lab, props in lut.items():
        out[label_vol == lab] = props
    return out


# ---------------------------------------------------------------------------
# BrainWeb-20 full-tissue mapping (NO 5-layer collapse).
# BrainWeb crisp models use 12 intensity codes that match the .m tissueratio12
# EXACTLY (verified empirically on subject_04). Each code maps to a row of the
# 9-row _P0 table: 0 bg | 1 WM | 2 GM | 3 CSF | 4 SKULL | 5 vessel | 6 fat |
# 7 muscle | 8 skin. Ambiguous classes: around-fat->fat,
# dura->skull, marrow->skull.
# ---------------------------------------------------------------------------
BW_CODE_TO_P0ROW = {
    0:   0,   # background/air
    16:  3,   # CSF
    32:  2,   # GM
    48:  1,   # WM
    64:  6,   # fat
    80:  7,   # muscle
    96:  8,   # skin (muscle/skin)
    112: 4,   # skull
    128: 5,   # vessel
    145: 6,   # around-fat   -> fat
    161: 4,   # dura mater   -> skull
    177: 4,   # bone marrow  -> skull
}
BW_CODE_NAME = {0: "bg", 16: "CSF", 32: "GM", 48: "WM", 64: "fat", 80: "muscle",
                96: "skin", 112: "skull", 128: "vessel", 145: "around-fat",
                161: "dura", 177: "marrow"}


# OASIS-3 6-tissue scheme (T1w + TOF-angio derived): the 5-layer OASIS scheme
# PLUS vessel from the TOF angio. Same F-series 9-row optical table as
# SHARM/BrainWeb, so the optical scale is directly comparable across datasets.
#
# A scalp split into skin/fat/muscle was tried and DROPPED: it rested on an
# unvalidated bright-vs-dark intensity heuristic with no anatomical prior (52.7%
# of the resulting "muscle" fell in the neck). Scalp is kept as one tissue rather
# than shipping three labels we cannot defend.
OASIS6_LABEL_TO_P0ROW = {
    0: 0,   # air / background
    1: 8,   # scalp / skin
    2: 4,   # skull
    3: 3,   # CSF
    4: 2,   # grey matter
    5: 1,   # white matter
    6: 5,   # blood vessel   <- from TOF angio
}
OASIS6_LABEL_NAME = {0: "air", 1: "scalp", 2: "skull", 3: "CSF", 4: "GM",
                     5: "WM", 6: "vessel"}


def oasis6_labels_to_property_volume(label_vol: np.ndarray, wavelength: str) -> np.ndarray:
    """(X,Y,Z) OASIS 6-tissue uint8 label volume -> (X,Y,Z,4) physical optical volume."""
    p0 = np.array(_P0[wavelength], dtype=np.float64)
    present = set(np.unique(label_vol).tolist())
    unknown = present - set(OASIS6_LABEL_TO_P0ROW)
    assert not unknown, f"unexpected OASIS6 labels {sorted(unknown)}"
    out = np.zeros((*label_vol.shape, 4), dtype=np.float32)
    for lab, row in OASIS6_LABEL_TO_P0ROW.items():
        out[label_vol == lab] = p0[row]
    return out


# OASIS-3 9-tissue HYBRID scheme -- the scheme actually used for the AD-vs-HC cohort.
# Provenance per tissue (deliberately mixed, one source per tissue, see generate_oasis_v2):
#   CSF / GM / WM   <- ANTs Atropos (EM+MRF) inside the deepbet brain mask (classic method)
#   skull/skin/fat/muscle/air/eyes <- GRACE (UNETR, trained on older adults)
#   vessel          <- TOF angio threshold
# Rationale for keeping Atropos rather than GRACE's brain classes: GRACE over-calls CSF
# on this elderly cohort (CSF 628k ~ GM 622k > WM 407k on OAS30884), and CSF's mu_a is
# ~1/10 of GM's, so a CSF over-estimate would bias photon transport badly.
# Two decisions folded in here:
#   * GRACE cortical AND cancellous bone both -> the single SKULL row (row 4). The LUT
#     has no separate diploe entry and we will not invent an unvalidated optical value.
#   * GRACE eyes -> CSF (row 3): vitreous humour is ~99% water, matching the existing
#     SHARM `vitreous -> CSF` precedent (SHARM_LABEL_TO_P0ROW label 14).
OASIS9_LABEL_TO_P0ROW = {
    0: 0,   # air / background (incl. sinus air; enclosed cavities become CSF at sim time)
    1: 8,   # skin            <- GRACE
    2: 4,   # skull           <- GRACE cortical + cancellous, collapsed
    3: 3,   # CSF             <- Atropos (+ GRACE subarachnoid CSF & eyes outside brain)
    4: 2,   # grey matter     <- Atropos
    5: 1,   # white matter    <- Atropos
    6: 5,   # blood vessel    <- TOF angio
    7: 6,   # subcutaneous fat<- GRACE
    8: 7,   # muscle          <- GRACE
}
OASIS9_LABEL_NAME = {0: "air", 1: "skin", 2: "skull", 3: "CSF", 4: "GM", 5: "WM",
                     6: "vessel", 7: "fat", 8: "muscle"}

# GRACE native label -> our OASIS9 label. GRACE indices verified empirically by depth +
# adjacency + voxel count (see external/GRACE/infer_single.py LABELS).
GRACE_TO_OASIS9 = {
    0: None,  # background -> leave to fallback
    1: 5,     # WM  (only used where it falls outside our deepbet brain mask)
    2: 4,     # GM  (ditto)
    3: 3,     # eyes -> CSF   [vitreous ~99% water; SHARM precedent]
    4: 3,     # CSF (subarachnoid, outside the brain mask)
    5: 0,     # air (sinuses/mastoid)
    6: 6,     # blood -> vessel row (same tissue as the TOF vessels; unioned with them)
    7: 2,     # cancellous bone -> SKULL  [collapsed, no separate diploe row]
    8: 2,     # cortical bone   -> SKULL
    9: 1,     # skin
    10: 7,    # fat
    11: 8,    # muscle
}


def oasis9_labels_to_property_volume(label_vol: np.ndarray, wavelength: str) -> np.ndarray:
    """(X,Y,Z) OASIS 9-tissue uint8 label volume -> (X,Y,Z,4) physical optical volume."""
    p0 = np.array(_P0[wavelength], dtype=np.float64)
    present = set(np.unique(label_vol).tolist())
    unknown = present - set(OASIS9_LABEL_TO_P0ROW)
    assert not unknown, f"unexpected OASIS9 labels {sorted(unknown)}"
    out = np.zeros((*label_vol.shape, 4), dtype=np.float32)
    for lab, row in OASIS9_LABEL_TO_P0ROW.items():
        out[label_vol == lab] = p0[row]
    return out


def bw_labels_to_property_volume(label_vol: np.ndarray, wavelength: str) -> np.ndarray:
    """(X,Y,Z) BrainWeb intensity-code volume -> (X,Y,Z,4) physical optical volume.

    Uses the FULL 9-tissue F-series table (vessel/fat/muscle/skin kept distinct),
    unlike the scb 5-layer `labels_to_property_volume`.
    """
    p0 = np.array(_P0[wavelength], dtype=np.float64)
    present = set(np.unique(label_vol).tolist())
    unknown = present - set(BW_CODE_TO_P0ROW)
    assert not unknown, f"unexpected BrainWeb labels {sorted(unknown)}"
    out = np.zeros((*label_vol.shape, 4), dtype=np.float32)
    for code, row in BW_CODE_TO_P0ROW.items():
        out[label_vol == code] = p0[row]
    return out


# ---------------------------------------------------------------------------
# SHARM-16 full-tissue mapping (NO 5-layer collapse).
# SHARM head models (Rashed et al. 2022/2025, IXI-derived, 256^3, 1 mm iso) use
# 16 uint8 tissue labels. Each maps to a row of the SAME 9-row _P0 F-series table
# (0 bg | 1 WM | 2 GM | 3 CSF | 4 SKULL | 5 vessel | 6 fat | 7 muscle | 8 skin),
# so SHARM heads sit on the identical optical scale as the existing scb/bw heads.
#
# Faithful 1:1 rows (skin, muscle, fat, GM, WM, CSF, vessel, skull) match the .m
# table exactly. Four SHARM classes have no dedicated F-series row and are mapped
# by the closest physiological proxy (documented, mirrors the BrainWeb precedent
# around-fat->fat / dura->skull / marrow->skull):
#   6  skull cancellous -> SKULL   (single mixed-bone row; = cortical, as marrow->skull)
#   9  mucous tissue     -> muscle  (generic soft vascular tissue proxy)
#   12 dura              -> SKULL   (.m dura->skull precedent)
#   14 vitreous humor    -> CSF     (~99% water; CSF is the water-like low-mu row)
#   15 eye lens          -> muscle  (soft-tissue proxy; 296 vox ~0.002%, negligible)
# NOTE label 1 is unused in SHARM (labels are 0,2,3,...,16).
SHARM_LABEL_TO_P0ROW = {
    0:  0,   # air / background
    2:  8,   # skin
    3:  7,   # muscle
    4:  6,   # fat
    5:  4,   # skull (cortical bone)
    6:  4,   # skull (cancellous bone) -> SKULL  [proxy]
    7:  1,   # cerebellum white matter -> WM
    8:  2,   # cerebellum gray matter  -> GM
    9:  7,   # mucous tissue           -> muscle [proxy]
    10: 2,   # brain gray matter       -> GM
    11: 1,   # brain white matter      -> WM
    12: 4,   # dura                    -> SKULL  [proxy]
    13: 3,   # cerebrospinal fluid     -> CSF
    14: 3,   # vitreous humor          -> CSF    [proxy]
    15: 7,   # eye lens                -> muscle [proxy]
    16: 5,   # blood vessels           -> vessel
}
SHARM_LABEL_NAME = {
    0: "air", 2: "skin", 3: "muscle", 4: "fat", 5: "skull-cortical",
    6: "skull-cancellous", 7: "cerebellum-WM", 8: "cerebellum-GM", 9: "mucous",
    10: "brain-GM", 11: "brain-WM", 12: "dura", 13: "CSF", 14: "vitreous",
    15: "eye-lens", 16: "vessel",
}


def sharm_labels_to_property_volume(label_vol: np.ndarray, wavelength: str) -> np.ndarray:
    """(X,Y,Z) SHARM uint8 16-label volume -> (X,Y,Z,4) physical optical volume.

    Uses the FULL 9-tissue F-series table (vessel/fat/muscle/skin distinct),
    like `bw_labels_to_property_volume`, not the scb 5-layer collapse.
    """
    p0 = np.array(_P0[wavelength], dtype=np.float64)
    present = set(np.unique(label_vol).tolist())
    unknown = present - set(SHARM_LABEL_TO_P0ROW)
    assert not unknown, f"unexpected SHARM labels {sorted(unknown)}"
    out = np.zeros((*label_vol.shape, 4), dtype=np.float32)
    for lab, row in SHARM_LABEL_TO_P0ROW.items():
        out[label_vol == lab] = p0[row]
    return out


def assert_lut_in_phys_range():
    """Every tissue/wavelength optical value must lie inside optical_config.PHYS_RANGE."""
    for wl in WAVELENGTHS:
        for lab in range(6):
            mua, mus, g, n = para5_row(wl, lab)
            if lab == 0:
                continue                         # background row is [0,0,1,1]
            for name, val in zip(OC.CHANNELS, (mua, mus, g, n)):
                lo, hi = OC.PHYS_RANGE[name]
                assert lo - 1e-9 <= val <= hi + 1e-9, \
                    f"{wl} label{lab} {name}={val:.4f} outside PHYS_RANGE {(lo,hi)}"
        p0 = np.array(_P0[wl], dtype=np.float64)     # bw full-tissue rows too
        for code, row in BW_CODE_TO_P0ROW.items():
            if row == 0:
                continue
            for name, val in zip(OC.CHANNELS, p0[row]):
                lo, hi = OC.PHYS_RANGE[name]
                assert lo - 1e-9 <= val <= hi + 1e-9, \
                    f"{wl} bw {BW_CODE_NAME[code]} {name}={val:.4f} outside PHYS_RANGE {(lo,hi)}"
        for lab, row in SHARM_LABEL_TO_P0ROW.items():     # sharm full-tissue rows too
            if row == 0:
                continue
            for name, val in zip(OC.CHANNELS, p0[row]):
                lo, hi = OC.PHYS_RANGE[name]
                assert lo - 1e-9 <= val <= hi + 1e-9, \
                    f"{wl} sharm {SHARM_LABEL_NAME[lab]} {name}={val:.4f} outside PHYS_RANGE {(lo,hi)}"


if __name__ == "__main__":
    # self-consistency: F810 WM must equal the existing-head maximum [0.092, 38, 0.87, 1.37]
    wm810 = para5_row("F810", 5)
    print("F810 WM:", wm810, "(expect [0.092, 38, 0.87, 1.37])")
    print("\n5-layer LUT (mu_a, mu_s, g, n):")
    names = {0: "bg", 1: "scalp", 2: "skull", 3: "CSF", 4: "GM", 5: "WM"}
    for wl in WAVELENGTHS:
        print(f"  {wl}:")
        for lab in range(6):
            print(f"    {names[lab]:6s} {np.round(para5_row(wl, lab), 4)}")
    assert_lut_in_phys_range()
    print("\nALL LUT values within PHYS_RANGE OK")
