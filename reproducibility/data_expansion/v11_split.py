"""
v11 subject-disjoint train/val/test split over the 225 rigid-MNI-normalized heads
(196 SHARM + 13 scb + 16 bw).

Design (mirrors v10's locked-test discipline, and keeps v10's EXACT scb/bw val/test
heads so v11-vs-v10-hero is compared on the identical scb/bw subjects):
  TEST  (locked, evaluated once, excluded from BOTH INR train and FM finetune):
        v10 test scb/bw {scb15,scb16,bw14,bw15,bw16} + 15 held-out SHARM
  VAL   (all model/recipe/early-stop decisions):
        v10 val  scb/bw {scb08,scb11,bw17,bw18,bw20} + 15 held-out SHARM
  TRAIN: the remaining 185 heads (166 SHARM + 9 scb + 10 bw)

SHARM val/test subjects are drawn deterministically (seeded) from sh001..sh196.
"""
import numpy as np

# available heads (present in dataset_v2_head810 / dataset_sharm810)
SCB = [f"scb{i:02d}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16)]   # 13
BW  = [f"bw{i:02d}"  for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 20)]  # 16
SH  = [f"sh{i:03d}" for i in range(1, 197)]                                    # 196
HEADS = SCB + BW + SH                                                          # 225

# v10 locked heads (kept identical for the v10-vs-v11 comparison)
V10_TEST_SCBBW = ["scb15", "scb16", "bw14", "bw15", "bw16"]
V10_VAL_SCBBW  = ["scb08", "scb11", "bw17", "bw18", "bw20"]

N_SH_TEST = 15
N_SH_VAL  = 15


def _sharm_holdout(seed=1111):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(SH)); rng.shuffle(idx)
    test = sorted(SH[i] for i in idx[:N_SH_TEST])
    val  = sorted(SH[i] for i in idx[N_SH_TEST:N_SH_TEST + N_SH_VAL])
    return test, val


_sh_test, _sh_val = _sharm_holdout()
TEST_HEADS  = sorted(V10_TEST_SCBBW + _sh_test)
VAL_HEADS   = sorted(V10_VAL_SCBBW + _sh_val)
HELDOUT     = set(TEST_HEADS) | set(VAL_HEADS)
TRAIN_HEADS = [h for h in HEADS if h not in HELDOUT]


def source_of(h):
    return "sh" if h.startswith("sh") else ("scb" if h.startswith("scb") else "bw")


def summary():
    def bd(hs):
        from collections import Counter
        c = Counter(source_of(h) for h in hs)
        return f"{len(hs)} (sh {c['sh']} scb {c['scb']} bw {c['bw']})"
    return (f"TRAIN {bd(TRAIN_HEADS)}\nVAL   {bd(VAL_HEADS)}\nTEST  {bd(TEST_HEADS)}\n"
            f"total {len(HEADS)}  heldout-disjoint={len(HELDOUT)==len(TEST_HEADS)+len(VAL_HEADS)}")


if __name__ == "__main__":
    print(summary())
    print("\nTEST:", TEST_HEADS)
    print("\nVAL:", VAL_HEADS)
    assert not (set(TRAIN_HEADS) & HELDOUT), "train leaks into heldout!"
    assert len(set(HEADS)) == 225
    print("\nOK: subject-disjoint, 225 heads")


# ---- OASIS-3 external-validation heads --------------------------------------------------------
# OASIS phantoms (oa*) exist ONLY as an unseen-subject generalisation test. Without this they fall
# through `label_of`'s default and are labelled "train", which would both contaminate any rebuilt
# training buffer and silently void the held-out claim.
OASIS_PREFIX = "oa"
