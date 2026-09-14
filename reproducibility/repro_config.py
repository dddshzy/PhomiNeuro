"""Filesystem configuration for the reproducibility pipeline.

All generated data stay under ``PHOMINEURO_WORK_DIR`` by default. External
datasets and the VISTA3D checkpoint are supplied explicitly through environment
variables so that no machine-specific path is embedded in the source code.
"""

from __future__ import annotations

import os
from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parent
PUBLIC_REPO_ROOT = REPRO_ROOT.parent


def _path(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


WORK_DIR = _path("PHOMINEURO_WORK_DIR", REPRO_ROOT / "work")
DATASET_DIR = _path("PHOMINEURO_DATASET_DIR", Path(WORK_DIR) / "dataset")
STANDARD_DATASET_DIR = _path(
    "PHOMINEURO_STANDARD_DATASET_DIR", Path(WORK_DIR) / "dataset_v2_head810"
)
MNI_DATASET_DIR = _path(
    "PHOMINEURO_MNI_DATASET_DIR", Path(WORK_DIR) / "dataset_v11_mni"
)
SIM_ROOT = _path("PHOMINEURO_SIM_ROOT", Path(WORK_DIR) / "simulations")
SIM_DATA_DIR = _path("PHOMINEURO_SIM_DATA_DIR", Path(SIM_ROOT) / "data")
MNI_SIM_DIR = _path(
    "PHOMINEURO_MNI_SIM_DIR", Path(SIM_ROOT) / "sim_v11_mni_810" / "data" / "wl810"
)
PYRAMID_DIR = _path("INR_PYRDIR", Path(WORK_DIR) / "extracted_pyramids_v16")
INR_CHECKPOINT_DIR = _path(
    "PHOMINEURO_INR_CHECKPOINT_DIR", Path(WORK_DIR) / "inr_checkpoints"
)
RESULTS_DIR = _path("PHOMINEURO_RESULTS_DIR", Path(WORK_DIR) / "results")
METRICS_DIR = _path("PHOMINEURO_METRICS_DIR", Path(RESULTS_DIR) / "metrics")
TABLE_DIR = _path("PHOMINEURO_TABLE_DIR", Path(RESULTS_DIR) / "tables")
HIPHOT_DIR = _path("PHOMINEURO_HIPHOT_DIR", Path(SIM_ROOT) / "sim_v11_hiphoton")
VOLUME_CACHE_DIR = _path(
    "PHOMINEURO_VOLUME_CACHE_DIR", Path(WORK_DIR) / "vol_cache_128"
)

FM_TRAINING_DIR = _path("FM_DATA_DIR", DATASET_DIR)
FM_SAVE_DIR = _path("FM_SAVE_DIR", Path(WORK_DIR) / "fm_encoder")
FM_ADAPTER_DIR = _path("FM_CKPT_DIR", PUBLIC_REPO_ROOT / "weights" / "fm_encoder")
VISTA3D_CKPT = _path(
    "VISTA3D_CKPT", Path(WORK_DIR) / "external" / "vista3d" / "model.pt"
)

BRAINWEB_CACHE = _path("BRAINWEB_CACHE", Path(WORK_DIR) / "sources" / "brainweb")
SCATTERBRAINS_DIR = _path(
    "SCATTERBRAINS_DIR", Path(WORK_DIR) / "sources" / "scatterBrains"
)
SHARM_DIR = _path("SHARM_DIR", Path(WORK_DIR) / "sources" / "SHARM")
OASIS_DIR = _path("OASIS_DIR", Path(WORK_DIR) / "sources" / "OASIS-3")
GRACE_DIR = _path("GRACE_DIR", Path(WORK_DIR) / "sources" / "GRACE")
SHARM_DATASET_DIR = _path(
    "PHOMINEURO_SHARM_DATASET_DIR", Path(WORK_DIR) / "dataset_sharm810"
)
OASIS_DATASET_DIR = _path(
    "PHOMINEURO_OASIS_DATASET_DIR", Path(WORK_DIR) / "dataset_oasis810"
)
OASIS_ANALYSIS_DIR = _path(
    "PHOMINEURO_OASIS_ANALYSIS_DIR", Path(WORK_DIR) / "dataset_OASIS"
)
OASIS_METADATA_DIR = _path(
    "PHOMINEURO_OASIS_METADATA_DIR", Path(OASIS_ANALYSIS_DIR) / "oasis3_metadata"
)
PUP_MNI_DIR = _path(
    "PHOMINEURO_PUP_MNI_DIR", Path(OASIS_ANALYSIS_DIR) / "pup_mni"
)


def ensure_output_dirs() -> None:
    """Create the directories written by the core training pipeline."""

    for path in (
        WORK_DIR,
        DATASET_DIR,
        STANDARD_DATASET_DIR,
        MNI_DATASET_DIR,
        SHARM_DATASET_DIR,
        OASIS_DATASET_DIR,
        OASIS_ANALYSIS_DIR,
        OASIS_METADATA_DIR,
        PUP_MNI_DIR,
        SIM_ROOT,
        SIM_DATA_DIR,
        PYRAMID_DIR,
        INR_CHECKPOINT_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        TABLE_DIR,
        FM_SAVE_DIR,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
