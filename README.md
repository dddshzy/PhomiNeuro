<p align="center">
  <img src="docs/assets/logo.png" alt="PhomiNeuro" width="380">
</p>

<h1 align="center">PhomiNeuro</h1>

<p align="center">
  A neural-field surrogate for time-resolved near-infrared light transport in the human head.
</p>

<p align="center">
  <a href="https://www.biorxiv.org/content/10.64898/2026.07.04.736179v1"><b>Preprint (bioRxiv)</b></a> &nbsp;·&nbsp;
  <a href="DATA_CARD.md">Data card</a> &nbsp;·&nbsp;
  <a href="NOTICE">Licences</a> &nbsp;·&nbsp;
  <a href="CITATION.cff">How to cite</a>
</p>

---

## Overview

PhomiNeuro is a neural-field surrogate for time-resolved near-infrared (NIR) light transport in the
human head. Given a head model, it predicts log₁₀ fluence at 810 nm at arbitrary tissue points, at
any of ten time gates spanning 0.1–1.9 ns. A single query is answered in seconds on a workstation
GPU, whereas an equivalent Monte-Carlo simulation requires minutes. The model is differentiable and
can therefore be embedded directly in optimisation procedures, including inverse-design tasks.

```python
from phomineuro import Scene, Predictor, ROOT

scene = Scene("scb16", "Cz", ROOT, dev, cache_dir="pyramid_cache",
              vista3d_ckpt=".../vista3d.pt", weights_dir="weights/fm_encoder")
pred  = Predictor("weights/phomineuro_s0.pt", dev)

logphi = pred.at_points(scene, [[120, 130, 170]])   # (1, 10) — log10 fluence, all gates
field  = pred.whole_head(scene, gate=1)             # (X, Y, Z) — a full volume
```

**Scope of this repository.** This repository is intended for demonstration and installation
verification. It ships a single head model so that the pipeline can be exercised end to end.
Accuracy figures, ablations, and the full evaluation protocol are reported in the
[preprint](https://www.biorxiv.org/content/10.64898/2026.07.04.736179v1); measured values obtained
on the single shipped head should not be quoted in place of the published evaluation.

---

## Repository contents

| | | size |
|---|---|---|
| `phomineuro/` | Package: encoder, scene builder, predictor | 60 KB |
| `weights/phomineuro_s{0,1,2}.pt` | Surrogate model, three seeds | 14.5 MB |
| `weights/fm_encoder/` | LoRA adapter and STRD block for the feature encoder | 74 MB |
| `demo_heads/` | One head model, `scb16` (scatterBrains) | 1.7 MB |
| `demo_heads/meta/` | Nineteen illumination sites (10-20 montage) | 17 KB |
| `demo_heads/mc_reference/` | One Monte-Carlo field, `scb16/Cz` | 18 MB |
| `examples/` | Three runnable scripts | 20 KB |
| `tests/` | Parity check and resource benchmark | 16 KB |

The following items are intentionally not included:

- **The VISTA3D backbone** (872 MB, NVIDIA). This is a third-party file and is downloaded
  separately, as described under Installation.
- **Feature pyramids** (4.1 GB per head). Extraction is deterministic and completes in
  approximately 13 seconds; shipping them would add four gigabytes per head to the repository.
- **Head models from BrainWeb, SHARM, and OASIS-3.** These three sources contribute to the training
  set described in the paper and are documented in [`docs/HEAD_MODELS.md`](docs/HEAD_MODELS.md).
  Their terms of use do not clearly permit redistribution of derived data. scatterBrains is the one
  source whose licence explicitly permits redistribution, and it is therefore the only head source
  shipped. Provenance and licensing rationale are recorded in [`DATA_CARD.md`](DATA_CARD.md).

---

## Installation

```bash
git clone https://github.com/dddshzy/PhomiNeuro.git && cd PhomiNeuro
pip install -r requirements.txt

# the VISTA3D backbone the encoder is built on (NVIDIA, public on HuggingFace)
huggingface-cli download nvidia/NV-Segment-CTMR \
  vista3d_pretrained_model/model.pt --local-dir ~/vista3d
export VISTA3D_CKPT=~/vista3d/vista3d_pretrained_model/model.pt
```

Python ≥ 3.10 and PyTorch ≥ 2.0 are required. A CUDA-capable GPU is strongly recommended; the
CPU-only path is described under Hardware requirements.

---

## Usage

Three examples are provided, ordered from simplest to most complete. Each is self-contained and
prints a description of its actions.

```bash
python examples/01_quickstart.py              # one point, 10 gates — 30 seconds
python examples/02_plot_slice.py --with-mc    # a slice, next to Monte Carlo — 1 minute
python examples/03_verify_install.py          # verify the installation against the reference
```

The first invocation of any example extracts the head's feature pyramid (approximately 13 seconds)
and caches it in `pyramid_cache/`. Subsequent invocations reuse the cache. The cache is a
performance optimisation only; deleting it costs 13 seconds of extraction time and has no effect on
correctness.

`examples/03_verify_install.py` compares model predictions against the shipped Monte-Carlo field
and reports agreement per gate. Its purpose is to confirm that the installation is functioning
correctly. The shipped scene is a single illumination site on a single head and does not constitute
an evaluation of model performance; performance is reported in the paper.

---

## Hardware requirements

Measured with `tests/bench_resources.py` on one 224×256×300 head (GH200, 96 GB, PyTorch 2.4 /
CUDA 12):

| stage | wall | GPU peak | host RSS |
|---|---|---|---|
| build encoder | 5.7 s | 0.99 GB | 2.7 GB |
| **extract pyramid** (once per head) | 13.1 s | **2.98 GB** | 6.7 GB |
| load surrogate | <0.1 s | 0.14 GB | 6.7 GB |
| 1 point × 10 gates | 0.2 s | 4.44 GB | 7.0 GB |
| 100k points × 10 gates | 0.1 s | 7.18 GB | 7.0 GB |
| **whole head, 10 gates** | 2.3 s | **9.94 GB** | 7.0 GB |

All operations described here run on a GPU with 12 GB of memory. The extraction and inference
stages do not overlap: extraction peaks at 3 GB and inference at 10 GB. On memory-constrained
systems, extraction may be run first and the process terminated before prediction. On 8 GB systems,
reduce `--chunk` to 20000 and query points rather than whole volumes.

**Disk.** The repository occupies approximately 110 MB, the backbone 872 MB, and each cached pyramid
4.1 GB. The pyramid is the dominant component of disk usage.

**CPU-only operation.** All operations run without a GPU, at reduced throughput: pyramid extraction
takes approximately 18 minutes instead of 13 seconds, and a whole-head query approximately 4 minutes
instead of 2.3 seconds. Point queries remain usable (approximately 2 seconds for 100k points). Pass
`--gpu -1`. Peak host memory is 9.6 GB, so 16 GB of RAM is the practical minimum.

---

## Correct usage

**Coordinates are voxel indices** on the head's own 1 mm MNI grid, `(i, j, k)` — not millimetres and
not world coordinates. `scene.tissue` identifies the voxels belonging to tissue.

**Predictions are in log₁₀ units.** Linear fluence is obtained as `10 ** logphi`. The model
regresses log fluence because the quantity spans approximately nine decades; an error that is small
in linear units may not be small in absolute terms.

**Three seeds are provided.** They correspond to three independent training runs and are not an
ensemble. Averaging them is a reasonable practice. Their spread provides a qualitative indication of
uncertainty; the spread widens in the deep brain and at late gates, which is expected of any
light-transport model.

**Illumination sites** are listed in `demo_heads/meta/`. `Scene` accepts an electrode label from the
10-20 montage (`Cz`, `T4`, `Fp1`, …) and resolves it to a beam entry point on the scalp. Nineteen
sites are provided for `scb16`.

### Sources of error

1. **Calling the encoder constructor without the documented seed.** The seed is part of the weights;
   this is described in detail under Reproducibility. `Scene` handles the seed internally. The error
   can only arise from a caller that builds the encoder directly; `phomineuro/encoder.py` documents
   the mechanism.
2. **Changing `sw_batch_size` during pyramid extraction.** Although it resembles a performance
   parameter, it controls the order in which sliding-window contributions are accumulated and is
   therefore a reproducibility parameter. The code raises an error if it is changed. To reduce
   memory usage, use `out_device="cpu"`, which is bit-identical.

---

## Reproducibility

VISTA3D is a single-channel model, whereas the input here has four channels (μₐ, μₛ, g, n). The
input projection is therefore constructed by inflating the pretrained kernel and adding a small
symmetry-breaking noise term drawn from the global RNG. This noise term is not present in the LoRA
adapter, not present in any optimiser state, and cannot be recovered from the checkpoint. Exact
reproduction requires `torch.manual_seed(0)` immediately before construction, which
`build_fm_encoder` performs.

This property is verified empirically: with `manual_seed(0)`, the extracted pyramid is
bit-identical across runs (`max|diff| = 0.0`); with a different seed, the input layers correlate at
0.861 and the downstream bottleneck features change by 73 % relative. Changing the seed therefore
changes every downstream result.

---

## Licensing

Licensing is not uniform across this repository; [`NOTICE`](NOTICE) should be consulted before any
redistribution.

| part | terms |
|---|---|
| Code (`phomineuro/`, `examples/`, `tests/`) | Apache-2.0 ([`LICENSE`](LICENSE)) |
| `weights/fm_encoder/` | Derivative of NVIDIA NV-Segment-CTMR — **non-commercial** |
| `weights/phomineuro_s*.pt` | Apache-2.0 |
| `demo_heads/`, `mc_reference/` | scatterBrains License Agreement (BSD-style) |

All files shipped here are covered by terms that explicitly permit redistribution. Sources whose
terms do not permit this — BrainWeb, SHARM, and OASIS-3 — contribute to the training set described
in the paper, but no file derived from them is included in this repository.
[`DATA_CARD.md`](DATA_CARD.md) records the provenance of every file and the rationale for each
exclusion.

---

## Citation

If PhomiNeuro is used in published work, please cite the preprint:

> *PhomiNeuro: a neural-field surrogate for time-resolved near-infrared light transport in the human
> head.* bioRxiv (2026). https://www.biorxiv.org/content/10.64898/2026.07.04.736179v1

Machine-readable metadata is provided in [`CITATION.cff`](CITATION.cff). The scatterBrains head
model and the NVIDIA VISTA3D encoder backbone should also be cited; both entries are included in
that file.
