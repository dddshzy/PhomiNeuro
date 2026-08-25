<p align="center">
  <img src="docs/assets/logo.png" alt="PhomiNeuro" width="380">
</p>

<h1 align="center">PhomiNeuro</h1>

<p align="center">
  <b>A neural-field surrogate for time-resolved near-infrared light transport in the human head.</b><br>
  Monte-Carlo-quality fluence, at interactive speed, on any head model you can segment.
</p>

<p align="center">
  <a href="https://www.biorxiv.org/content/10.64898/2026.07.04.736179v1"><b>Preprint (bioRxiv)</b></a> &nbsp;·&nbsp;
  <a href="DATA_CARD.md">Data card</a> &nbsp;·&nbsp;
  <a href="NOTICE">Licences</a> &nbsp;·&nbsp;
  <a href="CITATION.cff">How to cite</a>
</p>

---

PhomiNeuro predicts **log₁₀ fluence at 810 nm** inside a head model, at any point you ask for and at
any of **10 time gates** spanning 0.1–1.9 ns. It replaces a Monte-Carlo run that takes minutes with a
query that takes seconds, and it is differentiable, so it can sit inside an optimisation loop rather
than only being called from one.

```python
from phomineuro import Scene, Predictor, ROOT

scene = Scene("scb16", "Cz", ROOT, dev, cache_dir="pyramid_cache",
              vista3d_ckpt=".../vista3d.pt", weights_dir="weights/fm_encoder")
pred  = Predictor("weights/phomineuro_s0.pt", dev)

logphi = pred.at_points(scene, [[120, 130, 170]])   # (1, 10) — log10 fluence, all gates
field  = pred.whole_head(scene, gate=1)             # (X, Y, Z) — a full volume
```

> **This repository is a demonstration, not an evaluation.** It ships one head so you can install,
> run, and see the model work end to end. Accuracy figures, benchmarks, ablations and the full
> evaluation protocol are in the [preprint](https://www.biorxiv.org/content/10.64898/2026.07.04.736179v1) —
> please cite those numbers from the paper, not from anything you measure on this single demo head.

---

## What is in here

| | | size |
|---|---|---|
| `phomineuro/` | The package: encoder, scene builder, predictor | 60 KB |
| `weights/phomineuro_s{0,1,2}.pt` | The surrogate, three seeds | 14.5 MB |
| `weights/fm_encoder/` | LoRA adapter + STRD block for the feature encoder | 74 MB |
| `demo_heads/` | One head model, `scb16` (scatterBrains) | 1.7 MB |
| `demo_heads/meta/` | The 19 illumination sites (10-20 montage) | 17 KB |
| `demo_heads/mc_reference/` | One Monte-Carlo field, `scb16/Cz` | 18 MB |
| `examples/` | Three runnable scripts, smallest first | 20 KB |
| `tests/` | Parity check and a resource benchmark | 16 KB |

**Not** in here, and deliberately so:

- **The VISTA3D backbone** (872 MB, NVIDIA). Third-party, downloaded separately — see below.
- **Feature pyramids** (4.1 GB per head). Deterministic and rebuilt in ~13 s, so shipping them
  would trade 4 GB for 13 s.
- **Head models from BrainWeb, SHARM and OASIS-3.** All three are training inputs and all three are
  documented in [`docs/HEAD_MODELS.md`](docs/HEAD_MODELS.md), but their terms do not clearly permit
  redistributing derived data. scatterBrains is the one source whose licence explicitly does, which
  is why `scb16` is the head you get. See [`DATA_CARD.md`](DATA_CARD.md).

---

## Install

```bash
git clone https://github.com/dddshzy/myV18h3temp.git && cd myV18h3temp
pip install -r requirements.txt

# the VISTA3D backbone the encoder is built on (NVIDIA, public on HuggingFace)
huggingface-cli download nvidia/NV-Segment-CTMR \
  vista3d_pretrained_model/model.pt --local-dir ~/vista3d
export VISTA3D_CKPT=~/vista3d/vista3d_pretrained_model/model.pt
```

Python ≥ 3.10, PyTorch ≥ 2.0. A CUDA GPU is strongly recommended; see below for the CPU path.

---

## Run

Three examples, smallest first. Each is self-contained and each prints what it is doing.

```bash
python examples/01_quickstart.py              # one point, 10 gates — 30 seconds
python examples/02_plot_slice.py --with-mc    # a slice, next to Monte Carlo — 1 minute
python examples/03_verify_install.py          # check your install reproduces our reference
```

The first run of any of them extracts the head's feature pyramid (~13 s) and caches it to
`pyramid_cache/`. Later runs skip that. The cache is a pure speed-up — deleting it costs 13 s, never
correctness.

`examples/03_verify_install.py` compares the model against the shipped Monte-Carlo field and prints
the agreement per gate. Its purpose is to tell you **your installation is working**; it is a single
scene on a single head and is not a measure of how the model performs. That is in the paper.

---

## Hardware

Measured by `tests/bench_resources.py` on one 224×256×300 head, GH200 (96 GB), PyTorch 2.4 / CUDA 12:

| stage | wall | GPU peak | host RSS |
|---|---|---|---|
| build encoder | 5.7 s | 0.99 GB | 2.7 GB |
| **extract pyramid** (once per head) | 13.1 s | **2.98 GB** | 6.7 GB |
| load surrogate | <0.1 s | 0.14 GB | 6.7 GB |
| 1 point × 10 gates | 0.2 s | 4.44 GB | 7.0 GB |
| 100k points × 10 gates | 0.1 s | 7.18 GB | 7.0 GB |
| **whole head, 10 gates** | 2.3 s | **9.94 GB** | 7.0 GB |

**A 12 GB card runs everything here.** The two stages do not overlap — extraction peaks at 3 GB and
inference at 10 GB — so if memory is tight, extract the pyramid first, let the process exit, then
predict. On 8 GB, lower `--chunk` to 20000 and query points rather than whole volumes.

**Disk:** the repo is ~110 MB, the backbone 872 MB, and each cached pyramid 4.1 GB. The pyramid is
the item that will surprise you.

**Without a GPU** everything runs, slowly: pyramid extraction ~18 min instead of 13 s, and a whole
head ~4 min instead of 2.3 s. Point queries stay usable (~2 s for 100k points). Pass `--gpu -1`.
Host RSS peaks at 9.6 GB, so a 16 GB laptop is the practical floor.

---

## Using it correctly

**Coordinates are voxel indices** on the head's own 1 mm MNI grid, `(i, j, k)` — not millimetres and
not world coordinates. `scene.tissue` marks which voxels are tissue.

**Predictions are log₁₀.** Take `10 ** logphi` for linear fluence. The model regresses log fluence
because the quantity spans nine decades; an error that looks small in linear units is not small.

**Three seeds ship, and they are three independent trainings, not an ensemble.** Averaging them is
reasonable. Their spread is a rough indication of where the model is less certain — the gap widens
in the deep brain and at late gates, which is the expected behaviour of any light-transport model.

**Illumination sites** come from `demo_heads/meta/`. `Scene` takes an electrode name from the 10-20
montage (`Cz`, `T4`, `Fp1`, …) and resolves it to a beam entry point on the scalp. Nineteen sites
ship for `scb16`.

### Two ways to get silently wrong answers

1. **A different `manual_seed` before building the encoder.** The seed is part of the weights — see
   below. `Scene` handles this for you; only a caller that builds the encoder by hand can get it
   wrong, and `phomineuro/encoder.py` explains why.
2. **Changing `sw_batch_size` during pyramid extraction.** It looks like a performance knob and is
   actually a reproducibility one: it changes the order in which sliding windows accumulate. The
   code raises if you change it. To cut memory, use `out_device="cpu"`, which is bit-identical.

---

## Reproducibility: the seed is part of the weights

VISTA3D is a 1-channel model; our input has 4 channels (μₐ, μₛ, g, n). The input projection is
therefore built by inflating the pretrained kernel and adding a small symmetry-breaking noise term
drawn from the **global RNG**. That noise is not in the LoRA adapter, not in any optimiser state, and
cannot be recovered from the checkpoint. Only `torch.manual_seed(0)` immediately before construction
reproduces it, and `build_fm_encoder` does exactly that.

This is measured, not assumed: with `manual_seed(0)` the extracted pyramid is bit-identical
(`max|diff| = 0.0`, and `0.0` again from a separate process); with a different seed the input layers
correlate at 0.861 and the downstream bottleneck features move by 73 % relative. Change the seed and
every number changes with it.

---

## Licensing

**Licensing is not uniform across this repository, and you should read [`NOTICE`](NOTICE) before
redistributing any part of it.**

| part | terms |
|---|---|
| Code (`phomineuro/`, `examples/`, `tests/`) | Apache-2.0 ([`LICENSE`](LICENSE)) |
| `weights/fm_encoder/` | Derivative of NVIDIA NV-Segment-CTMR — **non-commercial** |
| `weights/phomineuro_s*.pt` | Apache-2.0 |
| `demo_heads/`, `mc_reference/` | scatterBrains License Agreement (BSD-style) |

Everything shipped here is under terms that explicitly permit redistribution. Sources whose terms do
not — BrainWeb, SHARM, OASIS-3 — contribute to the training set described in the paper but no derived
file from them appears in this repository. [`DATA_CARD.md`](DATA_CARD.md) records the provenance of
every file and the reasoning behind each exclusion.

---

## Citing

If PhomiNeuro is useful in your work, please cite the preprint:

> *PhomiNeuro: a neural-field surrogate for time-resolved near-infrared light transport in the human
> head.* bioRxiv (2026). https://www.biorxiv.org/content/10.64898/2026.07.04.736179v1

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). Please also cite **scatterBrains**
(the demo head) and **NVIDIA VISTA3D** (the encoder backbone) — both entries are in that file.
