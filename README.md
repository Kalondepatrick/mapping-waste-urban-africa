# Mapping unmanaged waste in an informal settlement from ultra-fine-resolution drone imagery

Ndirande, Blantyre, Malawi — 3.81 cm ground sample distance.

This repository contains the complete analysis for a study that asks **under what
conditions pretrained and foundation models improve the generalisability of
environmental monitoring in data-scarce settings**, using unmanaged waste piles as
the test case.

The headline result is that "data-scarce" conflates two different problems.
Pretraining solves one of them and not the other:

| axis | question | effect of pretraining |
|---|---|---|
| **label scarcity** | how few annotated tiles to reach accuracy *X*? | **~10× better** (DINOv3 at 64 tiles ≈ scratch U-Net at 639) |
| **target rarity** | does the map work at deployment prevalence? | **no measurable effect** |

Across 53 configurations (~159 training runs, 0.48 M to 304 M parameters, five
pretraining corpora, four label budgets, frozen and fine-tuned) the
*deployability gap* — the ratio of achieved false-positive rate to the rate needed
for precision 0.5 at landscape prevalence — never fell below **16×**, and that
minimum was bought by letting recall drop to 0.43. Among the ten pretrained and
foundation backbones it ranged 26–38×.
See [`docs/FINDINGS.md`](docs/FINDINGS.md).

---

## Repository layout

```
.
├── 00_data_preparation/     tiling, spatial partitioning, dataset audit, chip generation
├── 01_geobia/               object-based image analysis baseline (R)
├── 02_unet_tensorflow/      original U-Net, TensorFlow/Keras
├── 03_unet_pytorch/         current trainer: scratch, ImageNet, OpenEarthMap,
│                            DINOv3, DINOv3-SAT, SatlasPretrain  ← start here
├── 04_inference/            whole-area prediction and land-cover context
├── 05_hard_negatives/       false-positive mining and complementary fine-tuning
├── 06_evaluation/           pixel- and object-level metrics, prevalence decomposition
├── 07_figures/              manuscript figures
├── 08_annotation_tools/     QGIS plugin and Shiny review app used to digitise labels
│
├── data/                    ← SINGLE data parent folder (Zenodo upload root)
├── results/                 ← SINGLE experiment-output folder, one subfolder per family
├── env/                     environment specifications
├── hpc/                     SLURM submission scripts (Sapitwa)
├── docs/                    findings, code map, references
├── tools/                   Zenodo packaging, QGIS project
└── NotNeeded/               superseded scripts, kept for provenance — see its README
```

Every script path and every data path in this repository is **relative to the
repository root**. Run everything from the root.

---

## Quick start

### 1. Environment

```powershell
python -m venv .venv-torch
.venv-torch\Scripts\python.exe -m pip install -r env\requirements-torch.txt
```

PyTorch must match your CUDA version; see [`env/README.md`](env/README.md).
R packages: `source("env/r-packages.R")`.

### 2. Data

Download the dataset from Zenodo (see [`data/README.md`](data/README.md)) and
unpack it so that it becomes the `data/` folder of this repository. Nothing else
needs configuring — scripts resolve paths from the repository root.

The deposited record is the **core tier** (~7.4 GB): `data/tiled_data/partitioned/`,
`data/splits/`, `data/vectors/`, `data/pretrained_weights/` and
`data/trained_models/`. The source orthomosaic is not deposited — it is openly
published on OpenAerialMap and the record cites it there.

### 3. Verify before training

The preflight harness proves that pretrained weights actually load — it never
falls back to random initialisation, it raises instead:

```powershell
.venv-torch\Scripts\python.exe 03_unet_pytorch\preflight.py --both-freeze-modes
```

Writes `results/06_pretrained_backbones/preflight_report.json`, recording how many
tensors transferred for every backbone. Re-running skips configurations that
already passed.

### 4. Train

One configuration:

```powershell
.venv-torch\Scripts\python.exe 03_unet_pytorch\01_train_unet_v5_torch.py --dev-dir data\tiled_data\partitioned\modelDevelopment --test-dir data\tiled_data\partitioned\Validation --arch dinov3-vitl --exp-name my_run --results-dir .\results\03_unet_pytorch --split-seed 42 --seeds 1 2 3
```

The full local queue, resumable:

```powershell
.venv-torch\Scripts\python.exe 03_unet_pytorch\run_queue.py
```

On the cluster, `sbatch hpc/factorial/run_v6.sbatch` (see [`hpc/README.md`](hpc/README.md)).

### 5. Evaluate and plot

```powershell
.venv-torch\Scripts\python.exe 06_evaluation\02_compare_models.py
```

```powershell
.venv-torch\Scripts\python.exe 06_evaluation\03_prevalence_decomposition.py results\03_unet_pytorch\unet_f16_clean\seed_1
```

```powershell
.venv-torch\Scripts\python.exe 07_figures\01_plot_label_budget.py
```

---

## Experiment families in `results/`

| folder | what it contains | configs |
|---|---|---|
| `00_data_preparation/` | dataset audit table | — |
| `01_geobia/` | GeoBIA validation tables, threshold sensitivity | 6 models |
| `02_unet_tensorflow/` | original Keras U-Net and all hard-negative fine-tuning experiments | — |
| `03_unet_pytorch/` | PyTorch port, scratch and ImageNet encoders, hard-negative variants | 8 |
| `04_label_budget_factorial/` | pretraining domain × label budget {64, 128, 320, 639} | 28 |
| `05_negative_weight_sweep/` | complementary not-waste loss weight {0.1 … 4.0} | 6 |
| `06_pretrained_backbones/` | EfficientNet-B4 (ImageNet/OpenEarthMap), DINOv3, DINOv3-SAT, Satlas; frozen vs fine-tuned | 10 |
| `figures/` | manuscript figures |  |
| `logs/` | training logs |  |

Held constant across every run: 256 × 256 tiles at 3.81 cm, 9 classes (raw labels
1–9), ignore label 99, 250 m spatial blocks with `--split-seed 42`, seeds 1/2/3,
threshold chosen on validation only, both argmax and thresholded decisions
reported. Full table in [`03_unet_pytorch/MODELS.md`](03_unet_pytorch/MODELS.md).

---

## Reproducibility guarantees and known limits

1. **Pretrained weights are verified, never faked.** `rs_backbones.py` raises
   `PretrainedWeightsUnavailable` rather than silently using random
   initialisation, and ViT backbones are additionally compared tensor-by-tensor
   against a fresh random initialisation to catch a silent load failure.
2. **The test partition is spatially clean.** `--clean-test` restricts evaluation
   to the 216 of 222 validation tiles with zero footprint overlap with any
   development tile.
3. **Hardware nondeterminism is material.** The same configuration and seed gives
   F1 0.7121 / FPR 0.0095 on an RTX 4090 and 0.6992 / 0.0066 on an A100 MIG
   slice. FPR-derived quantities are reported as ranges.
4. **Pixel-level and object-level prevalence are never mixed.** They answer
   different questions and are computed by separate code paths
   (`03_unet_pytorch/object_metrics.py`).
5. **Open:** all 53 configurations used RGB only, and so is everything in
   `data/`. The 1,026 28-band tiles in `data/tiled_data/multiband/` are RGB plus
   colour indices (SAVI, VARI) and Haralick texture (variance, entropy,
   dissimilarity) computed for the GeoBIA stage — all derived from the same
   orthomosaic, so they add **no new sensor information**. They are untested, but
   they cannot settle whether the ceiling is informational.

Eight further known issues are recorded in
[`03_unet_pytorch/MODELS.md`](03_unet_pytorch/MODELS.md).

---

## Provenance of this layout

The repository was reorganised on 2026-11-23. Every move is recorded in
`MIGRATION_LOG.tsv` (`source → destination → whether git tracked the rename`), so
any historical path can be resolved. Nothing was deleted: superseded code lives
in `NotNeeded/` with an explanation per group.

## Licence and citation

Code: see [`LICENSE`](LICENSE). Data: see `data/README.md`.
Third-party pretrained weights keep their original licences — the DINOv3 weights
are released under the DINOv3 licence, which is **not** Apache/MIT; review its
terms before publishing derived results.

Citation metadata: [`CITATION.cff`](CITATION.cff).
