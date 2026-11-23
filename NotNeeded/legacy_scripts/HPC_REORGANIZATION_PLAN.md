# HPC reorganization plan

## Objective

Create a clean HPC copy of the manuscript workflow without changing the published experiment or losing unique scripts and results. The local Git repository and HPC checkout must contain the same canonical code. Machine-specific data and output paths remain outside the code history.

Do not reorganize the active HPC project in place. Build and validate a clean sibling directory first, then archive the old directory after the new copy reproduces the required checks.

## Target layout

```text
mapping-waste-urban-africa/
├── analysis/
│   ├── 00_data_preparation/
│   ├── 01_geobia/
│   ├── 02_unet/
│   ├── 03_evaluation/
│   │   ├── 01_overall/
│   │   ├── 02_landcover_context/
│   │   └── 03_waste_pile_size/
│   ├── 04_sensitivity/
│   │   ├── 01_spatial_resolution/
│   │   └── 02_illumination/
│   └── 05_reporting/
├── config/
├── environments/
├── hpc/
├── src/
│   ├── R/
│   └── python/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
├── outputs/
│   ├── metrics/
│   ├── tables/
│   ├── figures/
│   ├── predictions/
│   └── models/
├── logs/
├── tests/
└── docs/
```

Large `data/`, `outputs/models/`, `outputs/predictions/`, and `logs/` content should normally be stored outside Git. They may remain under the project directory on HPC if that is operationally convenient, provided they are ignored and their paths are configurable.

## Phase 1: Freeze and inventory the current HPC project

Before copying, moving, or deleting files:

1. Record the current Git branch and commit, if available.
2. Save a complete sorted file listing with sizes and modification dates.
3. Create SHA-256 manifests separately for code, data, trained models, and result tables.
4. Record the Python, TensorFlow, CUDA, R, GDAL, `sf`, `terra`, and OTB versions.
5. Record successful SLURM job IDs associated with the manuscript results.

The inventories should be stored outside the directory being reorganized and copied into `docs/provenance/` after verification.

## Phase 2: Create a clean sibling checkout

Create a new sibling directory such as:

```text
mapping-waste-urban-africa-clean/
```

Populate it from the canonical Git repository. Do not start by copying the complete old HPC tree, because doing so also copies duplicated scripts, logs, checkpoints, and scratch outputs.

## Phase 3: Reconcile code

For every `.R`, `.py`, and `.sh` file on HPC:

1. Match it to the manuscript stage it supports.
2. Compare its SHA-256 hash with the local version.
3. If identical, keep the Git version only.
4. If different, review the diff and select or merge the authoritative logic.
5. Rename generic files such as `evaluations2.py` only after identifying their inputs and outputs.
6. Preserve provenance in Git history; do not retain numbered duplicate copies in the final workflow.

The Python scripts that produce `landcover_proximity_metrics.csv`, `class1_true_accuracy_by_manual_breaks_unet.csv`, and `unet_per_waste_pile_metrics.csv` must be identified before the generic `evaluations*.py` files are removed or renamed.

## Phase 4: Transfer required data and artifacts

Transfer by category rather than copying the old project wholesale.

### Required study data

- original annotations and stable waste-location IDs
- validation imagery and masks
- GeoBIA manual and OTB segments
- extracted spectral and texture features
- the frozen 804/222 spatial partition metadata

### Required result evidence

- final model configuration and selected checkpoint
- machine-readable metrics underlying Tables 1–3
- per-tile context metrics underlying Figure 6
- per-pile metrics underlying Figure 8
- ablation metrics underlying Figures 4–5
- resolution and illumination sweep metrics underlying Figures 9–10

### Do not transfer into Git

- `.RData`, `.Rhistory`, `.RDataTmp*`
- `__pycache__/` and `*.pyc`
- routine `*.out`, `*.err`, and temporary logs
- intermediate checkpoints when a final selected model exists
- duplicate generated overlays and scratch figures
- personal IDE state

## Phase 5: Configure local and HPC paths

No analysis script should contain `C:/Users/...` or `/head/NFS/...` directly. Use the same relative structure and resolve these roots through configuration:

- `MWUA_PROJECT_ROOT`
- `MWUA_DATA_ROOT`
- `MWUA_OUTPUT_ROOT`

Keep a documented configuration template in Git. Keep machine-specific values in an ignored local file or environment variables.

## Phase 6: Validation gates

The clean HPC project is acceptable only when all of the following checks pass:

- 1,026 image-mask pairs are represented in the published experiment manifest.
- development contains 804 pairs and validation contains 222 pairs.
- no filename occurs in both sets.
- no spatial grid cell occurs in both sets.
- the waste-positive validation subset contains the documented 84 tiles.
- all seven reported models have traceable configuration and metric records.
- the 147 observations in the pile-size figure can be traced to stable pile or fragment IDs.
- final tables and figures can be regenerated from machine-readable results.
- local and HPC runs use the same Git commit and analysis configuration.

## Phase 7: Archive, then remove

After validation:

1. Make the old HPC project read-only or rename it as a dated archive.
2. Retain it through at least one successful end-to-end rerun of the clean project.
3. Delete only files already classified as duplicate, reproducible intermediate, cache, or obsolete log output.
4. Keep the inventory and checksum manifests after deletion.

Deletion targets must be reviewed as an explicit list. Broad wildcard deletion must not be used against the project root.

