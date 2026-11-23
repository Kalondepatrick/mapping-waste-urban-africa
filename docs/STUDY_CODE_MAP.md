# Study code and evidence map

> **Note (2026-11-23).** This document predates the repository
> reorganisation. The canonical locations below have been updated to the
> current layout; the triage and portability sections are kept as a record of
> how the consolidation was decided. For the current structure see the root
> `README.md`, and for the scientific result see `FINDINGS.md`.

## Purpose

This repository supports the study of waste-pile mapping from ultra-fine-resolution drone imagery in Ndirande, Blantyre. The manuscript is the organizing specification for the analytical workflow. Code is retained only when it produces, verifies, or documents a method, result, table, or figure reported in the study.

The Git repository is the canonical codebase. Large imagery, masks, trained models, logs, and intermediate rasters may be stored locally or on the HPC, but their locations must be supplied through configuration rather than hard-coded in analysis scripts.

## Canonical workflow

| Stage | Manuscript coverage | Canonical repository location | Required evidence |
|---|---|---|---|
| 00. Data preparation | Sections 4.1–4.2 | `00_data_preparation/` | AOI, annotations, tiling, spatial split, tile manifest |
| 01. GeoBIA | Section 4.2.1; Tables 1–2 | `01_geobia/` | manual/OTB segments, extracted features, RFE, fitted RF/MLP/SVM models |
| 02. U-Net | Section 4.2.2; Figures 4–5 | `02_unet_tensorflow/` and `03_unet_pytorch/` | baseline architecture, ablation configurations, training histories, selected model |
| 03. Common evaluation | Sections 4.2–4.3; Table 3; Figures 6–8 | `06_evaluation/` | common validation manifest, pixel metrics, context metrics, pile-size metrics |
| 04. Robustness tests | Section 4.4; Figures 9–10 | `05_hard_negatives/` | resolution and illumination perturbation definitions, predictions, metrics |
| 05. Publication outputs | Section 5 | `07_figures/` | scripts that build final tables and figures from saved result tables |

`04_inference/` should contain only inference over new or full-area imagery. Resolution and illumination tests belong in `05_hard_negatives/`, because they test model sensitivity rather than operational inference.

## Verified spatial partition

The experiment reported in the manuscript uses 1,026 image–mask pairs:

- development: 804 tiles
- validation/holdout: 222 tiles
- duplicate filenames across sets: 0
- grid cells shared across sets: 0

The preserved folder membership matches the seed-42 spatial grid assignment for every tile. The reproducible files are in `data/splits/` and `data/tiled_data/spatial_metadata/`:

- `study_spatial_partitions.gpkg`
- `study_grid_partitioning.csv`
- `study_tile_assignments.gpkg`
- `study_tile_manifest.csv`
- `study_partition_summary.csv`

The GeoPackage stores `grid_cells`, `quadrant_regions`, `partition_regions`, and `partition_by_quadrant` layers. The current unpartitioned source-image directory contains only 1,002 files and must not be used to redefine the published experiment.

## Manuscript-to-output checks

| Claim or output | Expected supporting record |
|---|---|
| 114 annotated waste locations | annotation inventory with stable waste-location IDs |
| 1,026 image–mask pairs | `study_tile_manifest.csv` plus file-integrity audit |
| 804/222 development-validation split | saved spatial partition and folder-membership audit |
| Six GeoBIA models | one metrics table keyed by segmentation strategy, feature set, and classifier |
| U-Net baseline and ablations | one experiment table containing configuration, run ID, seed, and validation metrics |
| 84 validation tiles containing waste | explicit waste-positive subset derived from the 222-tile validation manifest |
| Figure 8, `n = 147` piles | table linking each evaluated pile to the original annotation/location ID and any tile fragments |
| Resolution sensitivity | table keyed by native GSD, scale factor, effective GSD, and metric |
| Illumination sensitivity | table keyed by HSV value multiplier and metric |

The difference between 114 annotated locations and 147 evaluated piles must be explained by a recorded transformation, such as multipart features, redigitisation, or splitting at tile boundaries. It should not remain implicit.

## External-file triage

### `C:/Users/Peter/Desktop/TF/UNET`

- `new_.R` is the best candidate for the Figure 4–5 reporting script because it reads measured precision, recall, F1, and IoU from `master.csv`.
- `1111.R` and `final.R` must not be used for scientific results: they derive placeholder precision and recall values by multiplying F1 rather than reading measured metrics.
- `script.R` is empty and should not be migrated.
- The duplicate GeoBIA scripts `01_mask_to_vector.R`, `02_extraction.R`, and `predictions_evaluations.R` are byte-identical to repository copies and should not be copied again.
- `evaluations.py`, `evaluations2.py`, and `evaluations3.py` are competing experimental versions. Their metric definitions and validation paths must be reconciled into one canonical evaluation script before migration.
- The 1.56 GB `data/` subtree is data storage, not source code, and should not be copied into Git.

### HPC tree

- SLURM submission scripts and environment specifications are code and should live beside the analysis they launch or under `hpc/`.
- `.out`, `.err`, checkpoints, fitted models, generated rasters, and figures are artifacts. They should be stored in configured output locations and excluded from source control unless deliberately selected as publication outputs.
- HPC folder names should be mapped to the canonical stages above rather than maintained as a second independent project structure.

## Portability rules

1. No script may contain a user-specific Windows path or `/head/NFS/...` path.
2. Resolve code paths from the repository root.
3. Read large-data and output roots from environment variables or a local, ignored configuration file.
4. Use the same relative dataset structure locally and on HPC after resolving the configured data root.
5. Every reported table and figure must be generated from a saved machine-readable result table.
6. Scratch scripts, duplicate versions, and logs are not part of the canonical workflow.
7. Existing files are copied into the canonical layout only after their role and provenance have been verified; external originals are not deleted during consolidation.

Suggested environment variables:

- `MWUA_DATA_ROOT`: imagery, masks, vectors, and other large inputs
- `MWUA_OUTPUT_ROOT`: generated metrics, predictions, models, logs, and figures
- `MWUA_PROJECT_ROOT`: optional override for non-Git HPC deployments

