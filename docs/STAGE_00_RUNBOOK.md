# Stage 00 runbook: data preparation

## Goal

Migrate, execute, and verify data preparation before reorganizing later analytical stages. Stage 00 must recreate or validate the exact dataset used by the manuscript without silently defining a new experiment.

## Target folders

```text
analysis/00_data_preparation/
├── 01_annotations/
├── 02_tiling/
├── 03_spatial_partition/
└── 04_dataset_audit/
```

## Migration order

### 1. Annotations

Move only the authoritative scripts that inventory MapSwipe candidates, verify waste locations, and construct the reference annotations. Preserve stable location and pile identifiers.

Expected check: 114 annotated waste locations, with the transformation to the 147 pile records used in the size analysis explicitly documented.

### 2. Tiling

Move the scripts that create aligned 256 × 256 image and mask tiles. The scripts must retain georeferencing, use nearest-neighbour resampling for categorical masks, and write a tile inventory.

Expected checks:

- image and mask filenames match one-to-one;
- dimensions and coordinate reference systems match;
- masks contain only documented class values;
- the published dataset inventory contains 1,026 pairs.

### 3. Spatial partition

Use the seed-42 spatial partition that produced the published experiment. Do not create a new random training-validation split.

Authoritative local source:

```text
00_datapreparation/01_spatial_partitioning/01_grid_creation.R
```

Target canonical name:

```text
analysis/00_data_preparation/03_spatial_partition/01_create_spatial_split.R
```

Required saved outputs:

```text
data/splits/study_spatial_partitions.gpkg
data/splits/study_grid_partitioning.csv
data/splits/study_tile_assignments.gpkg
data/splits/study_tile_manifest.csv
data/splits/study_partition_summary.csv
```

Expected checks:

- development: 804 tiles;
- validation: 222 tiles;
- duplicate filenames across sets: 0;
- grid cells shared across sets: 0.

The saved GeoPackage must contain `grid_cells`, `quadrant_regions`, `partition_regions`, and `partition_by_quadrant`.

### 4. Dataset audit

The final Stage 00 script should fail when any required invariant is violated. It should report at least:

- total pairs and counts by set;
- missing or extra images, masks, and multiband files;
- duplicate filenames;
- spatial overlap between development and validation cells;
- class values and waste-positive tile counts;
- file checksums for the frozen manifest and partition layers.

Expected check: the validation subset contains 84 tiles with annotated waste.

## Execution discipline

Run scripts from the project root and capture their console output in `logs/00_data_preparation/`. Record the Git commit, configuration, random seed, and package versions with every completed Stage 00 run.

Do not proceed to Stage 01 until the Stage 00 audit passes. Do not delete the old HPC Stage 00 files until the canonical scripts and outputs reproduce all expected counts.

