# 00 — Data preparation

Defines the study grid, the spatial partition and the tile inventory. Run first.

| script | language | does |
|---|---|---|
| `01_grid_creation.R` | R | builds the study grid over the AOI and writes `data/vectors/study_grid/` |
| `02_create_spatial_split.R` | R | assigns tiles to development / validation using 250 m blocks on tile centroids, seed 42; writes `data/splits/` |
| `03_audit_dataset.R` | R | integrity audit: duplicate filenames, shared grid cells, image–mask pairing; writes `results/00_data_preparation/stage00_dataset_audit.csv` |
| `04_generate_background_and_hard_negative_chips.py` | Python | samples background and candidate hard-negative chips from `data/optical.tiff` |

The partition produced here is the definition of the experiment: **1,026
image–mask pairs, 804 development, 222 validation, 0 duplicate filenames and 0
shared grid cells**. It is already materialised in
`data/tiled_data/partitioned/` and `data/splits/`, so re-running these scripts is
only necessary if you are re-tiling from source imagery.
