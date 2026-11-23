# NotNeeded — superseded material, kept for provenance

Nothing here is deleted and nothing here is required to reproduce the study.
Each file was moved out of the active pipeline for the reason given below.
`../MIGRATION_LOG.tsv` records every original path.

If you are reproducing the paper, ignore this folder entirely.

## `superseded_trainers/`
Earlier generations of the U-Net training script and the chip-refinement
experiments built on them. `01_train_unet_v4.py` (now
`../02_unet_tensorflow/`) is the last TensorFlow trainer used for a reported
result; `01_train_unet_v5_torch.py` (now `../03_unet_pytorch/`) supersedes all
of them and reproduces the v4 baseline to within 0.002 F1.

Versions v1–v3 also carry a known bug: `--labels` defaulted to `range(1, 11)`,
which treated the no-data fill value 99 as a tenth class. Do not use them.

## `duplicate_finetuning/`
Parallel hard-negative and error-correction fine-tuning experiments, most of
them near-duplicates differing only in a threshold or a loss weight. The
canonical versions are in `../05_hard_negatives/`. `02_initial_inferences/`
holds the first-pass inference and rasterisation chain, superseded by
`../04_inference/`.

## `review_app_versions/`
Twelve iterations of the Shiny waste-review application. The final one is
`../08_annotation_tools/waste_review_app.R`. They are large, mutually
inconsistent, and only the last one produced the annotations that were used.

## `legacy_scripts/`
Whole directory trees that duplicated the numbered pipeline:
`scripts/` (an older copy of the U-Net and conventional-classifier work, plus a
prediction raster set), `analysis/`, and the original `00_datapreparation/`,
`02_custom__unet/`, `03_evaluations/`, `04_inferences/`, `05_finetuning/` and
`06_comparisons/` shells left behind once their canonical files were re-filed.
`ablation_study/` and `approaches/` were empty.

Also here: `hpc_01_create_project_structure.sh` and
`HPC_REORGANIZATION_PLAN.md`, which proposed a repository layout that this
reorganisation replaced.

## `exploratory_r/`
One-off R exploration: false-positive inspection, two `Improved*.R` drafts, and
the data-preparation scratch file.

## `scratch/`
Working files with no analytical role: audit and smoke-test scripts, a
directory listing, a crash screenshot set, an empty log pair, a Word lock file
and `file-list.txt`.

## `reports_docx/`
Interim Word reports written during the analysis. Superseded by the manuscript.

## `duplicate_models/`
Four fitted GeoBIA models that sat in the `01_geobia/` code folder and are
byte-identical to the canonical copies in `data/trained_models/geobia_r/`.
Nothing reads them.
