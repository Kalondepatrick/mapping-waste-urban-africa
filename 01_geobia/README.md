# 01 — GeoBIA baseline

Object-based image analysis: segment, extract features, classify. This is the
conventional-remote-sensing comparator for the deep-learning stages.

Run in order (all R):

| script | does |
|---|---|
| `01_mask_vectorization.R` | converts label rasters to polygons |
| `02_extraction_to_mask.R` | extracts spectral and texture statistics to the mask polygons |
| `03_creation_of_segments.R` | segmentation; writes `data/tiled_data/segments/` |
| `04_extraction_to_segments.R` | extracts features to segments, including Haralick texture from `data/tiled_data/haralick/` |
| `05_model_development.R` | recursive feature elimination and model fitting (RF, SVM, ANN, NB); writes `data/trained_models/geobia_r/` |
| `06_mask_to_vector.R` | vectorises predictions for evaluation |
| `07_extraction.R` | extracts prediction attributes to the validation units |
| `08_predictions_evaluations.R` | accuracy tables; writes `results/01_geobia/` |
| `09_validation.R` | loads the fitted models for inspection |

Outputs land in `results/01_geobia/`:
`geobia_multiclass_validation.csv`, `geobia_validation_summary.csv`,
`model_performance_comparison.csv`, `minimum_area_threshold_sensitivity.csv`,
`segmentation_optimization_log.csv`, `waste_context_groups.csv`.

Install R dependencies with `source("env/r-packages.R")`.
