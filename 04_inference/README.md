# 04 — Inference over new and full-area imagery

Applies a trained model beyond the tiled development set.

| script | does |
|---|---|
| `01_inferences.py` | whole-area probability raster from a Keras model; writes `data/predictions/waste_probability_full.tif` |
| `02_inferences_waste.py` | waste-specific inference and vectorisation; writes `data/predictions/waste_piles_vector/` |
| `03_inferences_diagnostics.py` | per-tile diagnostics for inspecting failures |
| `04_openearthmap_landcover.py` | OpenEarthMap land-cover context raster, used to attribute false positives to land-cover class |
| `05_accuracy.py` | accuracy of a prediction raster against reference polygons |

`OPENEARTHMAP.md` documents the land-cover step, which uses its own environment
(`env/requirements-openearthmap.txt`).

Model and output paths are set at the top of each script. Default inputs are the
Keras models in `data/trained_models/tensorflow/`; outputs go to
`data/predictions/`.

These are the scripts that regenerate the 31.6 GB of `data/predictions/`, which
is why that folder is excluded from the Zenodo package.
