# 06 — Evaluation

| script | does |
|---|---|
| `01_evaluations.py` | per-tile pixel metrics for a Keras model |
| `02_compare_models.py` | assembles every `results/03_unet_pytorch/*/metrics_all_seeds.csv` into one comparison table |
| `03_prevalence_decomposition.py` | decomposes the development-to-landscape precision shortfall into base-rate and confuser-shift components |
| `04_accuracy_by_landcover/` | accuracy stratified by land-cover context (R) |
| `05_accuracy_by_pile_size/` | accuracy stratified by waste-pile size (R) |

## Pixel level and object level are never mixed

They answer different questions and are computed by separate code paths. A pixel
precision of 0.72 and an object precision of 0.02 can both be true of the same
map:

- **pixel** — how much of the predicted waste *area* is really waste
- **object** — how many of the predicted *piles* are really piles

Object metrics live in `../03_unet_pytorch/object_metrics.py`, reported at IoU in
{0.1, 0.25, 0.5} with false-positive objects per square kilometre. Known bias:
connected components are computed per tile, so a pile crossing a tile boundary is
counted once per tile it touches. Tiles are 9.75 m and piles are usually far
smaller, so few objects are affected, but object counts are biased upward.

## Prevalence

Test-partition prevalence 3.30%, validation 5.42%, landscape 0.045%. Precision is not
prevalence-invariant; recall is. `03_prevalence_decomposition.py` reports the
implied precision at landscape prevalence and the FPR that precision 0.5 would
require.

```powershell
.venv-torch\Scripts\python.exe 06_evaluation\03_prevalence_decomposition.py results\03_unet_pytorch\unet_f16_clean\seed_1
```
