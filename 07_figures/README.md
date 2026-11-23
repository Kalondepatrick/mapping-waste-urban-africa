# 07 — Manuscript figures

Every figure is built from a saved machine-readable result table, never from
numbers typed by hand. All scripts resolve paths from the repository root, so
run them from there.

```powershell
Rscript 07_figures\04_plot_label_budget.R
```

## Shared style

`00_figure_theme.R` is sourced by every R script here. It provides:

| object | purpose |
|---|---|
| `paper_theme` | `theme_classic`, serif, 10 pt — the manuscript style |
| `save_figure()` | writes vector PDF **and** 600-dpi PNG at 7.5 × 4.3 in |
| `load_all_results()` | reads every `metrics_all_seeds.csv` under `results/` into one tidy table, averaged over seeds |
| `grey_palette`, `mono_shapes` | greyscale-safe encodings that survive black-and-white printing |
| `label_points()` | point labels, via ggrepel when installed |
| `P_LANDSCAPE` | the landscape prevalence constant, 0.00045 |

Figures are written to `results/figures/manuscript/`.

## Scripts

| script | figure | source table |
|---|---|---|
| `01_plot_label_budget.py` | label-budget curves (Python) | `results/04_label_budget_factorial/` |
| `02_plot_frontier.py` | not-waste loss frontier (Python) | `results/05_negative_weight_sweep/` |
| `03_model_comparisons.R` | area-threshold and verification audit, river-distance figures | `data/` vectors |
| `04_plot_label_budget.R` | **label-budget curves by pretraining domain** | all factorial runs |
| `05_plot_backbone_comparison.R` | **pretrained and foundation backbones, frozen vs fine-tuned** | all v6 runs |
| `06_plot_deployability_gap.R` | **the deployability gap across every configuration** | all families |

The R scripts 04–06 are the publication versions and cover the pretrained and
foundation-model results. The two Python scripts predate them and are kept
because they produced the figures used during analysis; they write into the
experiment folders rather than `results/figures/manuscript/`.

## External input

`03_model_comparisons.R` needs the visually verified pile inventory, which lives
in the companion `waste_spatio_temporal` repository. Point the environment
variable at it, or place the shapefile at `data/vectors/verified/`:

```powershell
$env:MWUA_VERIFIED_PILES = "C:\path\to\verefied_waste_piles.shp"
```

## What the figures show

- **`label_budget_curves`** — DINOv3 at 64 tiles (F1 0.7011) matches a scratch
  U-Net at 639 (0.6992): a tenfold reduction in annotation effort. The spread
  across pretraining domains shrinks from 0.141 at n = 64 to 0.061 at n = 639,
  so the advantage is a small-sample effect.
- **`backbone_comparison`** — fine-tuning gains 0.029 F1 on average over a frozen
  encoder. Best overall is DINOv3 ViT-L fine-tuned at 0.7412.
- **`deployability_gap`** — the central result. Of 53 configurations, 3 are
  excluded (recall collapsed, gap undefined); across the remaining 47 the gap
  ranges 16–75×. It is **not** invariant, but nothing approaches 1.0, and the
  16× minimum belongs to a hard-negative variant with recall 0.43.
- **`prevalence_precision`** — why development accuracy misleads: precision
  projected across prevalence for the best and median configurations, with the
  test partition (3.30%) and landscape (0.045%) marked.
