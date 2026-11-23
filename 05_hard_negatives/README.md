# 05 — Hard negatives and complementary fine-tuning

The false-positive reduction track. Mines confident errors from a trained model,
builds complementary "not-waste" labels, and fine-tunes against them.

| script | does |
|---|---|
| `00_select_candidate.R` | selects the model and probability threshold to mine from |
| `01_hard_negative_chips.py` | extracts hard-negative chips and complementary masks; writes `data/tiled_data/complementary_hard_negatives_v2/` |
| `02_finetune_complementary_hard_negatives.py` | fine-tunes with the complementary not-waste loss |
| `03_threshold_selection.R` | object-level threshold selection and sensitivity |
| `04_test_three_models_independent_area.py` | evaluates baseline vs complementary vs selective models on the independent test area |

## The complementary loss

Standard cross-entropy on labelled pixels, plus a term on pixels known to be
*not* waste:

```
loss = CE(labelled) + w * mean( -log(1 - p_waste) )  over not-waste pixels
```

The same term is implemented in the PyTorch trainer as `--neg-weight w`, which is
how the sweep in `results/05_negative_weight_sweep/` was produced.

## What it buys

FPR falls by up to 27x, but recall collapses with it (F1 0.712 to 0.097 as the
weight goes 0.0 to 0.5). The implied-precision frontier tops out at 0.099, and at
0.039 with recall held above 0.35. See `docs/FINDINGS.md`.
