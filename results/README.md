# Results

One folder per experiment family. Every folder holds one subfolder per
configuration, each with `config.json` and `metrics_all_seeds.csv`, so any row in
the paper can be traced to the settings that produced it.

| folder | experiments | configs |
|---|---|---|
| `00_data_preparation/` | dataset integrity audit | — |
| `01_geobia/` | GeoBIA validation and threshold sensitivity | 6 models |
| `02_unet_tensorflow/` | Keras U-Net and all hard-negative fine-tuning experiments | — |
| `03_unet_pytorch/` | PyTorch port: scratch, ImageNet encoders, hard-negative variants | 8 |
| `04_label_budget_factorial/` | pretraining domain x label budget {64, 128, 320, 639} | 28 |
| `05_negative_weight_sweep/` | not-waste loss weight {0.1 ... 4.0} | 6 |
| `06_pretrained_backbones/` | EfficientNet-B4 (ImageNet/OpenEarthMap), DINOv3, DINOv3-SAT, Satlas; frozen vs fine-tuned | 10 |
| `figures/` | figures, with the manuscript set in `figures/manuscript/` | |
| `logs/` | training logs | |

53 configurations, about 159 training runs, 0 unresolved failures.

## Reading a metrics file

`metrics_all_seeds.csv` has one row per seed, per decision rule (`argmax` and
`thresh`), per partition (`val` and `test`). Columns include precision, recall,
specificity, FPR, F1, IoU, PR-AUC, prevalence, implied precision at landscape
prevalence, the FPR that precision 0.5 would require, trainable and total
parameters, peak GPU memory, wall-clock time and epochs.

Lead with the **clean test partition** (`--clean-test`: the 216 of 222 validation
tiles with zero footprint overlap with any development tile).

## Two caveats

**Hardware nondeterminism is material.** The same configuration and seed gives
F1 0.7121 / FPR 0.0095 on an RTX 4090 and 0.6992 / 0.0066 on an A100 MIG slice —
a 30% difference in FPR. Report FPR-derived quantities as ranges.

**The factorial's small-n points are provisional.** `04_label_budget_factorial`
drew its label-budget subsets once rather than per seed, so its error bars at
n = 64 and n = 128 capture initialisation variance only, not which tiles were
drawn (n = 64 contains just 22 waste tiles). `--resample-budget-per-seed` is
implemented but the factorial has not been rerun with it.

Interpretation of all of this: [`../docs/FINDINGS.md`](../docs/FINDINGS.md).
