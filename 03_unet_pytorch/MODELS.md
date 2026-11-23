# Pretrained backbones: sources, verification, decisions

Record for reproducibility. Every row in the model-comparison table should be
traceable to an entry here.

## Hard rule

If pretrained weights are requested and cannot be loaded, the code **raises**
(`rs_backbones.PretrainedWeightsUnavailable`). It never falls back to random
initialisation. A row labelled "pretrained" had pretrained weights in it, and
`preflight_report.json` records how many tensors were transferred.

For ViT backbones this is checked independently: the loaded backbone is
compared tensor-by-tensor against a freshly random-initialised model of the
same specification. If they are bit-identical, loading silently failed and we
raise rather than report the row.

## Configurations

| key | architecture | pretraining | source |
|---|---|---|---|
| `efficientnet-b4-imagenet` | U-Net, EfficientNet-B4 encoder | ImageNet-1k | `segmentation_models_pytorch` |
| `efficientnet-b4-oem` | U-Net, EfficientNet-B4 encoder, scSE decoder | OpenEarthMap (aerial, ~0.25–0.5 m) | `data/pretrained_weights/openearthmap/RGB_Real_5_u-efficientnet-b4.pth` |
| `dinov3-vitl` | DINOv3 ViT-L/16 + U-Net decoder | LVD-1689M (web images) | timm `vit_large_patch16_dinov3.lvd1689m` |
| `dinov3-sat-vitl` | DINOv3 ViT-L/16 + U-Net decoder | SAT-493M (satellite) | timm `vit_large_patch16_dinov3.sat493m` |
| `satlas-aerial` | SatlasPretrain Swin-v2-B + U-Net decoder | SatlasPretrain single-image aerial | `satlaspretrain_models`, `Aerial_SwinB_SI` |

Each runs in two modes: **frozen** encoder and **fine-tuned** encoder
(warm-up with the encoder frozen, then fine-tune at a lower learning rate).

## Head replacement

All original classification/task heads are discarded and replaced with a fresh
9-class convolutional segmentation head.

- smp models: `segmentation_head` is constructed for 9 classes directly.
- OpenEarthMap: the checkpoint's own 9-class land-cover head is **dropped**
  (its classes are bareland/rangeland/developed/road/tree/water/cropland/
  building, not ours). Encoder and decoder are transferred.
- timm ViTs: built with `num_classes=0` so no classification head exists; the
  patch tokens feed a U-Net decoder with a fresh head.
- Satlas: task heads dropped; FPN features feed a U-Net decoder.

## Held constant across every model

| | value |
|---|---|
| tiles | 256 × 256, 3.81 cm GSD |
| classes | 9 (raw labels 1–9) |
| ignore label | **99**, excluded from loss and from every metric |
| spatial split | 250 m blocks, `--split-seed 42` — identical tiles for every model |
| label budgets | 64 / 128 / 320 / 639 training tiles |
| seeds | 1, 2, 3 |
| test partition | independent 222-tile `Validation` set; `--clean-test` restricts to the 216 tiles with **zero footprint overlap** with development tiles |
| threshold rule | `t*` maximising F1 on the validation split only, never on test |
| decision rules | both `argmax` and `thresh` reported |

## Reported metrics

**Pixel level** — precision, recall, specificity, FPR, F1, IoU, PR-AUC
(average precision), prevalence, implied precision at landscape prevalence,
and the FPR required for precision 0.5 at that prevalence.

**Object level** (`object_metrics.py`, kept strictly separate) — object
precision, recall, F1 at IoU ∈ {0.1, 0.25, 0.5}; false-positive objects per
km²; reference objects per km²; assessed area.

**Resources** — trainable parameters, total parameters, peak GPU memory,
wall-clock training time, epochs run.

## Known issues and decisions

1. **Label 99 is a no-data fill, not a class.** It is 0.03% of pixels.
   The original training treated it as a 10th class; `class_lookup.csv` shows
   the real scheme is 1–10 where 10 = "Unidentified" (absent from these tiles).
   Decision: `--labels 1..9 --ignore-label 99`.

2. **Six test tiles overlap development tiles** (two by 37% of their area).
   `results/03_unet_pytorch/test_tiles_clean.csv` lists the 216 clean tiles. Decision:
   report both, lead with the clean partition.

3. **Label-budget subsets were initially fixed across seeds**, so error bars
   captured initialisation variance only, not which tiles were drawn. This is
   a confound at small n (n=64 contains only 22 waste tiles). Decision:
   resample the subset per seed.

4. **Hardware nondeterminism is material.** Identical config and seed gives
   F1 0.7121 / FPR 0.0095 on an RTX 4090 and 0.6992 / 0.0066 on an A100 MIG —
   a 30% FPR difference. Decision: report the FPR gap as a range, not a point
   estimate.

5. **Object counting is per tile**, so a pile crossing a tile boundary is
   counted once per tile it touches. Tiles are 9.75 m and piles are typically
   much smaller, so few objects are affected, but it biases object counts
   upward.

6. **DINOv3 weights are released under the DINOv3 licence**, not Apache/MIT.
   Review the terms before publishing derived results.

7. **The Xet CDN (`us.aws.cdn.hf.co`) is unreachable from the local Windows
   machine**, which blocks the larger timm checkpoints there. The Sapitwa
   login node fetches them without trouble, so large-backbone runs belong on
   the cluster. Setting `HF_HUB_DISABLE_XET=1` is the local workaround to try.

8. **`torchgeo` (needed for DOFA) fails to install** with a dependency hash
   mismatch. No pip config or environment variable forces hashes, so a
   corrupted cache entry is the likely cause; retry with `--no-cache-dir`.
   Hash verification was **not** disabled.

## Engineering loop

`preflight.py` runs: dependencies → checkpoint load → forward pass → loss and
gradient → small-scale training on real tiles → resource measurement. Results
land in `results/06_pretrained_backbones/preflight_report.json`; passing configs are skipped on
re-run, and failures are recorded with a diagnosis rather than skipped
silently. Only configs that pass proceed to full training.
