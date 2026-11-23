# 03 — U-Net and pretrained backbones, PyTorch

The current trainer and the source of every model result in the paper.

| file | role |
|---|---|
| `01_train_unet_v5_torch.py` | the trainer. Scratch U-Net plus every registry backbone. |
| `rs_backbones.py` | backbone registry with **strict** checkpoint verification |
| `object_metrics.py` | object-level metrics, kept separate from pixel-level |
| `preflight.py` | six-stage verification harness; run before any full training |
| `run_queue.py` | resumable local queue over a set of configurations |
| `MODELS.md` | checkpoint sources, head replacement, held-constant settings, 8 known issues |

## Backbones

`scratch`, `efficientnet-b4-imagenet`, `efficientnet-b4-oem` (OpenEarthMap),
`dinov3-vitl` (LVD-1689M), `dinov3-sat-vitl` (SAT-493M), `satlas-aerial`
(SatlasPretrain single-image aerial). Each runs frozen or fine-tuned.

## Pretrained weights are never faked

`rs_backbones.build_registered` raises `PretrainedWeightsUnavailable` rather than
falling back to random initialisation. For ViT backbones the loaded weights are
additionally compared tensor-by-tensor against a freshly random-initialised model
of the same specification; bit-identity means the load silently failed, and the
configuration is refused rather than reported.

A row labelled "pretrained" had pretrained weights in it, and
`results/06_pretrained_backbones/preflight_report.json` records how many tensors
transferred.

## Order

```powershell
.venv-torch\Scripts\python.exe 03_unet_pytorch\preflight.py --both-freeze-modes
.venv-torch\Scripts\python.exe 03_unet_pytorch\run_queue.py
```

Single configuration:

```powershell
.venv-torch\Scripts\python.exe 03_unet_pytorch\01_train_unet_v5_torch.py --dev-dir data\tiled_data\partitioned\modelDevelopment --test-dir data\tiled_data\partitioned\Validation --arch dinov3-vitl --exp-name my_run --results-dir .\results\03_unet_pytorch --split-seed 42 --seeds 1 2 3
```

Useful flags: `--freeze-backbone`, `--clean-test` (restrict to the 216
non-overlapping test tiles), `--max-train-tiles N` with
`--resample-budget-per-seed`, `--neg-weight W` for the complementary not-waste
loss, `--ignore-label 99`.
