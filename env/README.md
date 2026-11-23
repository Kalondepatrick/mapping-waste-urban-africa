# Environments

Three environments are used. They do not need to coexist.

| file | used by | notes |
|---|---|---|
| `requirements-torch.txt` | `03_unet_pytorch/`, `06_evaluation/`, `07_figures/` | the current stack; all reported model results |
| `requirements-tensorflow.txt` | `02_unet_tensorflow/`, `04_inference/`, `05_hard_negatives/` | the original stack |
| `requirements-openearthmap.txt` | `04_inference/04_openearthmap_landcover.py` | land-cover context only |
| `r-packages.R` | `01_geobia/`, `00_data_preparation/`, `07_figures/03_model_comparisons.R` | GeoBIA and figures |

## PyTorch

```powershell
python -m venv .venv-torch
.venv-torch\Scripts\python.exe -m pip install -r env\requirements-torch.txt
```

The pinned `torch==2.11.0+cu128` build requires CUDA 12.8 and the PyTorch index:

```powershell
.venv-torch\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

On CPU, drop the `+cu128` suffix and expect training to be impractically slow;
the preflight and evaluation steps still run.

## TensorFlow

TensorFlow has had no native Windows GPU support since 2.10, which is why the
trainer was ported to PyTorch. The TensorFlow scripts are retained because the
published baseline was produced with them. Run them under WSL2 or Linux for GPU.

## R

```r
source("env/r-packages.R")
```

## HPC (Sapitwa)

The cluster venv was built against a module-loaded Python. Without the matching
module the venv interpreter fails with `libffi.so.8: cannot open shared object
file`, so every submission script must load it first:

```bash
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "$PROJ/.venv-torch/bin/activate"
```

Compute nodes have no outbound network. Pre-stage Hugging Face checkpoints on
the login node and export `HF_HUB_OFFLINE=1`; see `hpc/README.md`.
