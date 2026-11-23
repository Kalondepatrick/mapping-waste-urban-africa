#!/bin/bash
# Run ONCE on the Sapitwa LOGIN node (it has internet; compute nodes may not).
set -euo pipefail
PROJ=/head/NFS/pkalonde/ImageClassification/mapping-waste-urban-africa
cd "$PROJ"

module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null || echo "using system python3"
python3 --version

[ -d .venv-torch ] || python3 -m venv .venv-torch
./.venv-torch/bin/python -m pip install --upgrade pip
./.venv-torch/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
./.venv-torch/bin/python -m pip install segmentation_models_pytorch timm rasterio pandas scikit-learn matplotlib

echo "=== verifying ==="
./.venv-torch/bin/python - <<'PY'
import torch, timm, segmentation_models_pytorch as smp, rasterio
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("timm", timm.__version__, "smp", smp.__version__)
print("cuda available on login node:", torch.cuda.is_available(), "(False is normal here)")
PY

echo "=== pre-staging model weights into shared cache ==="
export HF_HOME="$PROJ/.hf_cache"
./.venv-torch/bin/python - <<'PY'
import timm
for n in ["vit_small_patch16_dinov3.lvd1689m",
          "vit_large_patch16_dinov3.sat493m",
          "convnext_small.dinov3_lvd1689m"]:
    try:
        timm.create_model(n, pretrained=True, num_classes=0)
        print("cached", n)
    except Exception as e:
        print("FAILED", n, type(e).__name__, str(e)[:120])
import segmentation_models_pytorch as smp
for e in ["resnet34", "efficientnet-b0", "efficientnet-b4"]:
    smp.Unet(encoder_name=e, encoder_weights="imagenet", classes=9)
    print("cached", e)
PY
echo "setup complete"
