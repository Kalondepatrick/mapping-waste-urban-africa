#!/bin/bash
#SBATCH --job-name=waste-v6
#SBATCH -p gpu-nodes
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=1-10%4
#SBATCH --output=logs/v6_%A_%a.out
#SBATCH --error=logs/v6_%A_%a.out
#
# RS-specific pretrained backbones, frozen vs fine-tuned.
# Weights are pre-staged on the login node; strict verification in
# rs_backbones.py raises rather than falling back to random init.

set -euo pipefail
PROJ=/head/NFS/pkalonde/ImageClassification/mapping-waste-urban-africa
cd "$PROJ"
mkdir -p logs results/06_pretrained_backbones

module purge
module load Python/3.12.3-GCCcore-13.3.0
source "$PROJ/.venv-torch/bin/activate"
export HF_HOME="$PROJ/.hf_cache"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$PROJ/.torch_cache"
export PYTHONUNBUFFERED=1

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/factorial/v6_jobs.txt)
ARCH=$(echo "$LINE" | awk '{print $1}')
MODE=$(echo "$LINE" | awk '{print $2}')
NAME="v6_$(echo "$ARCH" | tr -d '-')_${MODE}"
FREEZE=""
[ "$MODE" = "frozen" ] && FREEZE="--freeze-backbone"

echo "task ${SLURM_ARRAY_TASK_ID}: arch=$ARCH mode=$MODE name=$NAME host=$(hostname)"
if [ -f "results/06_pretrained_backbones/$NAME/metrics_all_seeds.csv" ]; then
  echo "already complete, skipping"; exit 0
fi

python 03_unet_pytorch/01_train_unet_v5_torch.py \
  --dev-dir  data/tiled_data/partitioned/modelDevelopment \
  --test-dir data/tiled_data/partitioned/Validation \
  --arch "$ARCH" $FREEZE \
  --oem-weights data/pretrained_weights/openearthmap/RGB_Real_5_u-efficientnet-b4.pth \
  --exp-name "$NAME" \
  --results-dir ./results/06_pretrained_backbones \
  --split-seed 42 \
  --seeds 1 2 3

echo "task ${SLURM_ARRAY_TASK_ID} finished"
