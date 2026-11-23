#!/bin/bash
#SBATCH --job-name=waste-factorial
#SBATCH -p gpu-nodes
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-28%4
#SBATCH --output=logs/factorial_%A_%a.out
#SBATCH --error=logs/factorial_%A_%a.out
#
# Pretraining domain x label budget factorial.
# %4 caps concurrency at 4, matching the 4 MIG slices (2 per node on
# compute3/compute4).
#
# NOTE: .venv-torch was built against Python/3.12.3-GCCcore-13.3.0. Without
# that module the venv python dies with "libffi.so.8: cannot open shared
# object file". The module load below is required, not cosmetic.

set -euo pipefail
PROJ=/head/NFS/pkalonde/ImageClassification/mapping-waste-urban-africa
cd "$PROJ"
mkdir -p logs results/04_label_budget_factorial

module purge
module load Python/3.12.3-GCCcore-13.3.0
source "$PROJ/.venv-torch/bin/activate"

# Weights are pre-staged on the login node; compute nodes stay offline.
export HF_HOME="$PROJ/.hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/factorial/jobs.txt)
ARCH=$(echo "$LINE" | awk '{print $1}')
BUDGET=$(echo "$LINE" | awk '{print $2}')
NAME="fx_$(echo "$ARCH" | tr -d '-')_n${BUDGET}"

echo "task ${SLURM_ARRAY_TASK_ID}: arch=$ARCH budget=$BUDGET name=$NAME host=$(hostname)"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

if [ -f "results/04_label_budget_factorial/$NAME/metrics_all_seeds.csv" ]; then
  echo "already complete, skipping"; exit 0
fi

python 03_unet_pytorch/01_train_unet_v5_torch.py \
  --dev-dir  data/tiled_data/partitioned/modelDevelopment \
  --test-dir data/tiled_data/partitioned/Validation \
  --arch "$ARCH" \
  --max-train-tiles "$BUDGET" \
  --exp-name "$NAME" \
  --results-dir ./results/04_label_budget_factorial \
  --split-seed 42 \
  --seeds 1 2 3

echo "task ${SLURM_ARRAY_TASK_ID} finished"
