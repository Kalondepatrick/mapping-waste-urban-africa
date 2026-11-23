#!/bin/bash
#SBATCH --job-name=waste-negsweep
#SBATCH -p gpu-nodes
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-6%4
#SBATCH --output=logs/negsweep_%A_%a.out
#SBATCH --error=logs/negsweep_%A_%a.out
#
# Hard-negative weight sweep, to trace the precision-recall frontier AT
# LANDSCAPE PREVALENCE. torch_scratch_hn (weight 1.0) reached the required
# FPR but collapsed recall to 0.058; this sweep fills in the curve between
# "useful recall, unusable precision" and "usable precision, no recall".

set -euo pipefail
PROJ=/head/NFS/pkalonde/ImageClassification/mapping-waste-urban-africa
cd "$PROJ"
mkdir -p logs results/05_negative_weight_sweep

module purge
module load Python/3.12.3-GCCcore-13.3.0
source "$PROJ/.venv-torch/bin/activate"
export HF_HOME="$PROJ/.hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

W=$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/factorial/negweights.txt)
NAME="nw_scratch_w${W}"
echo "task ${SLURM_ARRAY_TASK_ID}: neg-weight=$W name=$NAME host=$(hostname)"

if [ -f "results/05_negative_weight_sweep/$NAME/metrics_all_seeds.csv" ]; then
  echo "already complete, skipping"; exit 0
fi

python 03_unet_pytorch/01_train_unet_v5_torch.py \
  --dev-dir  data/tiled_data/partitioned/modelDevelopment \
  --test-dir data/tiled_data/partitioned/Validation \
  --arch scratch \
  --extra-train-dirs data/tiled_data/finetuning_hard_negatives_complete \
  --negative-train-dirs data/tiled_data/complementary_hard_negatives_v2 \
  --max-negatives 2000 \
  --neg-weight "$W" \
  --exp-name "$NAME" \
  --results-dir ./results/05_negative_weight_sweep \
  --split-seed 42 \
  --seeds 1 2 3

echo "task ${SLURM_ARRAY_TASK_ID} finished"
