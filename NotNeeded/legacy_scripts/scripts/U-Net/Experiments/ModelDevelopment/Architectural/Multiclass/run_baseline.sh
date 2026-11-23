#!/bin/bash
#SBATCH --job-name=Ablation-Baseline
#SBATCH --output=log/tensor-GPU%j.out
#SBATCH --error=log/tensor-GPU%j.err
#SBATCH -p gpu-nodes
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=8G
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pkalonde@mlw.mw

# Load Python version supporting Tensorflow 2.18.1/2.21.0
module purge
module load Python/3.13.5-GCCcore-14.3.0

# Activate TF environment
source ~/venvs/tf-gpu/bin/activate

# Improve GPU performance
export TF_ENABLE_ONEDNN_OPTS=1
export TF_GPU_THREAD_MODE=gpu_private
export CUDA_VISIBLE_DEVICES=0

# Run the specific experiment script
python "reference/baseline.py"
