#!/bin/bash
#SBATCH --job-name=Architectural-Multiclass
#SBATCH --output=log/tensor-GPU%j.out
#SBATCH --error=log/tensor-GPU%j.err
#SBATCH  -p gpu-nodes
#SBATCH --gres=gpu:1
#SBATCH  -N 1
#SBATCH  -n 10
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pkalonde@mlw.mw


#Load Python version supporting Tensorflow 2.18.1/2.21.0 respectively. Note that version 3.13.5 loads the latest TF on Sapitwa HPC cluster
module purge
#module load Python/3.10.4-GCCcore-11.3.0
module load Python/3.13.5-GCCcore-14.3.0


# Activate your TF environment in your home directory. Make sure your scripts are here or provide the absolute path
source ~/venvs/tf-gpu/bin/activate

# Improve GPU performance - This will help with CPU performance
export TF_ENABLE_ONEDNN_OPTS=1
export TF_GPU_THREAD_MODE=gpu_private
export CUDA_VISIBLE_DEVICES=0

#Your code here
python full_ablationstudy.py
