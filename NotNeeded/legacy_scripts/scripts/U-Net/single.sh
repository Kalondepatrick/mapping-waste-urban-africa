#!/bin/bash
#SBATCH --job-name=Ablation-Experiment
#SBATCH --output=GPU-Test%j.out
#SBATCH --error=GPU-Test%j.err
#SBATCH  -p gpu-nodes
#SBATCH --gres=gpu:1
#SBATCH  -N 1
#SBATCH  -n 14
#SBATCH --mem=100G
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pkalonde@mlw.mw
#SBATCH --nodelist=compute4

module purge
#module load CUDA/11.7.0
module load Miniconda3/24.7.1-0

source activate waste_env     


module load python

python -u  shallow_bottleneck.py
