import os

# Define the list of experiments and their relative paths
# These paths are relative to your current 'skeleton' directory
experiments = [
    ("Baseline", "reference/baseline.py"),
    ("No-Skip", "structural/no_skip_connections.py"),
    ("No-BatchNorm", "structural/no_batchnorm.py"),
    ("Simple-Conv", "structural/simplified_conv_block.py"),
    ("No-Upsampling", "structural/no_decoder_upsampling.py"),
    ("Filters-16", "filter_sweep/filters_16.py"),
    ("Filters-32", "filter_sweep/filters_32.py"),
    ("Filters-128", "filter_sweep/filters_128.py"),
    ("Kernel-5x5", "kernel_sweep/kernel_5x5.py"),
    ("Dice-Loss", "loss_comparisons/dice_loss_only.py"),
    ("Weighted-Bias", "loss_comparisons/weighted_waste_bias.py"),
]

# The template for the Slurm (.sh) files
# Note: Using \n for line endings to ensure compatibility with the Sapitwa Linux environment
slurm_template = """#!/bin/bash
#SBATCH --job-name=Ablation-{name}
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
python "{script_path}"
"""

def generate_sh_files():
    # Ensure the log directory exists (standard practice for your --output path)
    if not os.path.exists("log"):
        os.makedirs("log")
        print("Created 'log' directory for Slurm outputs.")

    for name, script_path in experiments:
        # Create a lowercase filename, e.g., run_no-skip.sh
        file_name = f"run_{name.lower()}.sh"
        
        # Format the template with specific experiment details
        content = slurm_template.format(name=name, script_path=script_path)
        
        # Write the file using 'w' and newline='' to prevent Windows-style CRLF 
        # which can sometimes cause issues on Linux clusters
        with open(file_name, "w", newline='\n') as f:
            f.write(content)
            
        print(f"✅ Generated: {file_name}")

if __name__ == "__main__":
    generate_sh_files()
    print("\nAll submission scripts have been created in the current directory.")