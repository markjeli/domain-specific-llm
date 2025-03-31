#!/bin/bash -l

#SBATCH --job-name=domain-specific-llm
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --account=plgttaautopilot2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu
#SBATCH -e slurm/error.err # STDERR
#SBATCH -o slurm/out.out # STDOUT

module load Miniconda3/23.3.1-0
conda activate $HOME/.conda/envs/domain-llm-env

module load GCC/13.2.0
module load OpenMPI/4.1.6
module load CUDA/12.4.0

python -m pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"

python train_cpt.py

# it is important to start this script with: sbatch athena_job.sh