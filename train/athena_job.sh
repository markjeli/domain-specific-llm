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

module load GCC/11.3.0
module load OpenMPI/4.1.4
module load CUDA/11.7.0
module load PyTorch/1.13.1-CUDA-11.7.0

python -m pip install unsloth
python -m pip install transformers
python -m pip install datasets
python -m pip install trl

python train_cpt.py

# it is important to start this script with: sbatch athena_job.sh