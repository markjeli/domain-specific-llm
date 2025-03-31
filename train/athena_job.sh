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

module load GCCcore/12.3.0
module load Python/3.11.3
module load CUDA/12.1.1

source $SCRATCH/venv/bin/activate
export HF_HOME=$SCRATCH/.cache_dir/huggingface
export PIP_CACHE_DIR=$SCRATCH/.cache_dir/pip

pip install -U pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

python train_cpt.py

# it is important to start this script with: sbatch athena_job.sh