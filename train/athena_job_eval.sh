#!/bin/bash -l

#SBATCH --job-name=domain-specific-llm
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --account=plgttaautopilot2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu
#SBATCH -e slurm/eval_error.log # STDERR
#SBATCH -o slurm/eval_out.log # STDOUT

module load GCCcore/12.3.0
module load Python/3.11.3
module load CUDA/12.1.1

source $SCRATCH/venv/bin/activate
export HF_HOME=$SCRATCH/.cache_dir/huggingface
export PIP_CACHE_DIR=$SCRATCH/.cache_dir/pip

lm_eval --model hf --model_args pretrained=unsloth/Llama-3.2-1B,load_in_4bit=True,peft=Llama-3.2-1B-abstract --log_samples --output_path eval_results --tasks mmlu_pro --device cuda:0 --batch_size auto:2

