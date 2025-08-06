#!/bin/bash -l

#SBATCH --job-name=domain-specific-llm-evaluate
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --account=plgttaautopilot2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu
#SBATCH -e logs/eval-%j.err # STDERR
#SBATCH -o logs/eval-%j.out # STDOUT

module load GCCcore/12.3.0
module load Python/3.11.3
module load CUDA/12.1.1

source $SCRATCH/venv/bin/activate
export HF_HOME=$SCRATCH/.cache_dir/huggingface
export PIP_CACHE_DIR=$SCRATCH/.cache_dir/pip

MODEL_NAME="Llama-3.2-1B-abstract"

cd $HOME/domain-specific-llm/eval

lm_eval --model hf \
  --model_args pretrained=$MODEL_NAME \
  --log_samples \
  --output_path eval_results \
  --tasks mmlu \
  --device cuda:0 \
  --batch_size auto:2

lm_eval --model hf \
  --model_args pretrained=$MODEL_NAME \
  --log_samples \
  --output_path eval_results \
  --tasks multimedqa,simplemed \
  --device cuda:0 \
  --batch_size auto:2 \
  --trust_remote_code
