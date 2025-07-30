#!/bin/bash -l

#SBATCH --job-name=domain-specific-llm-train-sft
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --account=plgttaautopilot2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu
#SBATCH -e logs/train_sft-%j.err # STDERR
#SBATCH -o logs/train_sft-%j.out # STDOUT

module load GCCcore/12.3.0
module load Python/3.11.3
module load CUDA/12.1.1

source $SCRATCH/venv/bin/activate
export HF_HOME=$SCRATCH/.cache_dir/huggingface
export PIP_CACHE_DIR=$SCRATCH/.cache_dir/pip
export WANDB_PROJECT=magisterka

cd $HOME/domain-specific-llm/train

python train_sft.py \
  --save_dir $SCRATCH/final_models/sft/Llama-3.2-1B-Instruct-conversation \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --load_in_4bit True \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant True \
  --max_seq_length 2048 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --optim adamw_8bit \
  --packing True \
  --bf16 True \
  --eval_strategy steps \
  --save_strategy epoch \
  --output_dir $SCRATCH/model-outputs/sft/Llama-3.2-1B-Instruct-conversation \
  --report_to wandb \
  --run_name llama-3.2-1B-Instruct
