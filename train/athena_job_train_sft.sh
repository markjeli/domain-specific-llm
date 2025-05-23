#!/bin/bash -l

#SBATCH --job-name=domain-specific-llm
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --account=plgttaautopilot2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu
#SBATCH -e slurm/train_error.log # STDERR
#SBATCH -o slurm/train_out.log # STDOUT

module load GCCcore/12.3.0
module load Python/3.11.3
module load CUDA/12.1.1

source $SCRATCH/venv/bin/activate
export HF_HOME=$SCRATCH/.cache_dir/huggingface
export PIP_CACHE_DIR=$SCRATCH/.cache_dir/pip
export WANDB_PROJECT=magisterka

python train_sft.py \
  --save_dir outputs/final_model \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --load_in_8bit False \
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
  --output_dir $SCRATCH/model-outputs \
  --report_to wandb \
  --run_name llama-3.2-1B-abstract-4bit
