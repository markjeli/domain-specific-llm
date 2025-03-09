import logging
import os

import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set the environment variable
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"


def main():
    save_dir = "Llama-3.2-1B-medic-chatbot"
    save_model = True

    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",  # for 16bit loading
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train[:2500]")

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    def format_chat_template(row):
        row_json = [
            {"role": "user", "content": row["Patient"]},
            {"role": "assistant", "content": row["Doctor"]},
        ]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc=4,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        # dataset_text_field="text",
        
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        
        
        args=SFTConfig(
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logging.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logging.info(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    logging.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logging.info(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    logging.info(f"Peak reserved memory = {used_memory} GB.")
    logging.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logging.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logging.info(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %."
    )

    if save_model:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
