from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    abstract_dataset_path = "short_abstract.csv"
    save_dir = "Llama-3.2-1B-medic-abstract"
    save_model = False

    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    # This will not work with torch 2.6.0 and unsloth 2025.2.14
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",  # for 16bit loading
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    dataset = load_dataset(
        "csv", data_files=abstract_dataset_path, split="train[:2500]"
    )
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        return {"Abstract": [example + EOS_TOKEN for example in examples["Abstract"]]}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="Abstract",
        max_seq_length=max_seq_length,
        dataset_num_proc=8,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            num_train_epochs=1,
            learning_rate=5e-5,
            embedding_learning_rate=5e-6,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.00,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
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
