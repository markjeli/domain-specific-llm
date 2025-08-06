from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from trl import SFTConfig, SFTTrainer


@dataclass
class ScriptArguments:
    save_dir: str = field(
        default=None,
        metadata={"help": "Path to save the model."},
    )
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B",
        metadata={"help": "Hugging Face model ID or path to the local model."},
    )
    adapter_path: str = field(
        default=None,
        metadata={
            "help": "Path to the adapter model. If provided, the model will be loaded with this adapter."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit."},
    )
    bnb_4bit_quant_type: str = field(
        default="fp4",
        metadata={
            "help": "Quantization type for 4-bit quantization. Options: 'fp4', 'nf4'."
        },
    )
    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={
            "help": "Use double quantization for 4-bit quantization. Recommended for better performance."
        },
    )


def main(user_config: ScriptArguments, sft_config: SFTConfig):
    if user_config.load_in_4bit or user_config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=user_config.load_in_8bit,
            load_in_4bit=user_config.load_in_4bit,
            bnb_4bit_quant_type=user_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=user_config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    # Use instruct version of the model because it contains chat template and its tokens.
    # Other (harder) option will be to extend previous tokenizer with new tokens and chat template.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        user_config.model_name_or_path,
        device_map="auto",
        quantization_config=quantization_config,
    )

    if user_config.adapter_path:
        model = PeftModel.from_pretrained(
            model, user_config.adapter_path, device_map="auto"
        )
        model = model.merge_and_unload()
        model.save_pretrained(f"{user_config.save_dir}/merged_model")
        tokenizer.save_pretrained(f"{user_config.save_dir}/merged_model")

        model = AutoModelForCausalLM.from_pretrained(
            f"{user_config.save_dir}/merged_model",
            device_map="auto",
            quantization_config=quantization_config,
        )

    if quantization_config is not None:
        lora_config = LoraConfig(
            r=256,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    def format_chat_template(row):
        row_json = [
            {"role": "user", "content": row["Patient"]},
            {"role": "assistant", "content": row["Doctor"]},
        ]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    train_chat_dataset = train_dataset.map(
        format_chat_template,
        num_proc=4,
        remove_columns=["Description", "Patient", "Doctor"],
    )

    validation_dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split="train[:15%]")

    def format_orca_dataset(row):
        messages = [
            {"role": m["from"], "content": m["value"]} for m in row["conversations"]
        ]
        row["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return row

    validation_chat_dataset = validation_dataset.map(
        format_orca_dataset, remove_columns=validation_dataset.column_names
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_chat_dataset,
        eval_dataset=validation_chat_dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    trainer.save_model(user_config.save_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    user_config, sft_config = parser.parse_args_into_dataclasses()

    main(user_config, sft_config)
