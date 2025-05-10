from dataclasses import dataclass, field

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
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
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit."},
    )


def main(user_config: ScriptArguments, sft_config: SFTConfig):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=user_config.load_in_8bit, load_in_4bit=user_config.load_in_4bit
    )

    # Use instruct version of the model because it contains chat template and its tokens.
    # Other (harder) option will be to extend previous tokenizer with new tokens and chat template.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        user_config.model_name_or_path,
        device_map="auto",
        quantization_config=quantization_config,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=[
            "lm_head",
            "embed_token",
        ],  # Needed for Llama chat template. It will learn chat template tokens.
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")

    def format_chat_template(row):
        row_json = [
            {"role": "user", "content": row["Patient"]},
            {"role": "assistant", "content": row["Doctor"]},
        ]
        row["text"] = tokenizer.apply_chat_template(row_json)
        return row

    tokenized_chat_dataset = dataset.map(
        format_chat_template,
        num_proc=4,
        remove_columns=["Description", "Patient", "Doctor"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_chat_dataset,
        args=sft_config,
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    trainer.save_model(user_config.save_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    user_config, sft_config = parser.parse_args_into_dataclasses()

    main(user_config, sft_config)
