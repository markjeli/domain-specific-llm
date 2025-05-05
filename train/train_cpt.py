from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


@dataclass
class ScriptArguments:
    save_dir: str = field(
        default=None,
        metadata={"help": "Path to save the model."},
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
    dataset_path = "medical_abstracts_train.csv"
    model_name_or_path = "meta-llama/Llama-3.2-1B"
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=user_config.load_in_8bit, load_in_4bit=user_config.load_in_4bit
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", quantization_config=quantization_config
    )

    dataset = load_dataset("csv", data_files=dataset_path, split="train")

    EOS_TOKEN = tokenizer.eos_token
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    def tokenize_function(examples):
        return tokenizer(
            [example + EOS_TOKEN for example in examples["Abstract"]],
            padding="max_length",
            max_length=sft_config.max_seq_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["Abstract", "DOI", "Date"]
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=sft_config,
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    trainer.save_model(user_config.save_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    user_config, sft_config = parser.parse_args_into_dataclasses()
    main(user_config, sft_config)
