from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl import SFTConfig, SFTTrainer


@dataclass
class AdditionalArguments:
    save_dir: str = field(
        default=None,
        metadata={"help": "Path to save the model."},
    )


def main(user_config: AdditionalArguments, sft_config: SFTConfig):
    dataset_path = "medical_abstracts_train.csv"
    model_name_or_path = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

    dataset = load_dataset("csv", data_files=dataset_path, split="train")

    EOS_TOKEN = tokenizer.eos_token
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    def formatting_prompts_func(examples):
        return {"Abstract": [example + EOS_TOKEN for example in examples["Abstract"]]}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    trainer.save_model(user_config.save_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((AdditionalArguments, SFTConfig))
    user_config, sft_config = parser.parse_args_into_dataclasses()
    main(user_config, sft_config)
