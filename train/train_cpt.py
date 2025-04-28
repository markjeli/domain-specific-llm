from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


def main():
    dataset_path = "medical_abstracts_train.csv"
    model_name_or_path = "meta-llama/Llama-3.2-1B"
    output_dir = "outputs"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

    dataset = load_dataset("csv", data_files=dataset_path, split="train[:2500]")

    EOS_TOKEN = tokenizer.eos_token
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    def formatting_prompts_func(examples):
        return {"Abstract": [example + EOS_TOKEN for example in examples["Abstract"]]}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    training_args = SFTConfig(
        dataset_text_field="Abstract",
        max_seq_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        optim="adamw_8bit",
        output_dir=output_dir,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    trainer.save_model("outputs/final_model")


if __name__ == "__main__":
    main()
