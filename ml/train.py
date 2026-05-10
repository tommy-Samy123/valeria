import os
import torch
import torch._dynamo
from datasets import load_dataset
from unsloth import FastVisionModel, get_chat_template
from trl import SFTTrainer, SFTConfig

# Configuration
MODEL_NAME = "unsloth/gemma-4-E2B-it"
DATASET_NAME = "i-am-mushfiq/FirstAidQA"
OUTPUT_DIR = "gemma_4_lora"

def main():
    # Setup optimizations
    torch._dynamo.config.recompile_limit = 64
    
    print("Loading model and processor...")
    model, processor = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )
    
    # Configure LoRA
    model = FastVisionModel.get_peft_model(
        model,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        use_rslora=False,
        target_modules="all-linear",
    )
    
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    def convert_to_conversation(sample):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": sample["question"]}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["answer"]}],
                },
            ]
        }

    dataset = dataset.map(convert_to_conversation)
    
    # Setup Chat Template
    processor = get_chat_template(processor, "gemma-4")
    
    def formatting_func(example):
        text = processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return [text]
    
    # Setup Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
        formatting_func=formatting_func,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_grad_norm=0.3,
            warmup_steps=10,
            max_steps=200,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="steps",
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            max_length=2048,
            packing=False,
        ),
    )
    
    # Print memory stats before
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    # Train
    print("Starting training...")
    trainer_stats = trainer.train()
    
    # Print stats after
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    
    # Save model
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
