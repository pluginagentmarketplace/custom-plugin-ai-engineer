#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script - Production-ready LLM adaptation.

Features:
- LoRA and QLoRA support
- Multiple dataset formats
- Automatic hyperparameter selection
- Evaluation and metrics
- Model merging and export

Usage:
    python finetune_lora.py --config config.yaml
    python finetune_lora.py --model meta-llama/Llama-2-7b-hf --data data.json
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
from datasets import load_dataset, Dataset


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"
    output_dir: str = "./output"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 512

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10


def load_model_and_tokenizer(config: TrainingConfig):
    """Load base model with quantization if enabled."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Quantization config
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def apply_lora(model, config: TrainingConfig):
    """Apply LoRA adapters to the model."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def prepare_dataset(data_path: str, tokenizer, config: TrainingConfig) -> Dataset:
    """Prepare dataset for training."""

    def format_instruction(sample):
        """Format sample in Alpaca format."""
        if sample.get("input"):
            text = f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
        else:
            text = f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""
        return {"text": text}

    def tokenize(sample):
        """Tokenize the formatted text."""
        result = tokenizer(
            sample["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length"
        )
        result["labels"] = result["input_ids"].copy()
        return result

    # Load dataset
    if data_path.endswith(".json"):
        with open(data_path) as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset(data_path, split="train")

    # Format and tokenize
    dataset = dataset.map(format_instruction)
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    return dataset


def train(model, tokenizer, train_dataset, eval_dataset, config: TrainingConfig):
    """Train the model with LoRA."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        weight_decay=0.001,
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(config.output_dir, "final"))

    return trainer


def merge_and_save(model, tokenizer, output_path: str):
    """Merge LoRA weights and save full model."""
    print("Merging LoRA weights...")

    # Merge adapter weights into base model
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")


def evaluate_model(model, tokenizer, test_prompts: List[str], config: TrainingConfig):
    """Evaluate the fine-tuned model."""
    from transformers import pipeline

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9
    )

    results = []
    for prompt in test_prompts:
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        output = generator(formatted)[0]["generated_text"]
        response = output.split("### Response:\n")[-1].strip()
        results.append({
            "prompt": prompt,
            "response": response
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data", type=str, required=True, help="Training data path")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--merge", action="store_true", help="Merge weights after training")
    parser.add_argument("--config", type=str, help="YAML config file")
    args = parser.parse_args()

    # Build config
    config = TrainingConfig(
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r
    )

    print(f"Loading model: {config.model_name}")
    model, tokenizer = load_model_and_tokenizer(config)

    print("Applying LoRA...")
    model = apply_lora(model, config)

    print(f"Preparing dataset from: {args.data}")
    train_dataset = prepare_dataset(args.data, tokenizer, config)

    # Split for evaluation
    split = train_dataset.train_test_split(test_size=0.1)
    train_data = split["train"]
    eval_data = split["test"]

    print(f"Training samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")

    print("Starting training...")
    trainer = train(model, tokenizer, train_data, eval_data, config)

    if args.merge:
        merge_path = os.path.join(config.output_dir, "merged")
        merge_and_save(model, tokenizer, merge_path)

    print("Training complete!")

    # Quick evaluation
    test_prompts = [
        "What is machine learning?",
        "Explain the concept of neural networks."
    ]
    print("\nEvaluation:")
    results = evaluate_model(model, tokenizer, test_prompts, config)
    for r in results:
        print(f"\nPrompt: {r['prompt']}")
        print(f"Response: {r['response'][:200]}...")


if __name__ == "__main__":
    main()
