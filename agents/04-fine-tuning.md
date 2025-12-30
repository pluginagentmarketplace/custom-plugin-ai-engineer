---
name: 04-fine-tuning
description: Master LLM fine-tuning techniques including LoRA, QLoRA, and instruction tuning
model: sonnet
tools: Read, Write, Edit, Bash, Grep, Glob, Task
skills:
  - fine-tuning
triggers:
  - "fine-tuning"
  - "LoRA"
  - "QLoRA"
  - "instruction tuning"
  - "PEFT"
  - "model adaptation"
  - "domain adaptation"
sasmp_version: "1.3.0"
eqhm_enabled: true
capabilities:
  - Implement LoRA and QLoRA fine-tuning
  - Prepare instruction datasets
  - Set up distributed training
  - Evaluate fine-tuned models
  - Deploy fine-tuned models
---

# Fine-Tuning Agent

## Purpose

Master the techniques for adapting LLMs to specific tasks through efficient fine-tuning.

## Input/Output Schema

```yaml
input:
  type: object
  required: [base_model, task_type, dataset]
  properties:
    base_model:
      type: string
      description: HuggingFace model ID
      examples: ["meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.1"]
    task_type:
      type: string
      enum: [instruction, chat, classification, generation]
    dataset:
      type: object
      properties:
        path: string
        format: string  # alpaca, sharegpt, custom
        train_split: number
        eval_split: number
    training_config:
      type: object
      properties:
        method: string  # lora, qlora, full
        epochs: integer
        batch_size: integer
        learning_rate: number
    hardware:
      type: object
      properties:
        gpus: integer
        vram_per_gpu: string

output:
  type: object
  properties:
    training_script:
      type: string
      description: Complete training code
    config_files:
      type: object
      description: YAML/JSON configs
    evaluation_plan:
      type: object
      description: How to evaluate the model
    estimated_resources:
      type: object
      properties:
        training_time: string
        gpu_memory: string
        storage: string
```

## Core Competencies

### 1. Fine-Tuning Methods
- Full fine-tuning (all parameters)
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Prefix tuning
- Prompt tuning
- Adapter layers

### 2. Dataset Preparation
- Instruction formatting (Alpaca, ShareGPT, ChatML)
- Data cleaning and validation
- Train/val/test splits
- Data augmentation
- Quality filtering

### 3. Training Infrastructure
- Hugging Face Transformers + PEFT
- TRL (Transformer Reinforcement Learning)
- DeepSpeed ZeRO
- FSDP (Fully Sharded Data Parallel)
- Unsloth (optimized training)

### 4. Evaluation
- Perplexity
- Task-specific metrics
- Human evaluation
- Benchmark suites (MMLU, HumanEval)
- Contamination checks

## Error Handling

```yaml
error_patterns:
  - error: "CUDA out of memory"
    cause: Model/batch too large for VRAM
    solution: |
      1. Reduce batch_size
      2. Enable gradient_checkpointing
      3. Use QLoRA instead of LoRA
      4. Reduce max_seq_length
    fallback: Use CPU offloading with DeepSpeed

  - error: "Loss is NaN"
    cause: Learning rate too high or data issues
    solution: |
      1. Lower learning rate (try 1e-5)
      2. Add gradient clipping (max_norm=1.0)
      3. Check for NaN/Inf in dataset
      4. Use bf16 instead of fp16
    fallback: Start from checkpoint before NaN

  - error: "Model not improving"
    cause: Learning rate too low or data quality
    solution: |
      1. Increase learning rate
      2. Check dataset quality
      3. Verify data format matches model
      4. Increase LoRA rank
    fallback: Try different base model

  - error: "Catastrophic forgetting"
    cause: Over-training on narrow dataset
    solution: |
      1. Reduce epochs
      2. Lower learning rate
      3. Add regularization
      4. Mix in general data
    fallback: Use smaller LoRA rank
```

## Fallback Strategies

```yaml
training_fallback:
  memory_pressure:
    - action: reduce_batch_size
      factor: 0.5
      min: 1

    - action: enable_gradient_checkpointing
      saves: ~40% VRAM

    - action: switch_to_qlora
      saves: ~50% VRAM

    - action: use_deepspeed_offload
      saves: ~60% VRAM

  training_instability:
    - action: reduce_learning_rate
      factor: 0.1

    - action: add_warmup_steps
      ratio: 0.03

    - action: enable_gradient_clipping
      max_norm: 1.0

    - action: switch_to_bf16
      reason: More stable than fp16
```

## Token & Cost Optimization

```yaml
optimization:
  hardware_costs:
    a100_80gb:
      hourly: $4.00
      typical_training: 2-8 hours
    a10g_24gb:
      hourly: $1.50
      typical_training: 4-16 hours
    rtx_4090:
      hourly: $0.50 (owned)
      qlora_capable: true

  training_efficiency:
    qlora_vs_lora:
      memory_savings: 50%
      speed_impact: -10%
      quality_impact: -2-5%

    unsloth_optimization:
      memory_savings: 40%
      speed_improvement: 2x
      compatible_models: [llama, mistral, gemma]

  dataset_optimization:
    deduplication: true
    quality_filter: true
    optimal_size: 10K-100K examples
    diminishing_returns_after: 500K examples
```

## Observability

```yaml
training_metrics:
  loss:
    - train_loss
    - eval_loss
    - gradient_norm

  learning:
    - learning_rate
    - epoch
    - step

  resources:
    - gpu_memory_allocated
    - gpu_utilization
    - throughput_samples_per_second

  quality:
    - eval_accuracy
    - eval_perplexity
    - benchmark_scores

logging:
  wandb:
    enabled: true
    project: fine-tuning
    log_model: checkpoint

  tensorboard:
    enabled: true
    log_dir: ./runs

  callbacks:
    - early_stopping
    - model_checkpoint
    - lr_scheduler
```

## Troubleshooting Guide

### Debug Checklist

```markdown
1. [ ] Verify dataset format
   ```python
   from datasets import load_dataset
   ds = load_dataset("json", data_files="train.json")
   print(ds["train"][0])  # Check structure
   ```

2. [ ] Validate tokenization
   ```python
   sample = tokenizer(ds["train"][0]["text"])
   print(f"Tokens: {len(sample['input_ids'])}")
   decoded = tokenizer.decode(sample['input_ids'])
   ```

3. [ ] Check model loading
   ```python
   print(model)
   print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
   ```

4. [ ] Monitor first batch
   ```python
   for batch in dataloader:
       outputs = model(**batch)
       print(f"Loss: {outputs.loss.item()}")
       break
   ```

5. [ ] Verify checkpointing
   ```bash
   ls -la output_dir/
   # Should see checkpoint-* directories
   ```
```

### Common Failure Modes

| Symptom | Root Cause | Fix |
|---------|------------|-----|
| OOM at start | Model too large | Use QLoRA + 4-bit |
| OOM mid-training | Gradient accumulation | Clear cache, reduce batch |
| Loss spikes | Bad data sample | Filter outliers |
| No improvement | LR too low | Increase 10x |
| Overfitting | Too many epochs | Early stopping |

### LoRA Hyperparameter Guide

```yaml
rank (r):
  4-8: Simple tasks, single domain
  16-32: General instruction tuning
  64-128: Complex adaptation, multiple tasks
  recommendation: Start with 16, adjust based on results

alpha:
  formula: alpha = 2 * rank
  effect: Higher = more LoRA influence
  typical: 32 for r=16

target_modules:
  minimal: [q_proj, v_proj]
  recommended: [q_proj, k_proj, v_proj, o_proj]
  aggressive: + [gate_proj, up_proj, down_proj]

dropout:
  none: 0.0 (most common)
  light: 0.05
  aggressive: 0.1 (prevents overfitting)
```

## LoRA Configuration Example

```python
from peft import LoraConfig

config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Alpha scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

## Example Prompts

- "Set up LoRA fine-tuning for Llama 3.1"
- "Prepare an instruction dataset for fine-tuning"
- "Implement QLoRA for memory-efficient training"
- "Evaluate my fine-tuned model against benchmarks"
- "Merge LoRA adapters into base model"
- "Debug why my training loss is not decreasing"

## Dependencies

```yaml
skills:
  - fine-tuning (PRIMARY)

agents:
  - 01-llm-fundamentals (base model knowledge)
  - 05-evaluation-monitoring (model evaluation)

external:
  - transformers >= 4.36.0
  - peft >= 0.7.0
  - trl >= 0.7.0
  - bitsandbytes >= 0.41.0
  - datasets >= 2.14.0
```
