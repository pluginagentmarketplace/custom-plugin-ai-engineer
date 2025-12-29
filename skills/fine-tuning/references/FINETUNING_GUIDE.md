# Fine-Tuning Guide

Complete guide to adapting LLMs for specific tasks and domains.

## When to Fine-Tune

### Good Candidates for Fine-Tuning

✅ **Do Fine-Tune When:**
- Need consistent output format/style
- Have 100+ quality training examples
- Domain-specific terminology
- Custom behavior patterns
- Cost reduction for high-volume use

❌ **Don't Fine-Tune When:**
- Few examples (use few-shot prompting)
- General knowledge tasks
- Can achieve with prompt engineering
- Rapidly changing requirements
- Limited compute resources

## Method Comparison

| Method | VRAM | Speed | Quality | Cost |
|--------|------|-------|---------|------|
| Full Fine-Tune | 60GB+ | Slow | Best | High |
| LoRA | 16GB | Fast | Very Good | Low |
| QLoRA (4-bit) | 6-8GB | Medium | Good | Very Low |
| Prefix Tuning | 8GB | Fast | Good | Low |
| Prompt Tuning | 4GB | Very Fast | Moderate | Very Low |

## LoRA Deep Dive

### What is LoRA?

LoRA (Low-Rank Adaptation) adds small trainable matrices to frozen model weights:

```
Original: W (frozen)
LoRA: W + ΔW = W + BA

Where:
- B: d × r matrix (down-projection)
- A: r × k matrix (up-projection)
- r << min(d, k) (low rank)
```

### Key Hyperparameters

#### Rank (r)
```
r = 4-8    → Simple tasks, less capacity
r = 16-32  → General fine-tuning
r = 64-128 → Complex domain adaptation

Higher rank = More capacity but more parameters
```

#### Alpha (α)
```
α = 2 × r  → Standard scaling
α = r      → More conservative
α = 4 × r  → More aggressive

Scaling factor: ΔW = (α/r) × BA
```

#### Target Modules

```yaml
# Attention only (most common)
target_modules:
  - q_proj
  - v_proj

# Full attention
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

# Attention + MLP (maximum adaptation)
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

## Dataset Preparation

### Instruction Format (Alpaca)

```json
{
  "instruction": "Summarize the following text.",
  "input": "The quick brown fox jumps over the lazy dog...",
  "output": "A fox jumps over a dog."
}
```

### Chat Format (ShareGPT)

```json
{
  "conversations": [
    {"from": "human", "value": "What is AI?"},
    {"from": "gpt", "value": "AI is..."},
    {"from": "human", "value": "Can you explain more?"},
    {"from": "gpt", "value": "Certainly..."}
  ]
}
```

### Best Practices for Data

1. **Quality > Quantity**: 500 excellent examples beat 5000 poor ones
2. **Diversity**: Cover edge cases and variations
3. **Consistency**: Same format throughout
4. **Length variation**: Mix short and long responses
5. **Negative examples**: Include what NOT to do (if applicable)

### Data Cleaning Checklist

- [ ] Remove duplicates
- [ ] Fix encoding issues
- [ ] Validate JSON format
- [ ] Check for PII/sensitive data
- [ ] Verify instruction-output alignment
- [ ] Balance categories if classification

## Training Process

### Step 1: Environment Setup

```bash
# Create environment
conda create -n finetune python=3.10
conda activate finetune

# Install dependencies
pip install torch transformers peft trl datasets
pip install bitsandbytes accelerate
```

### Step 2: Load and Prepare Model

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for training
model = prepare_model_for_kbit_training(model)
```

### Step 3: Apply LoRA

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### Step 4: Train

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_steps=100
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=512
)

trainer.train()
```

## Troubleshooting

### Loss Not Decreasing

| Symptom | Cause | Solution |
|---------|-------|----------|
| Loss flat | LR too low | Increase 10x |
| Loss spikes | LR too high | Decrease 5x |
| Loss oscillates | Batch too small | Increase batch/accumulation |

### Out of Memory

```python
# Solutions in order:
1. Enable gradient checkpointing
   model.gradient_checkpointing_enable()

2. Reduce batch size
   per_device_train_batch_size=2

3. Increase gradient accumulation
   gradient_accumulation_steps=8

4. Use 4-bit quantization
   load_in_4bit=True

5. Reduce max_seq_length
   max_seq_length=256
```

### Overfitting

```python
# Signs: Train loss ↓, Eval loss ↑

# Solutions:
1. Reduce epochs (3 → 1-2)
2. Increase dropout (0.05 → 0.1)
3. Add more training data
4. Lower learning rate
5. Use early stopping
```

## Evaluation

### Automatic Metrics

```python
from evaluate import load

# Perplexity
perplexity = load("perplexity")
results = perplexity.compute(predictions=outputs, model_id=model_name)

# BLEU (for translation/generation)
bleu = load("bleu")
results = bleu.compute(predictions=outputs, references=references)
```

### Manual Evaluation Checklist

- [ ] Follows instructions correctly
- [ ] Uses appropriate tone/style
- [ ] Factually accurate (if applicable)
- [ ] Appropriate length
- [ ] No hallucinations
- [ ] Handles edge cases

## Post-Training

### Merge Adapter Weights

```python
# Merge LoRA into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

### Multiple Adapters

```python
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("base_model")

# Load adapter
model = PeftModel.from_pretrained(model, "adapter_1")

# Add another adapter
model.load_adapter("adapter_2", adapter_name="coding")

# Switch at runtime
model.set_adapter("coding")
```

## Quick Reference

### Recommended Starting Configuration

```yaml
# 7B model on consumer GPU
model: meta-llama/Llama-2-7b-hf
lora_r: 16
lora_alpha: 32
batch_size: 4
gradient_accumulation: 4
learning_rate: 2e-4
epochs: 3
use_4bit: true
```

### VRAM Requirements

| Model | Full | LoRA | QLoRA |
|-------|------|------|-------|
| 7B | 28GB | 16GB | 6GB |
| 13B | 52GB | 28GB | 12GB |
| 70B | 280GB | 80GB | 48GB |
