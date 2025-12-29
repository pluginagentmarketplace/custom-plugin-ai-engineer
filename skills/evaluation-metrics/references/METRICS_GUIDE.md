# LLM Evaluation Metrics Guide

Understanding and selecting the right metrics for LLM evaluation.

## Metric Categories

### 1. Text Similarity Metrics

| Metric | Type | Best For | Range |
|--------|------|----------|-------|
| BLEU | N-gram overlap | Translation | 0-1 |
| ROUGE | N-gram recall | Summarization | 0-1 |
| METEOR | Alignment | Translation | 0-1 |
| BERTScore | Semantic similarity | General | 0-1 |
| BLEURT | Learned metric | Quality | 0-1 |

### 2. RAG Metrics

| Metric | Measures | Threshold |
|--------|----------|-----------|
| Faithfulness | Grounded in context | >0.8 |
| Answer Relevancy | Answers the question | >0.7 |
| Context Precision | Context is relevant | >0.7 |
| Context Recall | Important info retrieved | >0.7 |

### 3. Task-Specific Metrics

| Task | Primary Metrics |
|------|-----------------|
| Classification | Accuracy, F1, Precision, Recall |
| Code Generation | Pass@k, Functional Correctness |
| Translation | BLEU, chrF, COMET |
| Summarization | ROUGE, BERTScore |
| QA | Exact Match, F1, Has Answer |

## Detailed Metric Explanations

### BLEU (Bilingual Evaluation Understudy)

```python
from evaluate import load

bleu = load("bleu")
results = bleu.compute(
    predictions=["the cat sat on the mat"],
    references=[["the cat is on the mat"]]
)
# Score: 0.61
```

**Pros:**
- Fast and deterministic
- Standard for translation

**Cons:**
- Ignores semantics
- Penalizes valid paraphrases

**When to use:** Translation, constrained generation

### ROUGE (Recall-Oriented Understudy)

```python
from evaluate import load

rouge = load("rouge")
results = rouge.compute(
    predictions=["the cat sat on mat"],
    references=["the cat is on the mat"]
)
# ROUGE-1: 0.80, ROUGE-L: 0.80
```

**Variants:**
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence

**When to use:** Summarization, recall matters

### BERTScore

```python
from evaluate import load

bertscore = load("bertscore")
results = bertscore.compute(
    predictions=["the kitty rested on the rug"],
    references=["the cat sat on the mat"],
    model_type="microsoft/deberta-xlarge-mnli"
)
# F1: 0.89 (captures semantic similarity)
```

**When to use:** Semantic evaluation, paraphrasing

### Perplexity

```python
from evaluate import load

perplexity = load("perplexity", module_type="metric")
results = perplexity.compute(
    predictions=["the cat sat on the mat"],
    model_id="gpt2"
)
# Lower is better (measures fluency)
```

**When to use:** Fluency evaluation, language model comparison

## RAG Evaluation Framework

### RAGAS Metrics

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,       # Is answer grounded?
        answer_relevancy,   # Is answer relevant?
        context_precision,  # Is context precise?
        context_recall      # Is context complete?
    ]
)
```

### Faithfulness Example

```
Context: "Paris is the capital of France. It has a population of 2.1 million."
Question: "What is the capital of France?"
Answer: "Paris is the capital of France and has 10 million people."

Faithfulness Score: 0.5
Reason: Population claim not supported by context
```

## Hallucination Detection

### Methods

1. **Self-Consistency**: Generate multiple times, check agreement
2. **Factual Verification**: Check against knowledge base
3. **Source Attribution**: Trace claims to sources
4. **Confidence Scoring**: Low confidence = likely hallucination

### Implementation

```python
def detect_hallucination(claim, context):
    # Method 1: Check context support
    context_support = compute_similarity(claim, context)

    # Method 2: Self-consistency
    regenerations = [llm.generate(prompt) for _ in range(5)]
    consistency = compute_agreement(regenerations)

    # Method 3: Factual verification
    facts = extract_facts(claim)
    verified = verify_facts(facts)

    return {
        "context_support": context_support,
        "consistency": consistency,
        "factual_accuracy": verified,
        "hallucination_risk": 1 - min(context_support, consistency, verified)
    }
```

## Human Evaluation

### When to Use

- Final quality assessment
- Subjective qualities (creativity, helpfulness)
- Validating automatic metrics
- Edge cases and failures

### Guidelines

```yaml
Evaluation Criteria:
  Helpfulness:
    1: Not helpful at all
    2: Slightly helpful
    3: Moderately helpful
    4: Very helpful
    5: Extremely helpful

  Accuracy:
    1: Completely wrong
    2: Mostly wrong
    3: Partially correct
    4: Mostly correct
    5: Fully accurate

  Safety:
    1: Harmful content
    2: Potentially problematic
    3: Neutral
    4: Appropriate
    5: Exemplary safety
```

### Best Practices

1. **Multiple annotators** (min 3)
2. **Clear guidelines** with examples
3. **Inter-annotator agreement** (Cohen's Kappa > 0.6)
4. **Calibration exercises** before main evaluation
5. **Blind evaluation** (don't reveal model identity)

## Metric Selection Guide

### By Use Case

```
Translation → BLEU, COMET, chrF
Summarization → ROUGE, BERTScore
QA → Exact Match, F1
Code → Pass@k, Functional Correctness
Chat → Human eval, Safety metrics
RAG → Faithfulness, Relevancy
```

### Decision Tree

```
Is there a reference answer?
├── Yes
│   ├── Exact match needed? → Exact Match, Accuracy
│   ├── Semantic similarity? → BERTScore, BLEURT
│   └── Structure matters? → BLEU, ROUGE
│
└── No
    ├── Fluency check? → Perplexity
    ├── Factual accuracy? → Hallucination detection
    └── User satisfaction? → Human evaluation
```

## Common Pitfalls

1. **Single metric reliance**: Use multiple metrics
2. **Ignoring distribution**: Look at variance, not just mean
3. **Metric gaming**: Models can overfit to metrics
4. **Static evaluation**: Real usage differs from benchmarks
5. **Ignoring failures**: Analyze worst cases
