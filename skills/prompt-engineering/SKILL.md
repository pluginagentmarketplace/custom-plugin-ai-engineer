---
name: prompt-engineering
description: Prompt design, optimization, few-shot learning, and chain of thought techniques for LLM applications.
sasmp_version: "1.3.0"
bonded_agent: 02-prompt-engineering
bond_type: PRIMARY_BOND
---

# Prompt Engineering

Master the art of crafting effective prompts for LLMs.

## Quick Start

### Basic Prompt Structure
```python
# Simple completion prompt
prompt = """
You are a helpful assistant specialized in {domain}.

Task: {task_description}

Context: {relevant_context}

Instructions:
1. {instruction_1}
2. {instruction_2}

Output format: {desired_format}
"""

# Using OpenAI
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)
```

### Few-Shot Learning
```python
few_shot_prompt = """
Classify the sentiment of the following reviews.

Examples:
Review: "This product exceeded my expectations!"
Sentiment: Positive

Review: "Terrible quality, broke after one day."
Sentiment: Negative

Review: "It's okay, nothing special."
Sentiment: Neutral

Now classify:
Review: "{user_review}"
Sentiment:
"""
```

## Core Techniques

### Chain of Thought (CoT)
```python
cot_prompt = """
Solve this step by step:

Problem: {problem}

Let's think through this carefully:
1. First, identify what we know...
2. Then, determine what we need to find...
3. Apply the relevant formula/logic...
4. Calculate the result...

Final Answer:
"""
```

### Self-Consistency
```python
# Generate multiple reasoning paths
responses = []
for _ in range(5):
    response = generate_with_cot(prompt, temperature=0.7)
    responses.append(response)

# Take majority vote
final_answer = majority_vote(responses)
```

### ReAct Pattern
```python
react_prompt = """
Answer the following question using the ReAct framework.

Question: {question}

Use this format:
Thought: [Your reasoning about what to do next]
Action: [The action to take: Search, Calculate, or Lookup]
Observation: [The result of the action]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer.
Final Answer: [Your answer]
"""
```

## Prompt Templates

### System Prompts by Role
```yaml
Expert Advisor:
  "You are an expert {role} with 20+ years of experience.
   Provide detailed, accurate, and actionable advice.
   Always cite sources when making claims.
   Ask clarifying questions if the request is ambiguous."

Code Assistant:
  "You are a senior software engineer specializing in {language}.
   Write clean, efficient, and well-documented code.
   Follow {style_guide} conventions.
   Include error handling and edge cases."

Data Analyst:
  "You are a data analyst helping interpret {data_type}.
   Explain insights in simple terms for non-technical audiences.
   Highlight key findings and recommendations.
   Note any limitations or caveats in the analysis."
```

### Output Formatting
```python
json_format_prompt = """
Extract the following information and return as JSON:

Text: {text}

Required fields:
- name (string)
- date (ISO 8601 format)
- amount (number)
- category (one of: income, expense, transfer)

Return ONLY valid JSON, no explanation.
"""

structured_output = """
Analyze the text and respond in this exact format:

## Summary
[2-3 sentence summary]

## Key Points
- Point 1
- Point 2
- Point 3

## Recommendations
1. [First recommendation]
2. [Second recommendation]
"""
```

## Advanced Techniques

### Prompt Chaining
```python
def analyze_document(document):
    # Step 1: Extract key information
    extraction_prompt = f"Extract key entities from: {document}"
    entities = llm(extraction_prompt)

    # Step 2: Analyze relationships
    analysis_prompt = f"Analyze relationships between: {entities}"
    relationships = llm(analysis_prompt)

    # Step 3: Generate summary
    summary_prompt = f"Summarize findings: {relationships}"
    summary = llm(summary_prompt)

    return summary
```

### Dynamic Prompt Generation
```python
def build_prompt(task_type, context, examples=None):
    base_prompt = PROMPT_TEMPLATES[task_type]

    if examples:
        example_str = format_examples(examples)
        base_prompt = f"{example_str}\n\n{base_prompt}"

    if context:
        base_prompt = base_prompt.replace("{context}", context)

    return base_prompt
```

## Best Practices

1. **Be Specific**: Clear instructions yield better results
2. **Provide Examples**: Few-shot learning improves consistency
3. **Set Constraints**: Define output format, length, style
4. **Use Delimiters**: Separate sections with clear markers
5. **Iterate**: Test and refine prompts systematically
6. **Version Control**: Track prompt changes like code

## Common Pitfalls

| Issue | Cause | Solution |
|-------|-------|----------|
| Inconsistent output | Vague instructions | Add explicit format requirements |
| Hallucinations | No grounding | Provide reference context |
| Verbose responses | No length constraint | Specify max length/format |
| Wrong tone | Missing persona | Add role/style instructions |
| Off-topic answers | Unclear scope | Define boundaries explicitly |

## Retry Logic for Format Failures

```python
def prompt_with_validation(prompt, validator, max_retries=3):
    for _ in range(max_retries):
        response = llm.generate(prompt)
        if validator(response):
            return response
        prompt = f"Previous response invalid. {prompt}"
    raise ValueError("Max retries exceeded")
```

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| Wrong format | Weak instructions | Add explicit examples |
| Too verbose | No length limit | Add word/sentence limits |
| Refuses task | Safety trigger | Rephrase request |

## Unit Test Template

```python
def test_prompt_generates_json():
    response = llm.generate(json_prompt)
    data = json.loads(response)
    assert "field" in data
```
