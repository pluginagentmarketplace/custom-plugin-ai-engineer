# Prompting Techniques Reference

A comprehensive guide to advanced prompting techniques for LLMs.

## Core Techniques

### 1. Zero-Shot Prompting

Direct instruction without examples.

```
Classify this email as spam or not spam:
"Congratulations! You've won $1,000,000!"

Classification:
```

**Best for:** Simple, well-defined tasks
**Limitation:** May not follow specific format requirements

### 2. Few-Shot Prompting

Provide examples to guide the response format.

```
Classify these emails:

Email: "Your order has shipped"
Classification: Not Spam

Email: "You've been selected for a prize"
Classification: Spam

Email: "Meeting at 3pm tomorrow"
Classification:
```

**Best for:** Format-sensitive outputs, novel tasks
**Tip:** 3-5 examples typically optimal

### 3. Chain of Thought (CoT)

Encourage step-by-step reasoning.

```
Solve this step by step:

Problem: If a train travels at 60 mph for 2.5 hours,
then 40 mph for 1 hour, how far did it travel total?

Let's think through this:
1. Distance at 60 mph = 60 × 2.5 = 150 miles
2. Distance at 40 mph = 40 × 1 = 40 miles
3. Total = 150 + 40 = 190 miles

Answer: 190 miles
```

**Best for:** Math, logic, multi-step reasoning
**Variant:** "Let's think step by step" as trigger phrase

### 4. Self-Consistency

Generate multiple reasoning paths, take majority vote.

```python
def self_consistent_answer(prompt, n=5):
    answers = []
    for _ in range(n):
        response = llm.generate(prompt, temperature=0.7)
        answer = extract_answer(response)
        answers.append(answer)
    return most_common(answers)
```

**Best for:** Reducing hallucinations in reasoning tasks
**Trade-off:** Higher latency and cost

### 5. Tree of Thoughts (ToT)

Explore multiple reasoning branches.

```
Problem: [Complex puzzle]

Let me consider multiple approaches:

Branch A: Start with constraint X
- Step A1: ...
- Step A2: ...
- Evaluation: This leads to contradiction

Branch B: Start with constraint Y
- Step B1: ...
- Step B2: ...
- Evaluation: This is promising, continue

Branch B continued:
- Step B3: ...
- Final answer: ...
```

**Best for:** Complex problem solving, puzzles
**Implementation:** Requires orchestration layer

## Advanced Patterns

### ReAct (Reason + Act)

Interleave reasoning with tool use.

```
Question: What is the population of the capital of France?

Thought: I need to find the capital of France first.
Action: Search[capital of France]
Observation: Paris is the capital of France.

Thought: Now I need the population of Paris.
Action: Search[population of Paris]
Observation: Paris has a population of about 2.1 million.

Thought: I have the answer.
Final Answer: The population of Paris, the capital of France,
is approximately 2.1 million.
```

### Reflexion

Self-critique and improve.

```
Task: Write a function to find prime numbers.

Initial attempt:
[Code attempt 1]

Self-critique:
- The loop is inefficient
- Edge cases not handled
- Missing documentation

Improved version:
[Code attempt 2 with improvements]

Final check:
- Efficiency: ✓ Improved
- Edge cases: ✓ Handled
- Documentation: ✓ Added
```

### Decomposition

Break complex tasks into subtasks.

```
Main task: Write a research paper on AI ethics.

Subtasks:
1. Research current AI ethics frameworks
2. Identify key ethical challenges
3. Analyze case studies
4. Propose recommendations
5. Write introduction and conclusion
6. Review and revise

Let me start with subtask 1...
```

## Prompt Components

### 1. Role/Persona

```
You are a senior data scientist with 15 years of experience
specializing in machine learning. You explain complex concepts
clearly and always consider practical implications.
```

### 2. Context

```
You are helping a team that is migrating from Python 2 to
Python 3. They have a large codebase with many deprecated
patterns.
```

### 3. Task

```
Analyze the following code and identify Python 2 patterns
that need to be updated for Python 3 compatibility.
```

### 4. Format

```
Respond in the following format:
## Issues Found
- Issue 1: [description]
- Issue 2: [description]

## Recommended Changes
1. [Change 1]
2. [Change 2]
```

### 5. Constraints

```
- Keep explanations under 100 words
- Use only standard library functions
- Maintain backward compatibility
```

### 6. Examples

```
Example input: "print 'hello'"
Example output: Issue: print statement → Use print() function
```

## Common Patterns

### The RISEN Framework

```
Role: You are a [role]
Instructions: [What to do]
Situation: [Context/Background]
Execution: [Specific steps]
Needle: [Key constraint or focus]
```

### The CRISPE Framework

```
Capacity: Act as [expert role]
Request: I want you to [action]
Insight: Provide [specific knowledge]
Statement: [Main task description]
Personality: [Tone and style]
Experiment: [Output format/examples]
```

## Output Control

### JSON Output

```
Extract the following information and return as valid JSON:

Text: "John Smith, age 32, works at Google"

Return format:
{
  "name": string,
  "age": number,
  "company": string
}

Response:
```

### Structured Markdown

```
Analyze this code and respond in this exact format:

## Summary
[1-2 sentences]

## Issues
| Issue | Severity | Location |
|-------|----------|----------|
| ...   | ...      | ...      |

## Recommendations
1. [First recommendation]
2. [Second recommendation]
```

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Better Approach |
|--------------|---------|-----------------|
| Vague instructions | Inconsistent outputs | Be specific |
| No examples | Format confusion | Add 1-3 examples |
| Too many constraints | Model confusion | Prioritize key constraints |
| Assuming context | Hallucinations | Provide necessary context |
| Long, run-on prompts | Instruction loss | Use clear sections |

## Quick Reference Card

| Technique | When to Use | Complexity |
|-----------|-------------|------------|
| Zero-shot | Simple, clear tasks | Low |
| Few-shot | Format-sensitive | Low |
| Chain of Thought | Reasoning tasks | Medium |
| Self-Consistency | High-stakes reasoning | Medium |
| ReAct | Tool-augmented tasks | High |
| Tree of Thoughts | Complex problem solving | High |
| Reflexion | Quality-critical outputs | High |

## Best Practices Checklist

- [ ] Clear role definition
- [ ] Specific task description
- [ ] Relevant context provided
- [ ] Output format specified
- [ ] Examples included (when helpful)
- [ ] Constraints explicitly stated
- [ ] Edge cases addressed
- [ ] Tested with diverse inputs
