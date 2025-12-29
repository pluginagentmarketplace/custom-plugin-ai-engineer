# AI Agent Patterns Reference

Common patterns for building effective AI agents.

## Core Patterns

### 1. ReAct (Reason + Act)

The fundamental pattern for tool-using agents.

```
Thought: I need to find information about X
Action: search
Action Input: "X definition"
Observation: [results]
Thought: Now I can answer
Final Answer: X is...
```

**Implementation:**
```python
REACT_PROMPT = """
Thought: [reason about what to do]
Action: [tool name]
Action Input: [tool input]
Observation: [tool output]
... (repeat)
Thought: [conclude]
Final Answer: [response]
"""
```

**Use when:** General task solving with tools

### 2. Plan-and-Execute

Separate planning from execution.

```
Plan:
1. Search for background information
2. Analyze the data
3. Synthesize findings
4. Write summary

Execute Step 1: [search]
Execute Step 2: [analyze]
...
```

**Implementation:**
```python
class PlanAndExecute:
    def run(self, task):
        # Phase 1: Generate plan
        plan = self.planner.plan(task)

        # Phase 2: Execute each step
        results = []
        for step in plan:
            result = self.executor.execute(step)
            results.append(result)

        # Phase 3: Synthesize
        return self.synthesizer.combine(results)
```

**Use when:** Complex, multi-step tasks

### 3. Reflection

Self-critique and improve.

```
Initial Attempt: [first response]
Reflection:
- Strengths: ...
- Weaknesses: ...
- Improvements: ...
Revised Response: [improved response]
```

**Implementation:**
```python
def reflect(response, task):
    critique = llm.generate(f"""
    Task: {task}
    Response: {response}

    Critique this response:
    - What's good?
    - What's missing?
    - How to improve?
    """)

    improved = llm.generate(f"""
    Original: {response}
    Critique: {critique}

    Write an improved response:
    """)

    return improved
```

**Use when:** Quality matters more than speed

### 4. Self-Ask

Break down questions recursively.

```
Question: Is X bigger than Y?
Follow-up: What is the size of X?
Intermediate Answer: X is 100
Follow-up: What is the size of Y?
Intermediate Answer: Y is 80
So the final answer is: Yes, X is bigger
```

**Use when:** Complex reasoning with sub-questions

## Multi-Agent Patterns

### 5. Debate/Adversarial

Multiple agents argue different positions.

```
Agent A (Pro): [argument for]
Agent B (Con): [argument against]
Agent A: [rebuttal]
Agent B: [counter]
Judge: [synthesis/decision]
```

**Use when:** Need balanced perspectives

### 6. Expert Panel

Specialists collaborate on tasks.

```python
experts = {
    "researcher": Agent(role="gather information"),
    "analyst": Agent(role="analyze data"),
    "writer": Agent(role="create content"),
    "critic": Agent(role="review quality")
}

def run_panel(task):
    research = experts["researcher"].run(task)
    analysis = experts["analyst"].run(research)
    draft = experts["writer"].run(analysis)
    final = experts["critic"].run(draft)
    return final
```

**Use when:** Complex tasks need diverse skills

### 7. Hierarchical

Manager delegates to workers.

```
Manager: [break task into subtasks]
├── Worker 1: [subtask 1]
├── Worker 2: [subtask 2]
└── Worker 3: [subtask 3]
Manager: [combine results]
```

**Use when:** Large tasks with clear subtasks

## Tool Integration Patterns

### 8. Tool Selection

Choose the right tool for the task.

```python
def select_tool(query, tools):
    prompt = f"""
    Query: {query}
    Available tools: {tools}

    Which tool should be used? Respond with just the tool name.
    """
    return llm.generate(prompt)
```

### 9. Tool Chaining

Use multiple tools in sequence.

```python
def chain_tools(query):
    # Step 1: Search
    results = search_tool.run(query)

    # Step 2: Extract
    data = extract_tool.run(results)

    # Step 3: Summarize
    summary = summarize_tool.run(data)

    return summary
```

### 10. Parallel Tool Execution

Run independent tools simultaneously.

```python
import asyncio

async def parallel_tools(queries):
    tasks = [
        asyncio.create_task(tool.run(q))
        for q in queries
    ]
    return await asyncio.gather(*tasks)
```

## Memory Patterns

### 11. Conversation Memory

Remember recent interactions.

```python
class ConversationMemory:
    def __init__(self, k=10):
        self.messages = []
        self.k = k

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-self.k:]
```

### 12. Summary Memory

Summarize long conversations.

```python
class SummaryMemory:
    def add_interaction(self, interaction):
        self.buffer.append(interaction)
        if len(self.buffer) > threshold:
            summary = self.llm.summarize(self.buffer)
            self.summary = f"{self.summary}\n{summary}"
            self.buffer = []
```

### 13. Entity Memory

Track mentioned entities.

```python
class EntityMemory:
    def extract_entities(self, text):
        # Extract and store entity information
        entities = ner.extract(text)
        for entity in entities:
            self.store[entity.name] = entity.info
```

## Error Handling Patterns

### 14. Retry with Backoff

```python
def retry_with_backoff(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i == max_retries - 1:
                raise
            time.sleep(2 ** i)
```

### 15. Fallback Chain

```python
def fallback_chain(query, strategies):
    for strategy in strategies:
        try:
            result = strategy(query)
            if is_valid(result):
                return result
        except Exception:
            continue
    return default_response
```

## Safety Patterns

### 16. Human-in-the-Loop

```python
def execute_with_approval(action):
    if is_high_risk(action):
        approved = get_human_approval(action)
        if not approved:
            return "Action cancelled"
    return execute(action)
```

### 17. Guardrails

```python
def apply_guardrails(response):
    # Content filtering
    if contains_harmful(response):
        return sanitize(response)

    # Output validation
    if not matches_schema(response):
        return reformat(response)

    return response
```

## Pattern Selection Guide

| Scenario | Pattern |
|----------|---------|
| Simple tool use | ReAct |
| Complex tasks | Plan-and-Execute |
| Quality critical | Reflection |
| Multiple perspectives | Debate |
| Diverse skills needed | Expert Panel |
| Large tasks | Hierarchical |
| High-risk actions | Human-in-the-Loop |
| Error-prone operations | Retry + Fallback |
