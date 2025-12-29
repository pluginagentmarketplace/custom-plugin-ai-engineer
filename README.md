# AI Engineer Plugin

A comprehensive Claude Code plugin for AI/ML engineers building LLM-powered applications.

## Overview

This plugin provides specialized agents, skills, and commands for modern AI engineering tasks including LLM development, RAG systems, fine-tuning, prompt engineering, and production deployment.

## Quick Start

```bash
# Install the plugin
claude plugin install ai-engineer

# Use the AI Engineer command
/ai-engineer help me build a RAG system

# Or invoke specific agents
Task(subagent_type="01-llm-fundamentals", prompt="Explain transformers")
```

## Features

### 🧠 LLM Development
- Multi-provider support (OpenAI, Anthropic, HuggingFace, Ollama)
- Model selection guidance
- Tokenization and context management
- Local inference setup

### 📝 Prompt Engineering
- Prompt design patterns (ReAct, CoT, Few-shot)
- Template management
- A/B testing framework
- Automatic optimization

### 📚 RAG Systems
- Document processing pipelines
- Chunking strategies
- Vector database integration
- Hybrid search implementation

### 🔧 Fine-Tuning
- LoRA and QLoRA implementation
- Dataset preparation
- Training configuration
- Model merging and deployment

### 📊 Evaluation
- Quality metrics (BLEU, ROUGE, BERTScore)
- RAG-specific metrics (faithfulness, relevancy)
- Hallucination detection
- A/B testing

### 🤖 AI Agents
- Multi-framework support (LangChain, CrewAI)
- Tool integration patterns
- Multi-agent orchestration
- Production safety guardrails

## Agents

| Agent | Description |
|-------|-------------|
| `01-llm-fundamentals` | LLM architecture, APIs, and inference |
| `02-prompt-engineering` | Prompt design and optimization |
| `03-rag-systems` | RAG pipeline development |
| `04-fine-tuning` | Model adaptation with LoRA/QLoRA |
| `05-evaluation-monitoring` | Quality metrics and production monitoring |
| `06-ai-agents` | Autonomous agent development |

## Skills

| Skill | Description |
|-------|-------------|
| `llm-basics` | LLM fundamentals and model selection |
| `prompt-engineering` | Prompt templates and optimization |
| `rag-systems` | RAG pipeline implementation |
| `fine-tuning` | LoRA configuration and training |
| `vector-databases` | Vector store integration |
| `model-deployment` | Inference server deployment |
| `evaluation-metrics` | LLM evaluation framework |
| `agent-frameworks` | AI agent development |

## Commands

- `/ai-engineer` - Main AI engineering assistant

## Use Cases

### Building a RAG System
```
/ai-engineer I want to build a Q&A system over my documentation.
Help me:
1. Choose the right vector database
2. Design the chunking strategy
3. Implement hybrid search
4. Set up evaluation metrics
```

### Fine-Tuning a Model
```
/ai-engineer Set up LoRA fine-tuning for Llama 2 7B on my
instruction dataset. Use QLoRA for memory efficiency.
```

### Deploying to Production
```
/ai-engineer Deploy my fine-tuned model using vLLM with
OpenAI-compatible API. Include monitoring and rate limiting.
```

## Directory Structure

```
custom-plugin-ai-engineer/
├── .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── agents/
│   ├── 01-llm-fundamentals.md
│   ├── 02-prompt-engineering.md
│   ├── 03-rag-systems.md
│   ├── 04-fine-tuning.md
│   ├── 05-evaluation-monitoring.md
│   └── 06-ai-agents.md
├── skills/
│   ├── llm-basics/
│   ├── prompt-engineering/
│   ├── rag-systems/
│   ├── fine-tuning/
│   ├── vector-databases/
│   ├── model-deployment/
│   ├── evaluation-metrics/
│   └── agent-frameworks/
├── commands/
│   └── ai-engineer.md
├── hooks/
│   └── hooks.json
└── README.md
```

## Requirements

### Python Dependencies
```bash
pip install openai anthropic transformers peft trl
pip install langchain chromadb sentence-transformers
pip install evaluate ragas
```

### Optional
- CUDA-capable GPU for local inference
- Docker for containerized deployment
- Kubernetes for production orchestration

## Best Practices

1. **Start Simple**: Use APIs before local deployment
2. **Measure First**: Establish baseline metrics before optimizing
3. **Iterate Quickly**: Use few-shot prompting before fine-tuning
4. **Monitor Everything**: Track latency, cost, and quality
5. **Safety First**: Implement guardrails before production

## Resources

- [OpenAI Cookbook](https://cookbook.openai.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LangChain Documentation](https://python.langchain.com/)
- [RAGAS Evaluation](https://docs.ragas.io/)

## Version

- Plugin Version: 1.0.0
- SASMP Version: 1.3.0
- EQHM: Enabled

## License

MIT License

## Contributing

Contributions welcome! Please follow the plugin development guidelines.
