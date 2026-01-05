<div align="center">

<!-- Animated Typing Banner -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=2E9EF7&center=true&vCenter=true&multiline=true&repeat=true&width=600&height=100&lines=Ai+Engineer+Assistant;6+Agents+%7C+8+Skills;Claude+Code+Plugin" alt="Ai Engineer Assistant" />

<br/>

<!-- Badge Row 1: Status Badges -->
[![Version](https://img.shields.io/badge/Version-1.1.0-blue?style=for-the-badge)](https://github.com/pluginagentmarketplace/custom-plugin-ai-engineer/releases)
[![License](https://img.shields.io/badge/License-Custom-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen?style=for-the-badge)](#)
[![SASMP](https://img.shields.io/badge/SASMP-v1.3.0-blueviolet?style=for-the-badge)](#)

<!-- Badge Row 2: Content Badges -->
[![Agents](https://img.shields.io/badge/Agents-6-orange?style=flat-square&logo=robot)](#-agents)
[![Skills](https://img.shields.io/badge/Skills-8-purple?style=flat-square&logo=lightning)](#-skills)
[![Commands](https://img.shields.io/badge/Commands-3-green?style=flat-square&logo=terminal)](#-commands)

<br/>

<!-- Quick CTA Row -->
[📦 **Install Now**](#-quick-start) · [🤖 **Explore Agents**](#-agents) · [📖 **Documentation**](#-documentation) · [⭐ **Star this repo**](https://github.com/pluginagentmarketplace/custom-plugin-ai-engineer)

---

### What is this?

> **Ai Engineer Assistant** is a Claude Code plugin with **6 agents** and **8 skills** for ai engineer development.

</div>

---

## 📑 Table of Contents

<details>
<summary>Click to expand</summary>

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Agents](#-agents)
- [Skills](#-skills)
- [Commands](#-commands)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

</details>

---

## 🚀 Quick Start

### Prerequisites

- Claude Code CLI v2.0.27+
- Active Claude subscription

### Installation (Choose One)

<details open>
<summary><strong>Option 1: From Marketplace (Recommended)</strong></summary>

```bash
# Step 1️⃣ Add the marketplace
/plugin marketplace add pluginagentmarketplace/custom-plugin-ai-engineer

# Step 2️⃣ Install the plugin
/plugin install ai-engineer-plugin@pluginagentmarketplace-ai-engineer

# Step 3️⃣ Restart Claude Code
# Close and reopen your terminal/IDE
```

</details>

<details>
<summary><strong>Option 2: Local Installation</strong></summary>

```bash
# Clone the repository
git clone https://github.com/pluginagentmarketplace/custom-plugin-ai-engineer.git
cd custom-plugin-ai-engineer

# Load locally
/plugin load .

# Restart Claude Code
```

</details>

### ✅ Verify Installation

After restart, you should see these agents:

```
ai-engineer-plugin:04-fine-tuning
ai-engineer-plugin:03-rag-systems
ai-engineer-plugin:06-ai-agents
ai-engineer-plugin:02-prompt-engineering
ai-engineer-plugin:05-evaluation-monitoring
... and 1 more
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **6 Agents** | Specialized AI agents for ai engineer tasks |
| 🛠️ **8 Skills** | Reusable capabilities with Golden Format |
| ⌨️ **3 Commands** | Quick slash commands |
| 🔄 **SASMP v1.3.0** | Full protocol compliance |

---

## 🤖 Agents

### 6 Specialized Agents

| # | Agent | Purpose |
|---|-------|---------|
| 1 | **04-fine-tuning** | Master LLM fine-tuning techniques including LoRA, QLoRA, and |
| 2 | **03-rag-systems** | Build production RAG systems with vector databases, embeddin |
| 3 | **06-ai-agents** | Build autonomous AI agents with tool use, planning, and mult |
| 4 | **02-prompt-engineering** | Master prompt design, optimization techniques, and effective |
| 5 | **05-evaluation-monitoring** | Implement LLM evaluation frameworks, monitoring, and observa |
| 6 | **01-llm-fundamentals** | Master LLM architecture, tokenization, transformer models, a |

---

## 🛠️ Skills

### Available Skills

| Skill | Description | Invoke |
|-------|-------------|--------|
| `vector-databases` | Vector database selection, indexing strategies, and semantic | `Skill("ai-engineer-plugin:vector-databases")` |
| `model-deployment` | LLM deployment strategies including vLLM, TGI, and cloud inf | `Skill("ai-engineer-plugin:model-deployment")` |
| `rag-systems` | Retrieval Augmented Generation systems with vector search, d | `Skill("ai-engineer-plugin:rag-systems")` |
| `prompt-engineering` | Prompt design, optimization, few-shot learning, and chain of | `Skill("ai-engineer-plugin:prompt-engineering")` |
| `llm-basics` | LLM architecture, tokenization, transformers, and inference  | `Skill("ai-engineer-plugin:llm-basics")` |
| `fine-tuning` | LLM fine-tuning with LoRA, QLoRA, and instruction tuning for | `Skill("ai-engineer-plugin:fine-tuning")` |
| `agent-frameworks` | AI agent development with LangChain, CrewAI, AutoGen, and to | `Skill("ai-engineer-plugin:agent-frameworks")` |
| `evaluation-metrics` | LLM evaluation frameworks, benchmarks, and quality metrics f | `Skill("ai-engineer-plugin:evaluation-metrics")` |

---

## ⌨️ Commands

| Command | Description |
|---------|-------------|
| `/ai-engineer` | AI Engineer assistant for LLM development, RAG systems, and AI applications |
| `/prompt-lab` | Interactive prompt engineering lab for designing, testing, and optimizing prompts |
| `/rag-builder` | RAG system builder for creating production-ready retrieval augmented generation pipelines |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [LICENSE](LICENSE) | License information |

---

## 📁 Project Structure

<details>
<summary>Click to expand</summary>

```
custom-plugin-ai-engineer/
├── 📁 .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── 📁 agents/              # 6 agents
├── 📁 skills/              # 8 skills (Golden Format)
├── 📁 commands/            # 3 commands
├── 📁 hooks/
├── 📄 README.md
├── 📄 CHANGELOG.md
└── 📄 LICENSE
```

</details>

---

## 📅 Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.1.0 |
| **Last Updated** | 2025-12-30 |
| **Status** | Production Ready |
| **SASMP** | v1.3.0 |
| **Agents** | 6 |
| **Skills** | 8 |
| **Commands** | 3 |

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch
3. Follow the Golden Format for new skills
4. Submit a pull request

---

## ⚠️ Security

> **Important:** This repository contains third-party code and dependencies.
>
> - ✅ Always review code before using in production
> - ✅ Check dependencies for known vulnerabilities
> - ✅ Follow security best practices
> - ✅ Report security issues privately via [Issues](../../issues)

---

## 📝 License

Copyright © 2025 **Dr. Umit Kacar** & **Muhsin Elcicek**

Custom License - See [LICENSE](LICENSE) for details.

---

## 👥 Contributors

<table>
<tr>
<td align="center">
<strong>Dr. Umit Kacar</strong><br/>
Senior AI Researcher & Engineer
</td>
<td align="center">
<strong>Muhsin Elcicek</strong><br/>
Senior Software Architect
</td>
</tr>
</table>

---

<div align="center">

**Made with ❤️ for the Claude Code Community**

[![GitHub](https://img.shields.io/badge/GitHub-pluginagentmarketplace-black?style=for-the-badge&logo=github)](https://github.com/pluginagentmarketplace)

</div>
