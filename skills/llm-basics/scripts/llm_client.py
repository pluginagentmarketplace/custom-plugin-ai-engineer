#!/usr/bin/env python3
"""
Unified LLM Client - Multi-provider support with consistent interface.

Supports: OpenAI, Anthropic, HuggingFace, Ollama (local)

Usage:
    from llm_client import LLMClient

    client = LLMClient(provider="openai", model="gpt-4")
    response = client.generate("Hello, world!")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Generator
import os
import json
import time


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None
    stream: bool = False


@dataclass
class Message:
    """Chat message structure."""
    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    """Standardized response from LLM."""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    latency_ms: float


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> LLMResponse:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def chat(self, messages: List[Message], config: GenerationConfig) -> LLMResponse:
        """Chat completion with message history."""
        pass

    @abstractmethod
    def stream(self, prompt: str, config: GenerationConfig) -> Generator[str, None, None]:
        """Stream generation token by token."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str, config: GenerationConfig) -> LLMResponse:
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop_sequences
        )

        latency = (time.time() - start_time) * 1000

        return LLMResponse(
            text=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            finish_reason=response.choices[0].finish_reason,
            latency_ms=latency
        )

    def chat(self, messages: List[Message], config: GenerationConfig) -> LLMResponse:
        start_time = time.time()

        formatted_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop_sequences
        )

        latency = (time.time() - start_time) * 1000

        return LLMResponse(
            text=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            finish_reason=response.choices[0].finish_reason,
            latency_ms=latency
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model: str = "claude-3-opus-20240229", api_key: Optional[str] = None):
        from anthropic import Anthropic
        self.model = model
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, config: GenerationConfig) -> LLMResponse:
        start_time = time.time()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        latency = (time.time() - start_time) * 1000

        return LLMResponse(
            text=response.content[0].text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            finish_reason=response.stop_reason,
            latency_ms=latency
        )

    def chat(self, messages: List[Message], config: GenerationConfig) -> LLMResponse:
        # Extract system message if present
        system = None
        chat_messages = []

        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        start_time = time.time()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=config.max_tokens,
            system=system,
            messages=chat_messages
        )

        latency = (time.time() - start_time) * 1000

        return LLMResponse(
            text=response.content[0].text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            finish_reason=response.stop_reason,
            latency_ms=latency
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Generator[str, None, None]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text


class OllamaProvider(BaseLLMProvider):
    """Ollama local inference provider."""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        import requests
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()

    def generate(self, prompt: str, config: GenerationConfig) -> LLMResponse:
        start_time = time.time()

        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                    "top_p": config.top_p
                },
                "stream": False
            }
        )
        response.raise_for_status()
        data = response.json()

        latency = (time.time() - start_time) * 1000

        return LLMResponse(
            text=data["response"],
            model=self.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            },
            finish_reason="stop",
            latency_ms=latency
        )

    def chat(self, messages: List[Message], config: GenerationConfig) -> LLMResponse:
        formatted_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        start_time = time.time()

        response = self.session.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": formatted_messages,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens
                },
                "stream": False
            }
        )
        response.raise_for_status()
        data = response.json()

        latency = (time.time() - start_time) * 1000

        return LLMResponse(
            text=data["message"]["content"],
            model=self.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            },
            finish_reason="stop",
            latency_ms=latency
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Generator[str, None, None]:
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]


class LLMClient:
    """Unified LLM client with multi-provider support."""

    PROVIDERS = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        **kwargs
    ):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        ProviderClass = self.PROVIDERS[provider]

        if model:
            self.provider = ProviderClass(model=model, **kwargs)
        else:
            self.provider = ProviderClass(**kwargs)

        self.default_config = GenerationConfig()

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt."""
        config = GenerationConfig(
            temperature=temperature or self.default_config.temperature,
            max_tokens=max_tokens or self.default_config.max_tokens,
            **kwargs
        )
        return self.provider.generate(prompt, config)

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Chat completion with message history."""
        parsed_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]
        config = GenerationConfig(**kwargs)
        return self.provider.chat(parsed_messages, config)

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream generation token by token."""
        config = GenerationConfig(**kwargs)
        return self.provider.stream(prompt, config)


# Example usage
if __name__ == "__main__":
    # Test with OpenAI
    client = LLMClient(provider="openai", model="gpt-4")
    response = client.generate("Explain transformers in one sentence.")
    print(f"Response: {response.text}")
    print(f"Tokens: {response.usage}")
    print(f"Latency: {response.latency_ms:.0f}ms")
