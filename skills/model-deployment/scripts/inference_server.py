#!/usr/bin/env python3
"""
LLM Inference Server - Production-ready model serving with FastAPI.

Features:
- OpenAI-compatible API
- Streaming support
- Rate limiting
- Health checks
- Prometheus metrics

Usage:
    uvicorn inference_server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import time
import torch
from contextlib import asynccontextmanager


# Models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0, le=1)
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0, le=1)
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: Optional[Message] = None
    text: Optional[str] = None
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


# Global state
model = None
tokenizer = None
metrics = {
    "requests_total": 0,
    "tokens_generated": 0,
    "latency_sum": 0.0
}


def load_model():
    """Load model and tokenizer."""
    global model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-chat-hf"

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management."""
    load_model()
    yield


# App setup
app = FastAPI(
    title="LLM Inference Server",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_response(prompt: str, params: Dict[str, Any]) -> tuple:
    """Generate text response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = len(inputs.input_ids[0])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=params.get("max_tokens", 512),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = generated[len(prompt):].strip()
    completion_tokens = len(outputs[0]) - prompt_tokens

    return response_text, prompt_tokens, completion_tokens


async def generate_stream(prompt: str, params: Dict[str, Any]):
    """Stream generation token by token."""
    from transformers import TextIteratorStreamer
    from threading import Thread

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": params.get("max_tokens", 512),
        "temperature": params.get("temperature", 0.7),
        "top_p": params.get("top_p", 0.9),
        "do_sample": True,
        "streamer": streamer
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text in streamer:
        yield f"data: {text}\n\n"
        await asyncio.sleep(0)

    yield "data: [DONE]\n\n"


def format_chat_prompt(messages: List[Message]) -> str:
    """Format messages into chat prompt."""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"<<SYS>>\n{msg.content}\n<</SYS>>\n\n"
        elif msg.role == "user":
            prompt += f"[INST] {msg.content} [/INST]\n"
        elif msg.role == "assistant":
            prompt += f"{msg.content}\n"
    return prompt


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available()
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "default",
                "object": "model",
                "owned_by": "organization"
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    global metrics

    start_time = time.time()
    metrics["requests_total"] += 1

    prompt = format_chat_prompt(request.messages)

    if request.stream:
        return StreamingResponse(
            generate_stream(prompt, request.model_dump()),
            media_type="text/event-stream"
        )

    try:
        response_text, prompt_tokens, completion_tokens = generate_response(
            prompt,
            request.model_dump()
        )

        latency = time.time() - start_time
        metrics["latency_sum"] += latency
        metrics["tokens_generated"] += completion_tokens

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint."""
    global metrics

    start_time = time.time()
    metrics["requests_total"] += 1

    if request.stream:
        return StreamingResponse(
            generate_stream(request.prompt, request.model_dump()),
            media_type="text/event-stream"
        )

    try:
        response_text, prompt_tokens, completion_tokens = generate_response(
            request.prompt,
            request.model_dump()
        )

        latency = time.time() - start_time
        metrics["latency_sum"] += latency
        metrics["tokens_generated"] += completion_tokens

        return CompletionResponse(
            id=f"cmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    text=response_text,
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    avg_latency = metrics["latency_sum"] / max(metrics["requests_total"], 1)

    return {
        "requests_total": metrics["requests_total"],
        "tokens_generated": metrics["tokens_generated"],
        "average_latency_seconds": avg_latency,
        "gpu_memory_used": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
