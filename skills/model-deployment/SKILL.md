---
name: model-deployment
description: LLM deployment strategies including vLLM, TGI, and cloud inference endpoints.
sasmp_version: "1.3.0"
bonded_agent: 05-evaluation-monitoring
bond_type: SECONDARY_BOND
---

# Model Deployment

Deploy LLMs to production with optimal performance.

## Quick Start

### vLLM Server
```bash
# Install
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000 \
    --tensor-parallel-size 1

# Query (OpenAI-compatible)
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "Hello, how are you?",
        "max_tokens": 100
    }'
```

### Text Generation Inference (TGI)
```bash
# Docker deployment
docker run --gpus all -p 8080:80 \
    -v ./data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --quantize bitsandbytes-nf4 \
    --max-input-length 4096 \
    --max-total-tokens 8192

# Query
curl http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{"inputs": "What is AI?", "parameters": {"max_new_tokens": 100}}'
```

### Ollama (Local Deployment)
```bash
# Install and run
curl -fsSL https://ollama.ai/install.sh | sh
ollama run llama2

# API usage
curl http://localhost:11434/api/generate -d '{
    "model": "llama2",
    "prompt": "Why is the sky blue?"
}'
```

## Deployment Options Comparison

| Platform | Ease | Cost | Scale | Latency | Best For |
|----------|------|------|-------|---------|----------|
| vLLM | ⭐⭐ | Self-host | High | Low | Production |
| TGI | ⭐⭐ | Self-host | High | Low | HuggingFace ecosystem |
| Ollama | ⭐⭐⭐ | Free | Low | Medium | Local dev |
| OpenAI | ⭐⭐⭐ | Pay-per-token | Very High | Low | Quick start |
| AWS Bedrock | ⭐⭐ | Pay-per-token | Very High | Medium | Enterprise |
| Replicate | ⭐⭐⭐ | Pay-per-second | High | Medium | Prototyping |

## FastAPI Inference Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model at startup
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    text: str
    tokens_used: int

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_tokens = len(outputs[0]) - len(inputs.input_ids[0])

        return GenerateResponse(
            text=generated,
            tokens_used=new_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": model_name}
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download model (or mount volume)
RUN python3 -c "from transformers import AutoModelForCausalLM; \
    AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  llm-server:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
      - name: llm
        image: llm-inference:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "24Gi"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-inference
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Optimization Techniques

### Quantization
```python
from transformers import BitsAndBytesConfig

# 8-bit quantization
config_8bit = BitsAndBytesConfig(load_in_8bit=True)

# 4-bit quantization (QLoRA-style)
config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=config_4bit,
    device_map="auto"
)
```

### Batching
```python
# Dynamic batching with vLLM
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")

# vLLM automatically batches concurrent requests
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
sampling = SamplingParams(temperature=0.7, max_tokens=100)

outputs = llm.generate(prompts, sampling)  # Batched execution
```

### KV Cache Optimization
```python
# vLLM with PagedAttention
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.9,
    max_num_batched_tokens=4096
)
```

## Monitoring

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

REQUEST_COUNT = Counter('inference_requests_total', 'Total requests')
LATENCY = Histogram('inference_latency_seconds', 'Request latency')
TOKENS = Counter('tokens_generated_total', 'Total tokens generated')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    REQUEST_COUNT.inc()
    start = time.time()
    response = await call_next(request)
    LATENCY.observe(time.time() - start)
    return response

# Start metrics server
start_http_server(9090)
```

## Best Practices

1. **Use quantization**: 4-bit for dev, 8-bit for production
2. **Implement batching**: vLLM/TGI handle this automatically
3. **Monitor everything**: Latency, throughput, errors, GPU utilization
4. **Cache responses**: For repeated queries
5. **Set timeouts**: Prevent hung requests
6. **Load balance**: Multiple replicas for high availability

## Error Handling & Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def call_inference_api(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/generate",
            json={"prompt": prompt},
            timeout=30.0
        )
        return response.json()
```

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| OOM on load | Model too large | Use quantization |
| High latency | No batching | Enable vLLM batching |
| Connection refused | Server not started | Check health endpoint |

## Unit Test Template

```python
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```
