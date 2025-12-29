# LLM Deployment Guide

Production deployment strategies for large language models.

## Deployment Options

### Quick Comparison

| Option | Ease | Scale | Cost | Latency |
|--------|------|-------|------|---------|
| Ollama | ⭐⭐⭐⭐⭐ | Low | Free | Medium |
| vLLM | ⭐⭐⭐ | High | Self-host | Low |
| TGI | ⭐⭐⭐ | High | Self-host | Low |
| OpenAI API | ⭐⭐⭐⭐⭐ | Very High | $$ | Low |
| AWS Bedrock | ⭐⭐⭐ | Very High | $$$ | Medium |
| Custom FastAPI | ⭐⭐ | Medium | Self-host | Variable |

## vLLM Deployment

### Docker Setup

```dockerfile
FROM vllm/vllm-openai:latest

ENV MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
ENV GPU_MEMORY_UTILIZATION=0.9

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "${MODEL_NAME}", \
     "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model meta-llama/Llama-2-7b-chat-hf
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Production Configuration

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 256 \
    --quantization awq
```

## Kubernetes Deployment

### Basic Deployment

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
      - name: vllm
        image: vllm/vllm-openai:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-7b-chat-hf"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
```

### Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: 10
```

## Optimization Techniques

### Quantization

```python
# 4-bit quantization with bitsandbytes
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# AWQ quantization (faster inference)
from vllm import LLM
llm = LLM(model="TheBloke/Llama-2-7B-AWQ", quantization="awq")
```

### VRAM Requirements

| Model | FP16 | 8-bit | 4-bit |
|-------|------|-------|-------|
| 7B | 14GB | 8GB | 4GB |
| 13B | 26GB | 14GB | 8GB |
| 70B | 140GB | 70GB | 40GB |

### Batching Strategies

```python
# Continuous batching (vLLM handles automatically)
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    max_num_batched_tokens=4096,  # Total tokens per batch
    max_num_seqs=256  # Max concurrent sequences
)

# Process multiple requests efficiently
prompts = ["Hello", "How are you", "What is AI"]
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

## Monitoring

### Key Metrics

```python
# Prometheus metrics to track
metrics = {
    "inference_requests_total": Counter,
    "inference_latency_seconds": Histogram,
    "tokens_generated_total": Counter,
    "batch_size": Histogram,
    "queue_length": Gauge,
    "gpu_memory_used": Gauge,
    "gpu_utilization": Gauge
}
```

### Grafana Dashboard Panels

1. **Request Rate**: `rate(inference_requests_total[5m])`
2. **Latency P99**: `histogram_quantile(0.99, inference_latency_seconds)`
3. **Tokens/Second**: `rate(tokens_generated_total[5m])`
4. **GPU Memory**: `gpu_memory_used / gpu_memory_total`
5. **Queue Depth**: `queue_length`

## Load Balancing

### NGINX Configuration

```nginx
upstream llm_servers {
    least_conn;
    server llm-1:8000 weight=1;
    server llm-2:8000 weight=1;
    keepalive 32;
}

server {
    listen 80;

    location /v1 {
        proxy_pass http://llm_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

## Cost Optimization

### Strategies

1. **Right-size instances**: Don't over-provision GPU
2. **Use spot instances**: 60-90% savings for batch jobs
3. **Implement caching**: Reduce redundant inference
4. **Quantize models**: Lower VRAM = smaller instances
5. **Autoscale**: Scale down during off-peak

### Cost Comparison (Monthly, ~1M requests)

| Option | Estimated Cost |
|--------|----------------|
| OpenAI GPT-4 | $3,000-30,000 |
| OpenAI GPT-3.5 | $200-2,000 |
| AWS Bedrock Claude | $1,500-15,000 |
| Self-hosted A100 | $3,000-5,000 |
| Self-hosted T4 | $300-800 |
| On-prem A100 | Capex + Power |

## Security Checklist

- [ ] API authentication (API keys or OAuth)
- [ ] Rate limiting per client
- [ ] Input validation and sanitization
- [ ] Output filtering (PII, harmful content)
- [ ] HTTPS/TLS encryption
- [ ] Network isolation (VPC, security groups)
- [ ] Audit logging
- [ ] Regular security updates
