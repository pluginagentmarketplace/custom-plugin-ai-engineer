# Chunking Strategies Guide

Choosing the right chunking strategy is critical for RAG performance.

## Overview

Chunking determines how documents are split into retrievable units.
The goal: Create chunks that are self-contained, contextually complete,
and appropriately sized for your embedding model.

## Strategy Comparison

| Strategy | Best For | Chunk Size | Overlap | Complexity |
|----------|----------|------------|---------|------------|
| Fixed | Simple docs | 500-1000 | 50-100 | Low |
| Sentence | Conversations | 3-5 sentences | 1 sentence | Low |
| Paragraph | Structured docs | Natural | None | Low |
| Recursive | Mixed content | Variable | Adaptive | Medium |
| Semantic | Long documents | By topic | Topic boundary | High |
| Agentic | Complex queries | Dynamic | Query-based | High |

## Fixed-Size Chunking

Split text at fixed character/token intervals.

```python
def fixed_chunk(text, size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

**Pros:**
- Simple and predictable
- Consistent chunk sizes
- Easy to implement

**Cons:**
- May split mid-sentence
- Context can be lost
- Suboptimal for varied content

**Use When:**
- Uniform document structure
- Need consistent chunk sizes
- Quick prototyping

## Sentence-Based Chunking

Group sentences together.

```python
import nltk

def sentence_chunk(text, sentences_per_chunk=3, overlap=1):
    sentences = nltk.sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk - overlap):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks
```

**Pros:**
- Preserves sentence boundaries
- Natural reading flow
- Good for Q&A

**Cons:**
- Variable chunk sizes
- May split related ideas
- Requires NLP library

**Use When:**
- Conversational content
- FAQ documents
- Customer support data

## Paragraph-Based Chunking

Use natural paragraph breaks.

```python
def paragraph_chunk(text, min_size=100, max_size=2000):
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < max_size:
            current += para + "\n\n"
        else:
            if len(current) >= min_size:
                chunks.append(current.strip())
            current = para + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks
```

**Pros:**
- Respects document structure
- Complete thoughts preserved
- No additional processing

**Cons:**
- Highly variable sizes
- Some paragraphs too long/short
- Depends on document formatting

**Use When:**
- Well-structured documents
- Technical documentation
- Blog posts and articles

## Recursive Chunking

Try multiple separators in order.

```python
def recursive_chunk(text, size=1000, separators=['\n\n', '\n', '. ', ' ']):
    if len(text) <= size:
        return [text]

    for sep in separators:
        if sep in text:
            splits = text.split(sep)
            chunks = []
            current = ""

            for split in splits:
                if len(current) + len(split) <= size:
                    current += split + sep
                else:
                    if current:
                        chunks.extend(recursive_chunk(current, size, separators[1:]))
                    current = split + sep

            if current:
                chunks.extend(recursive_chunk(current, size, separators[1:]))

            return chunks

    # No separator found, force split
    return [text[i:i+size] for i in range(0, len(text), size)]
```

**Pros:**
- Adaptive to content
- Preserves structure when possible
- Falls back gracefully

**Cons:**
- More complex logic
- Variable output sizes
- Harder to predict behavior

**Use When:**
- Mixed content types
- Markdown/HTML documents
- Code documentation

## Semantic Chunking

Group by meaning using embeddings.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def semantic_chunk(text, threshold=0.7):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Split into sentences
    sentences = text.split('. ')
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        # Compare current sentence to chunk average
        chunk_embedding = np.mean(
            model.encode(current_chunk), axis=0
        )
        similarity = cosine_similarity(
            [embeddings[i]], [chunk_embedding]
        )[0][0]

        if similarity >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append('. '.join(current_chunk))

    return chunks
```

**Pros:**
- Topic-coherent chunks
- Optimal for retrieval
- Adapts to content meaning

**Cons:**
- Computationally expensive
- Requires embedding model
- Slower processing

**Use When:**
- Long, varied documents
- Topic-based retrieval important
- Quality > Speed

## Agentic Chunking

LLM-guided chunking decisions.

```python
def agentic_chunk(text, llm_client, context=None):
    prompt = f"""Analyze this text and suggest optimal chunk boundaries.
Consider:
1. Topic changes
2. Logical sections
3. Self-contained units

Text: {text[:2000]}...

Suggest chunk boundaries (line numbers or markers):"""

    response = llm_client.generate(prompt)
    boundaries = parse_boundaries(response.text)

    # Apply suggested boundaries
    chunks = split_at_boundaries(text, boundaries)

    return chunks
```

**Pros:**
- Most intelligent chunking
- Context-aware decisions
- Handles edge cases

**Cons:**
- Expensive (LLM calls)
- Slow processing
- May not be deterministic

**Use When:**
- High-stakes retrieval
- Complex document structures
- Budget allows

## Chunk Size Guidelines

### By Embedding Model

| Model | Max Tokens | Recommended Chunk |
|-------|------------|-------------------|
| OpenAI text-embedding-3 | 8191 | 500-1000 |
| all-MiniLM-L6-v2 | 256 | 100-256 |
| BGE-large | 512 | 256-512 |
| E5-large | 512 | 256-512 |

### By Use Case

| Use Case | Chunk Size | Overlap | Reason |
|----------|------------|---------|--------|
| Q&A | 200-500 | 50-100 | Precise answers |
| Summarization | 1000-2000 | 100-200 | More context |
| Code | 500-1000 | 100-200 | Function-level |
| Legal | 500-1000 | 200 | Context important |
| Chat | 200-400 | 50 | Quick retrieval |

## Best Practices

### 1. Add Metadata

```python
chunk = {
    "text": chunk_text,
    "source": document_name,
    "page": page_number,
    "section": section_title,
    "chunk_index": i,
    "total_chunks": n
}
```

### 2. Include Context Headers

```python
def add_context(chunk, document):
    return f"Document: {document.title}\n\n{chunk}"
```

### 3. Handle Special Content

```python
# Preserve code blocks
if "```" in text:
    # Don't split inside code blocks
    pass

# Preserve tables
if "|" in text and "---" in text:
    # Keep tables together
    pass
```

### 4. Validate Chunks

```python
def validate_chunk(chunk):
    # Not too short
    if len(chunk) < 50:
        return False

    # Not too long
    if len(chunk) > 2000:
        return False

    # Has content (not just whitespace)
    if not chunk.strip():
        return False

    return True
```

## Decision Tree

```
Choose chunking strategy:
│
├── Simple, uniform docs?
│   └── Fixed-size chunking
│
├── Conversational/FAQ?
│   └── Sentence-based
│
├── Structured with headers?
│   └── Paragraph/Section-based
│
├── Mixed content?
│   └── Recursive chunking
│
├── Need topic coherence?
│   └── Semantic chunking
│
└── Complex, high-stakes?
    └── Agentic chunking
```

## Common Mistakes

1. **Chunks too small** → Loss of context
2. **Chunks too large** → Irrelevant content retrieved
3. **No overlap** → Missing boundary information
4. **Ignoring structure** → Split mid-sentence/paragraph
5. **One size fits all** → Different docs need different strategies
