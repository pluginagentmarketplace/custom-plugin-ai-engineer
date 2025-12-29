---
name: vector-databases
description: Vector database selection, indexing strategies, and semantic search optimization.
sasmp_version: "1.3.0"
bonded_agent: 01-llm-fundamentals
bond_type: SECONDARY_BOND
---

# Vector Databases

Master vector storage and retrieval for AI applications.

## Quick Start

### Chroma (Local Development)
```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize client
client = chromadb.Client()  # In-memory
# client = chromadb.PersistentClient(path="./chroma_db")  # Persistent

# Create collection with embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="documents",
    embedding_function=embedding_fn
)

# Add documents
collection.add(
    documents=["Document 1 text", "Document 2 text"],
    metadatas=[{"source": "file1"}, {"source": "file2"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=5
)
```

### Pinecone (Cloud Production)
```python
from pinecone import Pinecone, ServerlessSpec

# Initialize
pc = Pinecone(api_key="YOUR_API_KEY")

# Create index
pc.create_index(
    name="documents",
    dimension=1536,  # OpenAI embedding dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-west-2")
)

index = pc.Index("documents")

# Upsert vectors
index.upsert(vectors=[
    {"id": "doc1", "values": embedding1, "metadata": {"text": "..."}},
    {"id": "doc2", "values": embedding2, "metadata": {"text": "..."}}
])

# Query
results = index.query(
    vector=query_embedding,
    top_k=10,
    include_metadata=True
)
```

### Weaviate
```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType

# Connect
client = weaviate.connect_to_local()  # or connect_to_wcs()

# Create collection (class)
collection = client.collections.create(
    name="Document",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT)
    ]
)

# Add objects
collection.data.insert({
    "content": "Document text here",
    "source": "file.pdf"
})

# Semantic search
response = collection.query.near_text(
    query="search query",
    limit=5
)
```

## Database Comparison

| Feature | Chroma | Pinecone | Weaviate | Milvus | Qdrant |
|---------|--------|----------|----------|--------|--------|
| Deployment | Local/Cloud | Cloud | Self/Cloud | Self/Cloud | Self/Cloud |
| Ease of Use | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Scale | Small-Med | Large | Large | Very Large | Large |
| Filtering | Basic | Advanced | GraphQL | Advanced | Advanced |
| Cost | Free | Pay-per-use | Free/Paid | Free/Paid | Free/Paid |
| Best For | Dev/POC | Production | Hybrid Search | Enterprise | Production |

## Indexing Strategies

### HNSW (Hierarchical Navigable Small World)
```python
# Most common for approximate nearest neighbor
# Good balance of speed and accuracy

index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,              # Max connections per layer
        "efConstruction": 200 # Build-time accuracy
    }
}

search_params = {
    "ef": 100  # Search-time accuracy
}
```

### IVF (Inverted File Index)
```python
# Good for very large datasets
# Requires training phase

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {
        "nlist": 1024  # Number of clusters
    }
}

search_params = {
    "nprobe": 10  # Clusters to search
}
```

### Flat Index
```python
# Exact search, no approximation
# Use for small datasets (<100K vectors)

index_params = {
    "index_type": "FLAT",
    "metric_type": "COSINE"
}
```

## Distance Metrics

```python
# Cosine Similarity - Best for text embeddings
cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Euclidean Distance (L2) - Sensitive to magnitude
l2_dist = np.linalg.norm(a - b)

# Dot Product - For normalized vectors = cosine
dot_product = np.dot(a, b)

# When to use what:
# - Cosine: Text, semantic similarity
# - L2: Images, when magnitude matters
# - Dot Product: Pre-normalized vectors
```

## Hybrid Search

```python
class HybridSearch:
    def __init__(self, vector_store, bm25_index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def search(self, query: str, k: int = 10, alpha: float = 0.5):
        # Dense retrieval (semantic)
        dense_results = self.vector_store.search(query, k=k*2)

        # Sparse retrieval (keyword)
        sparse_results = self.bm25_index.search(query, k=k*2)

        # Reciprocal Rank Fusion
        scores = {}
        for rank, doc in enumerate(dense_results):
            scores[doc.id] = scores.get(doc.id, 0) + alpha / (rank + 60)
        for rank, doc in enumerate(sparse_results):
            scores[doc.id] = scores.get(doc.id, 0) + (1-alpha) / (rank + 60)

        # Sort and return top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]
```

## Metadata Filtering

```python
# Pinecone filtering
results = index.query(
    vector=embedding,
    top_k=10,
    filter={
        "category": {"$eq": "technical"},
        "date": {"$gte": "2024-01-01"},
        "$or": [
            {"author": "John"},
            {"author": "Jane"}
        ]
    }
)

# Chroma filtering
results = collection.query(
    query_embeddings=[embedding],
    n_results=10,
    where={
        "$and": [
            {"category": {"$eq": "technical"}},
            {"year": {"$gte": 2024}}
        ]
    }
)
```

## Performance Optimization

### Batch Operations
```python
# Insert in batches for better performance
BATCH_SIZE = 100

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    vectors = [(doc.id, doc.embedding, doc.metadata) for doc in batch]
    index.upsert(vectors=vectors)
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query_hash: str):
    return index.query(vector=query_embedding, top_k=10)
```

## Best Practices

1. **Choose right index**: HNSW for most cases, IVF for >1M vectors
2. **Normalize embeddings**: Ensures consistent similarity scores
3. **Use metadata**: Pre-filter before vector search
4. **Batch operations**: Better throughput for bulk inserts
5. **Monitor latency**: Set alerts for p99 response times
6. **Plan capacity**: Vectors grow fast, plan storage ahead
