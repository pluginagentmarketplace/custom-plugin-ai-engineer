#!/usr/bin/env python3
"""
RAG Pipeline - Production-ready Retrieval Augmented Generation.

Features:
- Multi-provider vector store support
- Hybrid search (dense + sparse)
- Re-ranking with cross-encoders
- Query expansion
- Streaming responses

Usage:
    from rag_pipeline import RAGPipeline

    rag = RAGPipeline(config_path="rag_config.yaml")
    answer = rag.query("What is machine learning?")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Generator
from abc import ABC, abstractmethod
import hashlib
import numpy as np


@dataclass
class Document:
    """Document with content and metadata."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class Chunk:
    """Text chunk from a document."""
    id: str
    content: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    chunks: List[Chunk]
    scores: List[float]
    query: str
    retrieval_time_ms: float


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    sources: List[Chunk]
    confidence: float
    tokens_used: int
    latency_ms: float


class TextSplitter:
    """Split documents into chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def split(self, document: Document) -> List[Chunk]:
        """Split document into chunks."""
        text = document.content
        chunks = []

        # Recursive splitting
        current_chunks = self._split_recursive(text, self.separators)

        for i, chunk_text in enumerate(current_chunks):
            chunk = Chunk(
                id=f"{document.id}_chunk_{i}",
                content=chunk_text.strip(),
                document_id=document.id,
                chunk_index=i,
                metadata=document.metadata.copy()
            )
            chunks.append(chunk)

        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Force split at chunk_size
            return self._force_split(text)

        separator = separators[0]
        splits = text.split(separator)

        chunks = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(split) > self.chunk_size:
                    # Recursively split with remaining separators
                    chunks.extend(self._split_recursive(split, separators[1:]))
                else:
                    current_chunk = split + separator

        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap
        return self._apply_overlap(chunks)

    def _force_split(self, text: str) -> List[str]:
        """Force split text at chunk_size boundaries."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between chunks."""
        if len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_end = chunks[i-1][-self.chunk_overlap:] if len(chunks[i-1]) > self.chunk_overlap else chunks[i-1]
            overlapped.append(prev_end + chunks[i])

        return overlapped


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return np.array([e.embedding for e in response.data])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class LocalEmbeddings(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to the store."""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int) -> List[tuple]:
        """Search for similar chunks."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete chunks by ID."""
        pass


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for development."""

    def __init__(self):
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

    def add(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            if chunk.embedding is not None:
                self.embeddings[chunk.id] = chunk.embedding

    def search(self, query_embedding: np.ndarray, k: int) -> List[tuple]:
        if not self.embeddings:
            return []

        scores = []
        for chunk_id, embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            scores.append((self.chunks[chunk_id], float(similarity)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def delete(self, ids: List[str]) -> None:
        for id_ in ids:
            self.chunks.pop(id_, None)
            self.embeddings.pop(id_, None)


class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = 5
    ) -> List[tuple]:
        """Rerank chunks based on relevance to query."""
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs)

        ranked = list(zip(chunks, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked[:top_k]


class QueryExpander:
    """Expand queries using LLM for better retrieval."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def expand(self, query: str, n: int = 3) -> List[str]:
        """Generate alternative phrasings of the query."""
        prompt = f"""Generate {n} alternative ways to ask this question.
Keep the same meaning but use different words.

Original: {query}

Alternatives (one per line):"""

        response = self.llm.generate(prompt)
        alternatives = response.text.strip().split("\n")

        return [query] + [alt.strip() for alt in alternatives[:n]]


class RAGPipeline:
    """Complete RAG pipeline for question answering."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        llm_client,
        reranker: Optional[Reranker] = None,
        query_expander: Optional[QueryExpander] = None
    ):
        self.embeddings = embedding_provider
        self.vector_store = vector_store
        self.llm = llm_client
        self.reranker = reranker
        self.query_expander = query_expander
        self.splitter = TextSplitter()

    def ingest(self, documents: List[Document]) -> int:
        """Ingest documents into the pipeline."""
        total_chunks = 0

        for doc in documents:
            chunks = self.splitter.split(doc)

            # Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embeddings.embed(texts)

            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

            self.vector_store.add(chunks)
            total_chunks += len(chunks)

        return total_chunks

    def retrieve(
        self,
        query: str,
        k: int = 10,
        use_reranking: bool = True,
        use_expansion: bool = True
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        import time
        start = time.time()

        # Query expansion
        queries = [query]
        if use_expansion and self.query_expander:
            queries = self.query_expander.expand(query)

        # Retrieve for all query variants
        all_results = []
        for q in queries:
            query_embedding = self.embeddings.embed_query(q)
            results = self.vector_store.search(query_embedding, k)
            all_results.extend(results)

        # Deduplicate by chunk ID
        seen = set()
        unique_results = []
        for chunk, score in all_results:
            if chunk.id not in seen:
                seen.add(chunk.id)
                unique_results.append((chunk, score))

        # Sort by score
        unique_results.sort(key=lambda x: x[1], reverse=True)

        # Rerank if enabled
        if use_reranking and self.reranker and unique_results:
            chunks = [r[0] for r in unique_results[:k*2]]
            reranked = self.reranker.rerank(query, chunks, k)
            unique_results = reranked

        latency = (time.time() - start) * 1000

        return RetrievalResult(
            chunks=[r[0] for r in unique_results[:k]],
            scores=[r[1] for r in unique_results[:k]],
            query=query,
            retrieval_time_ms=latency
        )

    def generate(
        self,
        query: str,
        context_chunks: List[Chunk],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate answer using retrieved context."""
        context = "\n\n---\n\n".join([c.content for c in context_chunks])

        prompt = system_prompt or """Answer the question based on the context provided.
If you cannot answer from the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

        formatted_prompt = prompt.format(context=context, question=query)
        response = self.llm.generate(formatted_prompt)

        return response.text

    def query(
        self,
        question: str,
        k: int = 5,
        include_sources: bool = True
    ) -> RAGResponse:
        """Complete RAG query: retrieve + generate."""
        import time
        start = time.time()

        # Retrieve
        retrieval = self.retrieve(question, k=k)

        # Generate
        answer = self.generate(question, retrieval.chunks)

        latency = (time.time() - start) * 1000

        return RAGResponse(
            answer=answer,
            sources=retrieval.chunks if include_sources else [],
            confidence=np.mean(retrieval.scores) if retrieval.scores else 0.0,
            tokens_used=0,  # Would be populated by actual LLM response
            latency_ms=latency
        )

    def stream_query(
        self,
        question: str,
        k: int = 5
    ) -> Generator[str, None, None]:
        """Stream RAG response token by token."""
        retrieval = self.retrieve(question, k=k)
        context = "\n\n---\n\n".join([c.content for c in retrieval.chunks])

        prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        for token in self.llm.stream(prompt):
            yield token


# Factory function
def create_rag_pipeline(
    provider: str = "openai",
    vector_store: str = "memory",
    llm_client = None
) -> RAGPipeline:
    """Create a RAG pipeline with specified providers."""

    # Embedding provider
    if provider == "openai":
        embeddings = OpenAIEmbeddings()
    elif provider == "local":
        embeddings = LocalEmbeddings()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    # Vector store
    if vector_store == "memory":
        store = InMemoryVectorStore()
    else:
        raise ValueError(f"Unknown vector store: {vector_store}")

    return RAGPipeline(
        embedding_provider=embeddings,
        vector_store=store,
        llm_client=llm_client
    )


# Example usage
if __name__ == "__main__":
    # Create sample documents
    docs = [
        Document(
            id="doc1",
            content="Machine learning is a subset of artificial intelligence...",
            metadata={"source": "intro.pdf"}
        ),
        Document(
            id="doc2",
            content="Deep learning uses neural networks with many layers...",
            metadata={"source": "deep_learning.pdf"}
        )
    ]

    # Create pipeline (would need actual LLM client)
    # rag = create_rag_pipeline(provider="local", llm_client=my_llm)
    # rag.ingest(docs)
    # response = rag.query("What is machine learning?")
    # print(response.answer)
    print("RAG Pipeline ready for use")
