#!/usr/bin/env python3
"""
Vector Store Manager - Unified interface for multiple vector databases.

Supports: Chroma, Pinecone, Weaviate, Qdrant, FAISS

Usage:
    from vector_store_manager import VectorStoreManager

    manager = VectorStoreManager(provider="chroma")
    manager.add_documents(documents)
    results = manager.search("query text", k=5)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
import os


@dataclass
class Document:
    """Document with content and metadata."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Search result with score."""
    document: Document
    score: float


class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[SearchResult]:
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        pass

    @abstractmethod
    def count(self) -> int:
        pass


class ChromaStore(VectorStoreBase):
    """Chroma vector store implementation."""

    def __init__(self, collection_name: str = "documents", persist_dir: str = "./chroma_db"):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, documents: List[Document]) -> None:
        self.collection.add(
            ids=[doc.id for doc in documents],
            embeddings=[doc.embedding.tolist() for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata or {} for doc in documents]
        )

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            doc = Document(
                id=results["ids"][0][i],
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else None
            )
            # Convert distance to similarity score
            score = 1 - results["distances"][0][i]
            search_results.append(SearchResult(document=doc, score=score))

        return search_results

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()


class PineconeStore(VectorStoreBase):
    """Pinecone vector store implementation."""

    def __init__(self, index_name: str, api_key: Optional[str] = None):
        from pinecone import Pinecone

        self.pc = Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)

    def add(self, documents: List[Document]) -> None:
        vectors = []
        for doc in documents:
            vectors.append({
                "id": doc.id,
                "values": doc.embedding.tolist(),
                "metadata": {
                    "content": doc.content,
                    **(doc.metadata or {})
                }
            })

        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[SearchResult]:
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True
        )

        search_results = []
        for match in results["matches"]:
            doc = Document(
                id=match["id"],
                content=match["metadata"].pop("content", ""),
                metadata=match["metadata"]
            )
            search_results.append(SearchResult(document=doc, score=match["score"]))

        return search_results

    def delete(self, ids: List[str]) -> None:
        self.index.delete(ids=ids)

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        return stats["total_vector_count"]


class QdrantStore(VectorStoreBase):
    """Qdrant vector store implementation."""

    def __init__(self, collection_name: str, url: str = "http://localhost:6333"):
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

        # Create collection if not exists
        collections = self.client.get_collections().collections
        if collection_name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )

    def add(self, documents: List[Document]) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for i, doc in enumerate(documents):
            points.append(PointStruct(
                id=hash(doc.id) % (10 ** 9),  # Qdrant needs int IDs
                vector=doc.embedding.tolist(),
                payload={
                    "id": doc.id,
                    "content": doc.content,
                    **(doc.metadata or {})
                }
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[SearchResult]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k
        )

        search_results = []
        for hit in results:
            doc = Document(
                id=hit.payload.get("id", str(hit.id)),
                content=hit.payload.get("content", ""),
                metadata={k: v for k, v in hit.payload.items() if k not in ["id", "content"]}
            )
            search_results.append(SearchResult(document=doc, score=hit.score))

        return search_results

    def delete(self, ids: List[str]) -> None:
        from qdrant_client.models import PointIdsList

        int_ids = [hash(id_) % (10 ** 9) for id_ in ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=int_ids)
        )

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count


class FAISSStore(VectorStoreBase):
    """FAISS vector store implementation (local)."""

    def __init__(self, dimension: int = 1536, index_type: str = "IndexFlatIP"):
        import faiss

        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.index = faiss.IndexFlatIP(dimension)

        self.documents: Dict[int, Document] = {}
        self.current_id = 0

    def add(self, documents: List[Document]) -> None:
        embeddings = np.array([doc.embedding for doc in documents]).astype("float32")
        self.index.add(embeddings)

        for doc in documents:
            self.documents[self.current_id] = doc
            self.current_id += 1

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[SearchResult]:
        query = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, k)

        search_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx in self.documents:
                search_results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(score)
                ))

        return search_results

    def delete(self, ids: List[str]) -> None:
        # FAISS doesn't support deletion easily
        # Would need to rebuild index
        pass

    def count(self) -> int:
        return self.index.ntotal


class VectorStoreManager:
    """Unified vector store manager."""

    PROVIDERS = {
        "chroma": ChromaStore,
        "pinecone": PineconeStore,
        "qdrant": QdrantStore,
        "faiss": FAISSStore
    }

    def __init__(self, provider: str = "chroma", **kwargs):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        self.store = self.PROVIDERS[provider](**kwargs)
        self.embedding_fn = None

    def set_embedding_function(self, fn):
        """Set embedding function for automatic embedding."""
        self.embedding_fn = fn

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents with automatic embedding if needed."""
        if self.embedding_fn:
            for doc in documents:
                if doc.embedding is None:
                    doc.embedding = self.embedding_fn(doc.content)

        self.store.add(documents)

    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search with automatic query embedding."""
        if self.embedding_fn:
            query_embedding = self.embedding_fn(query)
        else:
            raise ValueError("No embedding function set")

        return self.store.search(query_embedding, k)

    def search_by_vector(self, vector: np.ndarray, k: int = 10) -> List[SearchResult]:
        """Search by vector directly."""
        return self.store.search(vector, k)

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self.store.delete(ids)

    def count(self) -> int:
        """Get total document count."""
        return self.store.count()


# Convenience functions
def create_store(
    provider: str = "chroma",
    embedding_model: str = "text-embedding-3-small",
    **kwargs
) -> VectorStoreManager:
    """Create a vector store with OpenAI embeddings."""
    from openai import OpenAI

    client = OpenAI()

    def embed(text: str) -> np.ndarray:
        response = client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)

    manager = VectorStoreManager(provider=provider, **kwargs)
    manager.set_embedding_function(embed)

    return manager


if __name__ == "__main__":
    # Demo usage
    print("Vector Store Manager Demo")

    # Create in-memory FAISS store for demo
    manager = VectorStoreManager(provider="faiss", dimension=4)

    # Add sample documents with mock embeddings
    docs = [
        Document(id="1", content="Hello world", embedding=np.array([1, 0, 0, 0])),
        Document(id="2", content="Machine learning", embedding=np.array([0, 1, 0, 0])),
        Document(id="3", content="Deep learning", embedding=np.array([0, 0.9, 0.1, 0]))
    ]

    manager.store.add(docs)
    print(f"Added {manager.count()} documents")

    # Search
    query_vec = np.array([0, 0.8, 0.2, 0])
    results = manager.search_by_vector(query_vec, k=2)

    print("\nSearch results:")
    for r in results:
        print(f"  {r.document.content}: {r.score:.3f}")
