"""Persistent cosine vector index with explicit embedding compatibility checks."""
import hashlib
from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings
from langsmith import traceable

from app.config import Settings
from app.provider import Provider


class IndexNotReady(RuntimeError):
    pass


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[dict]: ...


def index_signature(settings: Settings) -> str:
    value = f"{settings.euri_base_url.rstrip('/')}|{settings.embedding_model}"
    return hashlib.sha256(value.encode()).hexdigest()


class VectorStore:
    def __init__(self, settings: Settings, provider: Provider):
        self.settings = settings
        self.provider = provider
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            settings.collection_name, embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_signature": index_signature(settings)},
        )
        metadata = self.collection.metadata or {}
        if metadata.get("embedding_signature") != index_signature(settings):
            raise IndexNotReady("Index embedding configuration changed. Ingest into a new COLLECTION_NAME.")

    def count(self) -> int:
        return self.collection.count()

    @traceable(name="vector_search", run_type="retriever")
    def search(self, query: str, k: int) -> list[dict]:
        count = self.count()
        if not count:
            raise IndexNotReady("Knowledge base is empty. Run python -m app.ingest first.")
        vectors = self.provider.embed([query])
        result = self.collection.query(query_embeddings=vectors, n_results=min(k, count),
                                       include=["documents", "metadatas", "distances"])
        return [
            {"chunk_id": chunk_id, "source": metadata["source"], "content": content,
             "distance": float(distance)}
            for chunk_id, metadata, content, distance in zip(
                result["ids"][0], result["metadatas"][0], result["documents"][0], result["distances"][0], strict=True,
            )
        ]
