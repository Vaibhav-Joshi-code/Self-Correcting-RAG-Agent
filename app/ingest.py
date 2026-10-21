"""Run with python -m app.ingest. Synchronize the managed corpus before serving."""
import hashlib
import json
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from app.config import Settings, get_settings
from app.observability import trace_context
from app.provider import EuriProvider
from app.store import VectorStore


def load_chunks(settings: Settings) -> list[dict]:
    if not settings.data_dir.is_dir():
        raise ValueError(f"DATA_DIR does not exist: {settings.data_dir}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
                                              chunk_overlap=settings.chunk_overlap, add_start_index=True)
    chunks = []
    for path in sorted(settings.data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if not path.resolve().is_relative_to(settings.data_dir.resolve()):
            raise ValueError("Corpus symlinks must remain inside DATA_DIR")
        content = path.read_text(encoding="utf-8-sig")
        source = path.relative_to(settings.data_dir).as_posix()
        for doc in splitter.create_documents([content], metadatas=[{"source": source}]):
            identity = f"{source}|{doc.metadata['start_index']}|{doc.page_content}"
            chunks.append({"id": hashlib.sha256(identity.encode()).hexdigest(),
                           "content": doc.page_content, "metadata": doc.metadata})
    if not chunks:
        raise ValueError("No non-empty UTF-8 .md/.txt documents; existing index was left intact.")
    return chunks


@traceable(name="ingest_corpus")
def synchronize(chunks: list[dict], store: VectorStore, batch_size: int) -> dict:
    existing = set(store.collection.get(include=[])["ids"])
    current = {chunk["id"] for chunk in chunks}
    added = [chunk for chunk in chunks if chunk["id"] not in existing]
    # Do not prune old chunks until every new embedding/upsert succeeds.
    for offset in range(0, len(added), batch_size):
        batch = added[offset:offset + batch_size]
        vectors = store.provider.embed([chunk["content"] for chunk in batch])
        store.collection.upsert(ids=[c["id"] for c in batch], embeddings=vectors,
                                documents=[c["content"] for c in batch], metadatas=[c["metadata"] for c in batch])
    stale = sorted(existing - current)
    for offset in range(0, len(stale), batch_size):
        store.collection.delete(ids=stale[offset:offset + batch_size])
    return {"files": len({c["metadata"]["source"] for c in chunks}), "chunks": len(chunks),
            "added": len(added), "unchanged": len(current & existing), "removed": len(stale)}


def main():
    settings = get_settings()
    chunks = load_chunks(settings)
    provider = EuriProvider(settings)
    try:
        with trace_context(settings, str(uuid4())):
            result = synchronize(chunks, VectorStore(settings, provider), settings.embedding_batch_size)
        print(json.dumps(result, indent=2))
    finally:
        provider.close()


if __name__ == "__main__":
    main()
