"""Explicit tracing context; secrets never become trace arguments."""
from contextlib import contextmanager
from functools import lru_cache
from langsmith import Client, tracing_context
from app.config import Settings


@lru_cache(maxsize=4)
def _client(endpoint: str, key: str) -> Client:
    return Client(api_url=endpoint, api_key=key)


@contextmanager
def trace_context(settings: Settings, request_id: str):
    client = None
    if settings.langsmith_tracing:
        client = _client(settings.langsmith_endpoint, settings.langsmith_api_key.get_secret_value())
    with tracing_context(
        enabled=settings.langsmith_tracing, client=client,
        project_name=settings.langsmith_project, tags=["self-correcting-rag"],
        metadata={"request_id": request_id, "chat_model": settings.chat_model,
                  "embedding_model": settings.embedding_model},
    ):
        yield
