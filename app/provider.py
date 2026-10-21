"""EURIAI through LangChain's OpenAI-compatible chat and embedding clients."""
import json
import math
import time
from collections.abc import Callable
from typing import Protocol, TypeVar

import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from openai import APIConnectionError, APIError, APIStatusError

from app.config import Settings

T = TypeVar("T")


class ProviderError(RuntimeError):
    """Sanitized error safe for API responses; no raw provider response bodies."""


class Provider(Protocol):
    def chat(self, messages: list[dict[str, str]], *, purpose: str) -> str: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _validate_embeddings(response: httpx.Response) -> None:
    """Reject malformed/reordered vectors before SDK numeric coercion can hide them."""
    if not response.is_success or not response.request.url.path.endswith("/embeddings"):
        return
    response.read()
    try:
        inputs = json.loads(response.request.content)["input"]
        expected = len(inputs) if isinstance(inputs, list) else 1
        rows = response.json()["data"]
        if len(rows) != expected or not rows:
            raise ValueError
        if any(type(row["index"]) is not int or row["index"] != i for i, row in enumerate(rows)):
            raise ValueError
        vectors = [row["embedding"] for row in rows]
        dimension = len(vectors[0])
        if not dimension or any(
            not isinstance(v, list) or len(v) != dimension or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
            for v in vectors
        ):
            raise ValueError
    except (KeyError, IndexError, TypeError, ValueError):
        raise ProviderError("EURIAI returned invalid or unordered embeddings.") from None


class EuriProvider:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        key = settings.euri_api_key.get_secret_value()
        if not key or key == "replace_me":
            raise ProviderError("Configure EURI_API_KEY before using EURIAI.")
        self.settings = settings
        self.client = httpx.Client(timeout=settings.request_timeout_seconds, transport=transport,
                                  event_hooks={"response": [_validate_embeddings]})
        common = dict(api_key=key, base_url=settings.euri_base_url.rstrip("/"),
                      timeout=settings.request_timeout_seconds, max_retries=0, http_client=self.client)
        # A single retry layer below avoids multiplying SDK and application retries.
        self.llm = ChatOpenAI(model=settings.chat_model, temperature=0.1,
                             max_tokens=settings.max_output_tokens, **common)
        self.evaluator = self.llm
        if settings.evaluator_model and settings.evaluator_model != settings.chat_model:
            self.evaluator = ChatOpenAI(model=settings.evaluator_model, temperature=0.1,
                                        max_tokens=settings.max_output_tokens, **common)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            check_embedding_ctx_length=False,
            chunk_size=settings.embedding_batch_size,
            # Avoid SDK base64 defaults and send ordinary text, never token-ID arrays.
            model_kwargs={"encoding_format": "float"},
            **common,
        )

    def close(self):
        self.client.close()

    def _call(self, operation: Callable[[], T], purpose: str) -> T:
        for attempt in range(self.settings.provider_max_retries + 1):
            delay = min(2 ** attempt, 8)
            try:
                return operation()
            except APIStatusError as exc:
                status = exc.status_code
                transient = status in {408, 429, 500, 502, 503, 504}
                if not transient or attempt == self.settings.provider_max_retries:
                    hint = ""
                    if purpose == "embeddings" and status >= 500:
                        hint = " Run python -m scripts.check_provider; verify embedding model access and try EMBEDDING_BATCH_SIZE=1."
                    raise ProviderError(f"EURIAI {purpose} failed (HTTP {status}).{hint}") from None
                try:
                    proposed = float(exc.response.headers.get("Retry-After", delay))
                    if math.isfinite(proposed):
                        delay = min(max(proposed, 0), 30)
                except ValueError:
                    pass
            except APIConnectionError as exc:
                # The SDK wraps response-hook validation errors as connection errors.
                if isinstance(exc.__cause__, ProviderError):
                    raise ProviderError(str(exc.__cause__)) from None
                if attempt == self.settings.provider_max_retries:
                    raise ProviderError(f"EURIAI {purpose} connection failed or timed out.") from None
            except APIError:
                raise ProviderError(f"EURIAI returned an invalid {purpose} response.") from None
            time.sleep(delay)
        raise ProviderError(f"EURIAI {purpose} failed.")

    @traceable(name="euriai.chat", run_type="chain")
    def chat(self, messages: list[dict[str, str]], *, purpose: str) -> str:
        model = self.evaluator if purpose == "evaluate" else self.llm
        try:
            response = self._call(lambda: model.invoke(messages), "chat")
            content = response.content
            if not isinstance(content, str) or not content.strip() or response.response_metadata.get("finish_reason") == "length":
                raise ValueError
            return content.strip()
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError("EURIAI returned an empty, truncated, or invalid completion.") from None

    @traceable(name="euriai.embed", run_type="embedding")
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty text strings.")
        vectors = []
        # Retry only the failed batch, without repeating previously successful batches.
        for offset in range(0, len(texts), self.settings.embedding_batch_size):
            batch = texts[offset:offset + self.settings.embedding_batch_size]
            try:
                result = self._call(lambda: self.embeddings.embed_documents(batch), "embeddings")
                if len(result) != len(batch):
                    raise ValueError
                vectors.extend(result)
            except (KeyError, IndexError, TypeError, ValueError):
                raise ProviderError("EURIAI returned invalid embeddings.") from None
        if any(len(v) != len(vectors[0]) for v in vectors):
            raise ProviderError("EURIAI embedding dimensions changed between batches.")
        return vectors
