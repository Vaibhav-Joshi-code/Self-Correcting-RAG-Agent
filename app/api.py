"""FastAPI entry point. Run one worker with embedded Chroma."""
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import get_settings
from app.provider import EuriProvider, ProviderError
from app.rag import RAGAgent
from app.schemas import AskRequest, AskResponse
from app.store import IndexNotReady, VectorStore

logger = logging.getLogger(__name__)
_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # Validate at startup; readiness checks the index separately.
    yield
    agent = getattr(app.state, "agent", None)
    if agent:
        agent.provider.close()
        app.state.agent = None


app = FastAPI(title="Self-Correcting RAG Agent", version="2.0.0",
              description="EURIAI + Chroma + LangGraph, with strict evaluation and LangSmith traces.", lifespan=lifespan)


def get_agent() -> RAGAgent:
    with _lock:
        if not getattr(app.state, "agent", None):
            settings = get_settings()
            provider = EuriProvider(settings)
            try:
                store = VectorStore(settings, provider)
                app.state.agent = RAGAgent(settings, provider, store)
            except Exception:
                provider.close()
                raise
    return app.state.agent


def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = get_settings().api_key.get_secret_value()
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.middleware("http")
async def request_identity(request: Request, call_next):
    request.state.request_id = str(uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready", dependencies=[Depends(require_api_key)])
def ready():
    try:
        agent = get_agent()
        count = agent.retriever.count()
        if not count:
            raise IndexNotReady("Knowledge base is empty. Run python -m app.ingest first.")
        return {"status": "ready", "indexed_chunks": count,
                "tracing_enabled": agent.settings.langsmith_tracing}
    except (IndexNotReady, ProviderError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except Exception as exc:
        logger.error("Readiness failed type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Knowledge base unavailable. Check configuration and ingestion.") from None


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(body: AskRequest, request: Request):
    request_id = request.state.request_id
    try:
        result = get_agent().ask(body.question, request_id=request_id)
        logger.info("RAG request=%s status=%s attempts=%s seconds=%s", request_id,
                    result["status"], result["attempts"], result["elapsed_seconds"])
        return result
    except IndexNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ProviderError as exc:
        logger.warning("Provider failed request=%s type=%s", request_id, type(exc).__name__)
        raise HTTPException(status_code=502, detail=str(exc)) from None
    except Exception as exc:
        # Deliberately omit arbitrary exception text and user content from logs.
        logger.error("RAG failed request=%s type=%s", request_id, type(exc).__name__)
        raise HTTPException(status_code=500, detail=f"Request failed. Reference: {request_id}") from None
