import json
import httpx
from pydantic import SecretStr
from langsmith import Client
from app.observability import trace_context
from app.provider import EuriProvider
from app.rag import RAGAgent
from conftest import FakeProvider, FakeRetriever


def test_langsmith_parent_child_traces_without_network(settings, monkeypatch):
    events = []
    client = Client(api_key="trace-test-secret", api_url="https://api.smith.langchain.com", auto_batch_tracing=False)
    monkeypatch.setattr(Client, "create_run", lambda *args, **kwargs: events.append(("create", kwargs)))
    monkeypatch.setattr(Client, "update_run", lambda *args, **kwargs: events.append(("update", kwargs)))
    monkeypatch.setattr("app.observability._client", lambda endpoint, key: client)
    settings.langsmith_tracing = True
    settings.langsmith_api_key = SecretStr("trace-test-secret")
    result = RAGAgent(settings, FakeProvider(), FakeRetriever()).ask("Leave days?")
    assert result["trace_id"] is not None
    created = [event for kind, event in events if kind == "create"]
    names = {event["name"] for event in created}
    assert {"self_correcting_rag", "retrieve", "generate", "evaluate", "finish"} <= names
    assert any(event.get("parent_run_id") for event in created)
    assert "trace-test-secret" not in json.dumps(events, default=str)


def test_provider_trace_does_not_include_auth_headers(settings, monkeypatch):
    events = []
    client = Client(api_key="trace-test-secret", auto_batch_tracing=False)
    monkeypatch.setattr(Client, "create_run", lambda *args, **kwargs: events.append(kwargs))
    monkeypatch.setattr(Client, "update_run", lambda *args, **kwargs: events.append(kwargs))
    monkeypatch.setattr("app.observability._client", lambda endpoint, key: client)
    settings.langsmith_tracing = True
    settings.langsmith_api_key = SecretStr("trace-test-secret")
    settings.euri_api_key = SecretStr("provider-test-secret")
    body = {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "24 days [S1]"}}], "usage": None}
    provider = EuriProvider(settings, httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    try:
        with trace_context(settings, "test-request"):
            assert provider.chat([], purpose="generate") == "24 days [S1]"
    finally:
        provider.close()
    serialized = json.dumps(events, default=str)
    assert "provider-test-secret" not in serialized and "trace-test-secret" not in serialized
    assert any(event.get("name") == "euriai.chat" for event in events)
