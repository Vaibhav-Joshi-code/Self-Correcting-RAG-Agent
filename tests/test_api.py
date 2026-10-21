from fastapi.testclient import TestClient
from pydantic import SecretStr
from app import api
from app.provider import ProviderError
from app.rag import RAGAgent
from app.store import IndexNotReady
from conftest import FakeProvider, FakeRetriever


def test_browser_health_validation_and_response(settings, monkeypatch):
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "get_agent", lambda: RAGAgent(settings, FakeProvider(), FakeRetriever()))
    with TestClient(api.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json() == {"status": "ok"}
        for question in ["", "   ", "a" * 4001]:
            assert client.post("/ask", json={"question": question}).status_code == 422
        response = client.post("/ask", json={"question": "Leave days?"})
        assert response.status_code == 200 and response.json()["accepted"]
        assert response.headers["X-Request-ID"] == response.json()["request_id"]


def test_optional_auth(settings, monkeypatch):
    settings.api_key = SecretStr("app-secret")
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "get_agent", lambda: RAGAgent(settings, FakeProvider(), FakeRetriever()))
    with TestClient(api.app) as client:
        assert client.post("/ask", json={"question": "test"}).status_code == 401
        assert client.post("/ask", headers={"X-API-Key": "app-secret"}, json={"question": "test"}).status_code == 200


def test_service_errors_are_distinguished_and_sanitized(settings, monkeypatch):
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    with TestClient(api.app) as client:
        for error, status in [(ProviderError("EURIAI unavailable"), 502), (IndexNotReady("Empty index"), 503),
                              (RuntimeError("secret-key-123"), 500)]:
            def fail():
                raise error
            monkeypatch.setattr(api, "get_agent", fail)
            response = client.post("/ask", json={"question": "test"})
            assert response.status_code == status and "secret-key-123" not in response.text
