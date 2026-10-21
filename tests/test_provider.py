import json
import httpx
import pytest
from app.provider import EuriProvider, ProviderError


def test_endpoint_auth_and_payload(settings):
    def handle(request):
        assert str(request.url) == "https://api.euron.one/api/v1/euri/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["temperature"] == 0.1
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop", "index": 0}]})
    provider = EuriProvider(settings, httpx.MockTransport(handle))
    try:
        assert provider.chat([{"role": "user", "content": "Hi"}], purpose="generate") == "ok"
    finally:
        provider.close()


def test_transient_http_retry(settings, monkeypatch):
    statuses = [429, 503, 200]
    sleeps = []
    monkeypatch.setattr("app.provider.time.sleep", sleeps.append)
    def handle(request):
        return httpx.Response(statuses.pop(0), headers={"Retry-After": "1"}, json={"data": [{"index": 0, "embedding": [1, 0]}]})
    provider = EuriProvider(settings, httpx.MockTransport(handle))
    try:
        assert provider.embed(["test"]) == [[1, 0]]
        assert len(sleeps) == 2
    finally:
        provider.close()


def test_auth_failure_not_retried_or_leaked(settings):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(401, text="private provider details and test-key")
    provider = EuriProvider(settings, httpx.MockTransport(handle))
    try:
        with pytest.raises(ProviderError, match="401") as exc:
            provider.embed(["test"])
        assert "test-key" not in str(exc.value) and len(calls) == 1
    finally:
        provider.close()


@pytest.mark.parametrize("data", [[], [{"index": 1, "embedding": [1]}],
                                    [{"index": 0, "embedding": []}], [{"index": 0, "embedding": [True]}]])
def test_malformed_embedding_response(settings, data):
    provider = EuriProvider(settings, httpx.MockTransport(lambda request: httpx.Response(200, json={"data": data})))
    try:
        with pytest.raises(ProviderError):
            provider.embed(["test"])
    finally:
        provider.close()


def test_timeout_retries_are_bounded(settings, monkeypatch):
    calls = []
    monkeypatch.setattr("app.provider.time.sleep", lambda seconds: None)
    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("secret details")
    provider = EuriProvider(settings, httpx.MockTransport(handle))
    try:
        with pytest.raises(ProviderError, match="timed out"):
            provider.embed(["test"])
        assert len(calls) == settings.provider_max_retries + 1
    finally:
        provider.close()


def test_nonfinite_retry_header_uses_safe_backoff(settings, monkeypatch):
    sleeps = []
    monkeypatch.setattr("app.provider.time.sleep", sleeps.append)
    provider = EuriProvider(settings, httpx.MockTransport(lambda request: httpx.Response(429, headers={"Retry-After": "NaN"})))
    try:
        with pytest.raises(ProviderError):
            provider.embed(["test"])
        assert sleeps == [1, 2]
    finally:
        provider.close()


def test_langchain_raw_text_embedding_payload_and_batches(settings):
    settings.embedding_batch_size = 1
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert request.url.path.endswith("/embeddings")
        assert request.headers["Authorization"] == "Bearer test-key"
        assert payload["encoding_format"] == "float"
        assert payload["model"] == settings.embedding_model
        assert len(payload["input"]) == 1 and isinstance(payload["input"][0], str)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.2, 0.8]}]})
    provider = EuriProvider(settings, httpx.MockTransport(handle))
    try:
        assert provider.embeddings.check_embedding_ctx_length is False
        assert provider.embed(["first policy", "second policy"]) == [[0.2, 0.8], [0.2, 0.8]]
        assert [c["input"] for c in calls] == [["first policy"], ["second policy"]]
    finally:
        provider.close()


def test_embedding_500_has_actionable_message_without_response_body(settings, monkeypatch):
    calls = []
    monkeypatch.setattr("app.provider.time.sleep", lambda _: None)
    def handle(request):
        calls.append(request)
        return httpx.Response(500, json={"error": {"message": "private details"}})
    provider = EuriProvider(settings, httpx.MockTransport(handle))
    try:
        with pytest.raises(ProviderError, match="EMBEDDING_BATCH_SIZE=1") as error:
            provider.embed(["test"])
        assert "private details" not in str(error.value)
        assert len(calls) == settings.provider_max_retries + 1
    finally:
        provider.close()


def test_retry_does_not_repeat_successful_embedding_batches(settings, monkeypatch):
    settings.embedding_batch_size = 1
    seen = []
    monkeypatch.setattr("app.provider.time.sleep", lambda _: None)
    def handle(request):
        text = json.loads(request.content)["input"][0]
        seen.append(text)
        if len(seen) == 2:
            return httpx.Response(503, json={"error": "temporarily unavailable"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.9]}]})
    provider = EuriProvider(settings, httpx.MockTransport(handle))
    try:
        assert len(provider.embed(["first", "second"])) == 2
        assert seen == ["first", "second", "second"]
    finally:
        provider.close()


def test_distinct_evaluator_model_uses_same_euriai_endpoint(settings):
    settings.evaluator_model = "configured-judge-model"
    models = []
    def handle(request):
        assert str(request.url).endswith("/euri/chat/completions")
        models.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]})
    provider = EuriProvider(settings, httpx.MockTransport(handle))
    try:
        provider.chat([{"role": "user", "content": "test"}], purpose="generate")
        provider.chat([{"role": "user", "content": "test"}], purpose="evaluate")
        assert models == [settings.chat_model, settings.evaluator_model]
    finally:
        provider.close()
