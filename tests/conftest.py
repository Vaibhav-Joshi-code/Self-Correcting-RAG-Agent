"""Offline tests never read real keys or emit hosted traces."""
import json
import os
import pytest
from app.config import Settings

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture
def settings():
    return Settings(_env_file=None, euri_api_key="test-key", langsmith_tracing=False)


def evaluation(**changes):
    value = dict(retrieval_relevance=5, grounding=5, hallucination_risk=1,
                 completeness=5, retry_required=False, reason="All claims supported.",
                 improved_query="annual leave carry forward deadline", missing_information=[], unsupported_claims=[])
    return {**value, **changes}


DOC = {"chunk_id": "leave-1", "source": "leave_policy.md", "content": "Employees receive 24 days of annual leave. Carry forward up to five days; use them by March 31.", "distance": 0.1}


class FakeProvider:
    def __init__(self, answers=None, evaluations=None):
        self.answers = list(answers or ["Employees receive 24 days of annual leave. [S1]"])
        self.evaluations = list(evaluations or [evaluation()])
        self.calls = []

    def chat(self, messages, *, purpose):
        self.calls.append((purpose, messages))
        values = self.answers if purpose == "generate" else self.evaluations
        value = values.pop(0) if len(values) > 1 else values[0]
        return value if isinstance(value, str) else json.dumps(value)


class FakeRetriever:
    def __init__(self, documents=None):
        self.documents = [DOC] if documents is None else documents
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.documents
