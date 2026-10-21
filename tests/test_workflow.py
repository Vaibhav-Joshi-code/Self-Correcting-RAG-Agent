import json
import pytest
from app.rag import ABSTENTION, RAGAgent, parse_evaluation
from conftest import DOC, FakeProvider, FakeRetriever, evaluation


def test_success_first_attempt(settings):
    result = RAGAgent(settings, FakeProvider(), FakeRetriever()).ask("How much annual leave?")
    assert result["accepted"] and result["attempts"] == 1
    assert result["status"] == "accepted" and result["trace_id"] is None
    assert result["sources"][0]["id"] == "S1"


def test_rejected_draft_recovers_with_feedback_and_wider_retrieval(settings):
    provider = FakeProvider(answers=["Employees get 50 days. [S1]", "Employees get 24 days. [S1]"],
                            evaluations=[evaluation(grounding=1, hallucination_risk=5, retry_required=True,
                                                    unsupported_claims=["50 days"]), evaluation()])
    retriever = FakeRetriever()
    result = RAGAgent(settings, provider, retriever).ask("How much annual leave?")
    assert result["accepted"] and result["attempts"] == 2
    assert result["history"][0]["accepted"] is False
    assert [k for _, k in retriever.calls] == [4, 8]
    assert "Search focus:" in retriever.calls[1][0]
    assert "50 days" in provider.calls[2][1][1]["content"]


def test_exhaustion_returns_abstention_not_hallucination(settings):
    provider = FakeProvider(answers=["Revenue was 10 billion. [S1]"], evaluations=[evaluation(grounding=1, completeness=1)])
    result = RAGAgent(settings, provider, FakeRetriever()).ask("What is company revenue?")
    assert not result["accepted"] and result["attempts"] == 3
    assert result["answer"] == ABSTENTION
    assert result["history"][-1]["answer"] == "Revenue was 10 billion. [S1]"


@pytest.mark.parametrize("changes", [
    {"retry_required": "false"}, {"retry_required": 0}, {"grounding": True},
    {"grounding": 6}, {"grounding": "5"}, {"grounding": 4.5}, {"unexpected": 1},
    {"missing_information": "none"}, {"reason": ""},
])
def test_evaluator_schema_rejects_unsafe_coercions(changes):
    with pytest.raises(ValueError):
        parse_evaluation(json.dumps(evaluation(**changes)))


def test_fenced_json_and_duplicate_keys():
    assert parse_evaluation("```json\n" + json.dumps(evaluation()) + "\n```").grounding == 5
    with pytest.raises(ValueError):
        parse_evaluation('{"grounding":1,"grounding":5}')


def test_invalid_evaluation_fails_closed(settings):
    result = RAGAgent(settings, FakeProvider(evaluations=["not JSON"]), FakeRetriever()).ask("Leave days?")
    assert not result["accepted"] and result["attempts"] == 3
    assert "strict schema" in result["evaluation"]["reason"]


@pytest.mark.parametrize("answer", ["24 days.", "24 days. [S99]", ""])
def test_citation_gate_overrides_positive_judge(settings, answer):
    result = RAGAgent(settings, FakeProvider(answers=[answer]), FakeRetriever()).ask("Leave days?")
    assert not result["accepted"]


@pytest.mark.parametrize("change", [
    {"retrieval_relevance": 1}, {"grounding": 1}, {"hallucination_risk": 5},
    {"completeness": 1}, {"retry_required": True},
    {"missing_information": ["deadline"]}, {"unsupported_claims": ["24 days"]},
])
def test_every_quality_gate_can_force_rejection(settings, change):
    result = RAGAgent(settings, FakeProvider(evaluations=[evaluation(**change)]), FakeRetriever()).ask("Leave days?")
    assert not result["accepted"]


@pytest.mark.parametrize("retries", [0, 1, 5])
def test_retry_budget_including_largest_graph(settings, retries):
    settings.max_retries = retries
    result = RAGAgent(settings, FakeProvider(evaluations=[evaluation(completeness=1)]), FakeRetriever()).ask("Leave days?")
    assert result["attempts"] == retries + 1


def test_empty_retrieval_skips_llm_and_never_accepts(settings):
    provider = FakeProvider()
    result = RAGAgent(settings, provider, FakeRetriever([])).ask("Anything?")
    assert not result["accepted"] and provider.calls == []


def test_context_is_deduplicated_and_bounded(settings):
    settings.max_context_chars = 1000
    doc = {**DOC, "content": "a" * 2000}
    result = RAGAgent(settings, FakeProvider(), FakeRetriever([doc, doc])).ask("Leave days?")
    assert len(result["sources"]) == 1 and len(result["sources"][0]["content"]) == 1000


def test_questions_are_validated(settings):
    agent = RAGAgent(settings, FakeProvider(), FakeRetriever())
    for question in ["  ", "a" * 4001]:
        with pytest.raises(ValueError):
            agent.ask(question)


def test_requests_do_not_share_history(settings):
    agent = RAGAgent(settings, FakeProvider(), FakeRetriever())
    first, second = agent.ask("First?"), agent.ask("Second?")
    assert len(first["history"]) == len(second["history"]) == 1
    assert first["request_id"] != second["request_id"]
