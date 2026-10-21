"""Deterministic, SCRIPTED workflow examples; no EURIAI calls or LangSmith uploads."""
import argparse
import json
from pathlib import Path
from app.config import Settings
from app.rag import RAGAgent

EVIDENCE = {"chunk_id": "demo-leave", "source": "leave_policy.md", "distance": 0.1,
            "content": "Employees receive 24 days of paid annual leave per calendar year."}


class ScriptedRetriever:
    def search(self, query, k):
        return [EVIDENCE]


class ScriptedProvider:
    def __init__(self, scenario):
        self.scenario, self.attempt = scenario, 0

    def chat(self, messages, *, purpose):
        if purpose == "generate":
            self.attempt += 1
            if self.scenario == "failure":
                return "ChaiTech revenue was INR 10 billion. [S1]"
            if self.scenario == "recovery" and self.attempt == 1:
                return "Employees receive 50 days of annual leave. [S1]"
            return "Employees receive 24 days of paid annual leave per calendar year. [S1]"
        passed = self.scenario == "success" or (self.scenario == "recovery" and self.attempt > 1)
        return json.dumps({
            "retrieval_relevance": 5 if self.scenario != "failure" else 1,
            "grounding": 5 if passed else 1, "hallucination_risk": 1 if passed else 5,
            "completeness": 5 if passed else 1, "retry_required": not passed,
            "reason": "All claims supported by S1." if passed else "The draft invents a figure absent from S1.",
            "improved_query": "ChaiTech annual paid leave allowance" if self.scenario != "failure" else "ChaiTech 2025 revenue financial report",
            "missing_information": [] if passed else ["Verified answer to the original question"],
            "unsupported_claims": [] if passed else ["Invented figure"],
        })


def run_scenario(name):
    settings = Settings(_env_file=None, langsmith_tracing=False, max_retries=2, top_k=4, max_retrieval_k=12)
    question = "What was ChaiTech's revenue in 2025?" if name == "failure" else "How many annual leave days do employees receive?"
    result = RAGAgent(settings, ScriptedProvider(name), ScriptedRetriever()).ask(question)
    return {"mode": "scripted_offline_demo", "scenario": name, "result": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save full JSON evidence for all scenarios")
    args = parser.parse_args()
    results = [run_scenario(name) for name in ("success", "recovery", "failure")]
    for item in results:
        result = item["result"]
        print(f"SCRIPTED {item['scenario']}: {result['status']}; attempts={result['attempts']}")
        for attempt in result["history"]:
            print(f"  Attempt {attempt['attempt']}: accepted={attempt['accepted']} draft={attempt['answer']}")
        print(f"  Final: {result['answer']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
