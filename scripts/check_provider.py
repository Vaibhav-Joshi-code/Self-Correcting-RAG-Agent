"""Isolate live EURIAI chat, single-text embedding, and batch support; no index writes."""
import argparse

from langsmith import tracing_context

from app.config import get_settings
from app.provider import EuriProvider, ProviderError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2, choices=range(1, 129))
    args = parser.parse_args()
    settings = get_settings().model_copy(update={"embedding_batch_size": args.batch_size})
    # This diagnostic uses synthetic text and does not upload hosted traces.
    provider = EuriProvider(settings)
    failed = False
    try:
        with tracing_context(enabled=False):
            checks = [
                ("chat", lambda: provider.chat([{"role": "user", "content": "Reply with OK."}], purpose="generate")),
                ("single-text embeddings", lambda: provider.embed(["Annual leave policy."])),
                (f"embeddings batch of {args.batch_size}", lambda: provider.embed(["Annual leave policy."] * args.batch_size)),
            ]
            for name, operation in checks:
                try:
                    result = operation()
                    detail = f"; vectors={len(result)}, dimensions={len(result[0])}" if isinstance(result, list) else ""
                    print(f"PASS {name}{detail}")
                except ProviderError as exc:
                    failed = True
                    print(f"FAIL {name}: {exc}")
    finally:
        provider.close()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
