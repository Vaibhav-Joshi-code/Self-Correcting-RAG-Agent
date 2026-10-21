import pytest
from app.config import Settings


@pytest.mark.parametrize("values", [
    {"chunk_size": 100, "chunk_overlap": 100}, {"max_retries": -1},
    {"max_retries": 6}, {"min_grounding": 6}, {"euri_base_url": "http://example.com"},
    {"top_k": 10, "max_retrieval_k": 4}, {"langsmith_tracing": True, "langsmith_api_key": ""},
])
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **values)
