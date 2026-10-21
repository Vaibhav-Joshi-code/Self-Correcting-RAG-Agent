import pytest
from app.ingest import load_chunks, synchronize
from app.store import IndexNotReady, VectorStore


class LocalEmbeddings:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Embedding failed")
        return [[1.0, 0.1, 0.2] for _ in texts]


@pytest.fixture
def store(settings, tmp_path):
    settings.data_dir = tmp_path / "data"
    settings.data_dir.mkdir()
    settings.chroma_dir = tmp_path / "chroma"
    settings.collection_name = "test-policies"
    return VectorStore(settings, LocalEmbeddings())


def test_real_chroma_sync_idempotent_edit_delete_and_retrieve(store):
    folder = store.settings.data_dir
    (folder / "one.md").write_text("Annual leave is 24 days.", encoding="utf-8")
    (folder / "two.txt").write_text("Learning budget is INR 20000.", encoding="utf-8")
    first = synchronize(load_chunks(store.settings), store, 32)
    second = synchronize(load_chunks(store.settings), store, 32)
    assert first["added"] == 2 and second["added"] == 0
    assert store.provider.calls == 1
    (folder / "one.md").write_text("Annual leave is 25 days.", encoding="utf-8")
    (folder / "two.txt").unlink()
    third = synchronize(load_chunks(store.settings), store, 32)
    assert third["added"] == 1 and third["removed"] == 2 and store.count() == 1
    assert store.search("annual leave", 4)[0]["content"] == "Annual leave is 25 days."


def test_failed_embedding_preserves_old_chunks(store):
    path = store.settings.data_dir / "one.md"
    path.write_text("Original policy.", encoding="utf-8")
    synchronize(load_chunks(store.settings), store, 32)
    path.write_text("Updated policy.", encoding="utf-8")
    store.provider.fail = True
    with pytest.raises(RuntimeError):
        synchronize(load_chunks(store.settings), store, 32)
    assert store.collection.get()["documents"] == ["Original policy."]


def test_empty_corpus_preserves_index(store):
    with pytest.raises(ValueError, match="No non-empty"):
        load_chunks(store.settings)


def test_incompatible_embedding_model_requires_new_collection(store):
    store.settings.embedding_model = "different-model"
    with pytest.raises(IndexNotReady, match="configuration changed"):
        VectorStore(store.settings, store.provider)


def test_empty_index_is_not_ready(store):
    with pytest.raises(IndexNotReady, match="empty"):
        store.search("test", 4)
