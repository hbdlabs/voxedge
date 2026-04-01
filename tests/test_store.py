import pytest
from pathlib import Path
from src.store import VectorStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(path=str(tmp_path / "qdrant"), vector_size=4)


def test_upsert_and_count(store: VectorStore):
    store.upsert(
        point_id=1,
        vector=[1.0, 2.0, 3.0, 4.0],
        payload={"text": "hello", "source_file": "test.txt"},
    )
    store.upsert(
        point_id=2,
        vector=[5.0, 6.0, 7.0, 8.0],
        payload={"text": "world", "source_file": "test.txt"},
    )
    assert store.count() == 2


def test_query_returns_nearest(store: VectorStore):
    store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"text": "north"})
    store.upsert(2, [0.0, 1.0, 0.0, 0.0], {"text": "east"})
    store.upsert(3, [0.9, 0.1, 0.0, 0.0], {"text": "north-ish"})
    results = store.query(vector=[1.0, 0.0, 0.0, 0.0], limit=2)
    assert len(results) == 2
    texts = [r["payload"]["text"] for r in results]
    assert "north" in texts
    assert "north-ish" in texts


def test_scroll_all(store: VectorStore):
    store.upsert(1, [1.0, 0.0, 0.0, 0.0], {"text": "a", "source_file": "a.txt"})
    store.upsert(2, [0.0, 1.0, 0.0, 0.0], {"text": "b", "source_file": "b.txt"})
    docs = store.list_documents()
    files = {d["source_file"] for d in docs}
    assert files == {"a.txt", "b.txt"}


def test_empty_store_count(store: VectorStore):
    assert store.count() == 0
