import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from src.main import create_app


@pytest.fixture
def mock_components():
    embedder = MagicMock()
    store = MagicMock()
    generator = MagicMock()
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, chunks, top_k: chunks[:top_k]
    return embedder, store, generator, reranker


@pytest.fixture
def app(mock_components):
    embedder, store, generator, reranker = mock_components
    store.count.return_value = 42
    store.list_documents.return_value = [
        {"source_file": "test.pdf", "chunks": 10, "language": "en", "ingested_at": "2026-04-01T00:00:00"}
    ]
    return create_app(embedder=embedder, store=store, generator=generator, reranker=reranker)


@pytest.mark.asyncio
async def test_health(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["corpus_chunks"] == 42


@pytest.mark.asyncio
async def test_corpus(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/corpus")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 42
    assert len(data["documents"]) == 1


@pytest.mark.asyncio
async def test_query_endpoint(app, mock_components):
    embedder, store, generator, reranker = mock_components
    embedder.embed.return_value = [[0.1] * 384]
    store.query.return_value = [
        {"id": 1, "score": 0.9, "payload": {"text": "Answer context.", "source_file": "doc.pdf", "chunk_index": 0}},
    ]
    generator.generate.return_value = "The answer."

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/query", json={"question": "What is it?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "The answer."
    assert len(data["sources"]) == 1


@pytest.mark.asyncio
async def test_query_missing_question(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/query", json={})
    assert resp.status_code == 422
