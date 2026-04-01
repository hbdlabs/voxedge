from pathlib import Path
from unittest.mock import MagicMock

from src.embedder import Embedder
from src.store import VectorStore
from src.ingest import ingest_file
from src.query import query_brain


def test_ingest_then_query(tmp_path: Path):
    """Full pipeline: ingest a document, then query it."""

    # 1. Create a test document
    doc = tmp_path / "vaccines.txt"
    doc.write_text(
        "The WHO recommends that children under 5 receive DPT, polio, "
        "and measles vaccines. These should be given according to the "
        "national immunization schedule. Side effects are usually mild."
    )

    # 2. Real embedder + real store
    embedder = Embedder()
    store = VectorStore(path=str(tmp_path / "qdrant"), vector_size=384)

    # 3. Ingest
    result = ingest_file(doc, embedder, store, chunk_size=200, chunk_overlap=30)
    assert result.chunks > 0
    assert result.language == "en"
    assert store.count() == result.chunks

    # 4. Query — mock only the generator
    mock_gen = MagicMock()
    mock_gen.generate.return_value = "DPT, polio, and measles vaccines are recommended."

    answer = query_brain(
        question="What vaccines should children get?",
        embedder=embedder,
        store=store,
        generator=mock_gen,
        top_k=3,
        score_threshold=0.0,
    )

    # 5. Verify the right chunks were retrieved
    assert len(answer.sources) > 0
    assert answer.sources[0]["file"] == "vaccines.txt"

    # 6. Verify generator was called with relevant context
    call_args = mock_gen.generate.call_args
    chunks_passed = call_args.kwargs["chunks"]
    assert any("vaccine" in c.lower() for c in chunks_passed)

    # 7. Answer comes back
    assert "DPT" in answer.answer
