from unittest.mock import MagicMock
from src.query import query_brain, QueryResult


def test_query_returns_answer_and_sources():
    """Full query pipeline returns answer with sources."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 384]

    mock_store = MagicMock()
    mock_store.query.return_value = [
        {"id": 1, "score": 0.9, "payload": {"text": "Vaccines help children.", "source_file": "who.pdf", "chunk_index": 3}},
        {"id": 2, "score": 0.8, "payload": {"text": "DPT is recommended.", "source_file": "who.pdf", "chunk_index": 4}},
    ]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Vaccines help protect children."

    result = query_brain(
        question="What about vaccines?",
        embedder=mock_embedder,
        store=mock_store,
        generator=mock_generator,
        top_k=5,
        score_threshold=0.5,
    )

    assert isinstance(result, QueryResult)
    assert result.answer == "Vaccines help protect children."
    assert len(result.sources) == 2
    assert result.sources[0]["file"] == "who.pdf"
    assert result.language == "en"


def test_query_no_results():
    """When no chunks match, generator still gets called with empty context."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 384]

    mock_store = MagicMock()
    mock_store.query.return_value = []

    mock_generator = MagicMock()

    result = query_brain(
        question="What is quantum physics?",
        embedder=mock_embedder,
        store=mock_store,
        generator=mock_generator,
    )

    assert result.answer == "I don't have information about that."
    assert result.sources == []
    mock_generator.generate.assert_not_called()
