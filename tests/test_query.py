from unittest.mock import MagicMock, patch
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


@patch("src.query.detect_language", return_value="vi")
def test_query_translates_foreign_query_for_retrieval(_detect):
    """translate_queries: a non-corpus-language query is translated before
    embedding + reranking, but generation uses the ORIGINAL question."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 384]

    chunk = {"id": 1, "score": 0.9, "payload": {"text": "TB is caused by bacteria.", "source_file": "tb.txt", "chunk_index": 0}}
    mock_store = MagicMock()
    mock_store.query.return_value = [chunk]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [chunk]

    mock_generator = MagicMock()
    mock_generator.translate.return_value = "What is tuberculosis?"
    mock_generator.generate.return_value = "Bệnh lao là một bệnh nhiễm trùng."

    result = query_brain(
        question="Bệnh lao là gì?",
        embedder=mock_embedder,
        store=mock_store,
        generator=mock_generator,
        reranker=mock_reranker,
        translate_queries=True,
        corpus_language="English",
    )

    # query was translated to the corpus language for retrieval...
    mock_generator.translate.assert_called_once()
    assert mock_generator.translate.call_args.kwargs["target_lang"] == "English"
    assert mock_generator.translate.call_args.kwargs["source_lang"] == "Vietnamese"
    # ...and the translated text (not the original) drove embedding + reranking
    mock_embedder.embed.assert_called_once_with(["What is tuberculosis?"])
    assert mock_reranker.rerank.call_args.kwargs["query"] == "What is tuberculosis?"
    # ...but generation used the ORIGINAL question, so the answer stays Vietnamese
    assert mock_generator.generate.call_args.kwargs["question"] == "Bệnh lao là gì?"
    assert result.language == "vi"


@patch("src.query.detect_language", return_value="en")
def test_query_skips_translation_when_already_corpus_language(_detect):
    """An English query against an English corpus is not translated."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 384]
    mock_store = MagicMock()
    mock_store.query.return_value = []
    mock_generator = MagicMock()

    query_brain(
        question="What is malaria?",
        embedder=mock_embedder,
        store=mock_store,
        generator=mock_generator,
        translate_queries=True,
        corpus_language="English",
    )

    mock_generator.translate.assert_not_called()
    mock_embedder.embed.assert_called_once_with(["What is malaria?"])
