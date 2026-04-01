from pathlib import Path
from unittest.mock import MagicMock
from src.ingest import ingest_file, IngestResult


def test_ingest_txt_file(tmp_path: Path):
    """Ingest a text file: parses, chunks, embeds, stores."""
    f = tmp_path / "test.txt"
    f.write_text("Hello world. " * 100)  # ~1300 chars, will produce multiple chunks

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

    mock_store = MagicMock()
    mock_store.count.return_value = 0  # starting point_id

    result = ingest_file(
        path=f,
        embedder=mock_embedder,
        store=mock_store,
        chunk_size=500,
        chunk_overlap=50,
    )

    assert isinstance(result, IngestResult)
    assert result.file == "test.txt"
    assert result.chunks > 0
    assert mock_embedder.embed.called
    assert mock_store.upsert_batch.called


def test_ingest_empty_file(tmp_path: Path):
    """Empty file produces zero chunks."""
    f = tmp_path / "empty.txt"
    f.write_text("")

    mock_embedder = MagicMock()
    mock_store = MagicMock()
    mock_store.count.return_value = 0

    result = ingest_file(f, mock_embedder, mock_store)

    assert result.chunks == 0
    assert not mock_store.upsert_batch.called
