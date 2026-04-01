from src.chunker import chunk_text


def test_chunk_short_text():
    """Text shorter than chunk_size returns single chunk."""
    text = "Hello world."
    result = chunk_text(text, chunk_size=500, overlap=50)
    assert result == ["Hello world."]


def test_chunk_splits_with_overlap():
    """Long text is split into overlapping chunks."""
    text = "A" * 200 + "B" * 200 + "C" * 200
    result = chunk_text(text, chunk_size=200, overlap=50)
    assert len(result) == 4
    # First chunk is 200 chars
    assert len(result[0]) == 200
    # Overlap: end of chunk N overlaps with start of chunk N+1
    assert result[0][-50:] == result[1][:50]
    assert result[1][-50:] == result[2][:50]


def test_chunk_empty_text():
    """Empty text returns empty list."""
    result = chunk_text("", chunk_size=500, overlap=50)
    assert result == []


def test_chunk_exact_size():
    """Text exactly chunk_size returns single chunk."""
    text = "X" * 500
    result = chunk_text(text, chunk_size=500, overlap=50)
    assert result == [text]
