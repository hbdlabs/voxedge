from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langdetect import detect

from src.chunker import chunk_text
from src.embedder import Embedder
from src.parser import parse_file
from src.store import VectorStore


@dataclass
class IngestResult:
    file: str
    chunks: int
    language: str


def ingest_file(
    path: Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> IngestResult:
    """Parse, chunk, embed, and store a single document."""
    text = parse_file(path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)

    if not chunks:
        return IngestResult(file=path.name, chunks=0, language="unknown")

    language = _detect_language(chunks[0])
    vectors = embedder.embed(chunks)
    start_id = store.count()
    now = datetime.now(timezone.utc).isoformat()

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append((
            start_id + i + 1,
            vector,
            {
                "text": chunk,
                "source_file": path.name,
                "chunk_index": i,
                "language": language,
                "ingested_at": now,
            },
        ))

    store.upsert_batch(points)
    return IngestResult(file=path.name, chunks=len(chunks), language=language)


def ingest_directory(
    directory: Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[IngestResult]:
    """Ingest all supported files in a directory."""
    results = []
    supported = {".txt", ".md", ".pdf", ".docx", ".doc", ".pptx", ".xlsx"}
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in supported:
            result = ingest_file(f, embedder, store, chunk_size, chunk_overlap)
            results.append(result)
    return results


def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"
