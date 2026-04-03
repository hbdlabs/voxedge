import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.chunker import chunk_text
from src.config import detect_language
from src.embedder import Embedder
from src.parser import parse_file
from src.store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    file: str
    chunks: int
    language: str


def _point_id(source_file: str, chunk_index: int) -> int:
    """Generate a deterministic point ID from source file and chunk index."""
    key = f"{source_file}:{chunk_index}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def ingest_file(
    path: Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_name: str | None = None,
) -> IngestResult:
    """Parse, chunk, embed, and store a single document."""
    name = source_name or path.name
    text = parse_file(path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)

    if not chunks:
        logger.warning("No chunks produced for file: %s", name)
        return IngestResult(file=name, chunks=0, language="unknown")

    language = detect_language(chunks[0])
    vectors = embedder.embed(chunks)
    now = datetime.now(timezone.utc).isoformat()

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append((
            _point_id(name, i),
            vector,
            {
                "text": chunk,
                "source_file": name,
                "chunk_index": i,
                "language": language,
                "ingested_at": now,
            },
        ))

    store.upsert_batch(points)
    logger.info("Ingested %s: %d chunks, language=%s", name, len(chunks), language)
    return IngestResult(file=name, chunks=len(chunks), language=language)


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
