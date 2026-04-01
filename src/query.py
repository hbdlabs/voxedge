from dataclasses import dataclass, field

from langdetect import detect

from src.embedder import Embedder
from src.generator import Generator
from src.store import VectorStore


@dataclass
class QueryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    language: str = "unknown"


def query_brain(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    generator: Generator,
    top_k: int = 5,
    score_threshold: float = 0.5,
    max_tokens: int = 512,
) -> QueryResult:
    """Embed question, retrieve context, generate answer."""
    language = _detect_language(question)

    query_vector = embedder.embed([question])[0]

    results = store.query(
        vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    )

    chunks = [r["payload"]["text"] for r in results]
    sources = [
        {
            "file": r["payload"]["source_file"],
            "chunk": r["payload"].get("chunk_index", 0),
            "score": round(r["score"], 4),
        }
        for r in results
    ]

    answer = generator.generate(
        chunks=chunks,
        question=question,
        max_tokens=max_tokens,
    )

    return QueryResult(answer=answer, sources=sources, language=language)


def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"
