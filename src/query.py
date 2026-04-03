import logging
from dataclasses import dataclass, field

from src.config import detect_language
from src.embedder import Embedder
from src.generator import Generator
from src.reranker import Reranker
from src.store import VectorStore

logger = logging.getLogger(__name__)


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
    reranker: Reranker | None = None,
    top_k: int = 3,
    retrieve_k: int = 20,
    score_threshold: float = 0.3,
    max_tokens: int = 512,
) -> QueryResult:
    """Embed question, retrieve candidates, rerank, generate answer."""
    logger.info("Query received: question_length=%d", len(question))
    language = detect_language(question)

    query_vector = embedder.embed([question])[0]

    # Retrieve more candidates with a loose threshold
    results = store.query(
        vector=query_vector,
        limit=retrieve_k,
        score_threshold=score_threshold,
    )

    if not results:
        logger.info("No chunks found for query")
        return QueryResult(
            answer="I don't have information about that.",
            sources=[],
            language=language,
        )

    # Rerank to pick the best chunks
    if reranker:
        results = reranker.rerank(query=question, chunks=results, top_k=top_k)
    else:
        results = results[:top_k]

    chunks = [r["payload"]["text"] for r in results]
    sources = [
        {
            "file": r["payload"]["source_file"],
            "chunk": r["payload"].get("chunk_index", 0),
            "score": round(r.get("rerank_score", r["score"]), 4),
        }
        for r in results
    ]

    logger.info("Found %d chunks after rerank", len(chunks))
    answer = generator.generate(
        chunks=chunks,
        question=question,
        max_tokens=max_tokens,
    )
    logger.info("Answer generated: answer_length=%d", len(answer))

    return QueryResult(answer=answer, sources=sources, language=language)


