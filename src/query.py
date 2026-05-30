import logging
from dataclasses import dataclass, field

from src.config import detect_language
from src.embedder import Embedder
from src.generator import Generator
from src.reranker import Reranker
from src.store import VectorStore

logger = logging.getLogger(__name__)

# langdetect returns ISO codes; the translation prompt wants language names.
_LANG_NAMES = {
    "en": "English", "vi": "Vietnamese", "es": "Spanish", "no": "Norwegian",
    "fr": "French", "de": "German", "pt": "Portuguese", "it": "Italian",
    "nl": "Dutch", "sw": "Swahili", "id": "Indonesian", "tl": "Tagalog",
    "ar": "Arabic", "hi": "Hindi", "th": "Thai", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh-cn": "Chinese", "zh-tw": "Chinese",
}


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
    translate_queries: bool = False,
    corpus_language: str = "English",
) -> QueryResult:
    """Embed question, retrieve candidates, rerank, generate answer.

    The detected language drives the response's `language` field and, when
    `translate_queries` is on, whether the query is translated into
    `corpus_language` for retrieval (see Settings.translate_queries).
    """
    logger.info("Query received: question_length=%d", len(question))
    language = detect_language(question)

    # Cross-lingual: retrieve in the corpus language, but answer from the user's
    # original question (so the reply stays in their language). Only translate
    # when we can name the detected language and it differs from the corpus's.
    retrieval_question = question
    if translate_queries:
        detected_name = _LANG_NAMES.get(language, "")
        if detected_name and detected_name.lower() != corpus_language.lower():
            translated = generator.translate(
                text=question,
                source_lang=detected_name,
                target_lang=corpus_language,
                max_tokens=min(max(len(question.split()) * 4, 16), 128),
            ).strip()
            if translated:
                retrieval_question = translated
                logger.info(
                    "Translated query for retrieval (%s->%s): %r -> %r",
                    detected_name, corpus_language, question, retrieval_question,
                )

    query_vector = embedder.embed([retrieval_question])[0]

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

    # Rerank to pick the best chunks (in the corpus language, like the embedding)
    if reranker:
        results = reranker.rerank(query=retrieval_question, chunks=results, top_k=top_k)
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


