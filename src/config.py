from typing import Literal

from langdetect import detect
from pydantic_settings import BaseSettings


def detect_language(text: str) -> str:
    """Detect language of text, returning 'unknown' on failure."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


class Settings(BaseSettings):
    model_path: str = "/data/models/tiny-aya-global-q4_k_m.gguf"
    model_profile: str = "aya"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    chunk_size: int = 250
    chunk_overlap: int = 30
    top_k: int = 5
    score_threshold: float = 0.3

    # --- Cross-lingual retrieval ---
    # When the corpus is in one language (e.g. English) but users ask in others,
    # the multilingual embedder and reranker can mis-route the query: a Vietnamese
    # question can retrieve the *wrong* English document, because cross-lingual
    # alignment is weak for lower-resource languages (verified — a Vietnamese
    # "what is tuberculosis?" retrieved the cholera doc).
    #
    # With `translate_queries` on, any query whose detected language differs from
    # `corpus_language` is translated INTO corpus_language *before* embedding,
    # vector search, and reranking — turning a weak cross-lingual match into a
    # strong monolingual one (this recovered both the Vietnamese and Spanish
    # queries that otherwise failed). The answer is still generated from the
    # user's ORIGINAL question, so it comes back in their language.
    #
    # Cost: one extra short LLM call per translated query. Off by default. Best
    # for a single-language corpus; leave off if the corpus is itself multilingual
    # (then "translate to which language?" has no single answer).
    translate_queries: bool = False
    corpus_language: str = "English"  # language the documents/embeddings are in; queries in other languages are translated to this before retrieval (only when translate_queries=True)

    corpus_dir: str = "/data/corpus"
    qdrant_dir: str = "/data/qdrant"
    max_tokens: int = 100
    local_language: str = "Norwegian"
    cache_dir: str = ""  # If set, FastEmbed/reranker models cached here
    host: str = "0.0.0.0"
    port: int = 8080
    n_ctx: int = 4096
    n_threads: int = 4
    mode: Literal["full", "chat"] = "full"  # "full" = RAG + chat + translate, "chat" = chat + translate only
    api_key: str = ""  # If set, all requests require Bearer token. Empty = no auth.

    model_config = {"env_prefix": "EDGE_"}


settings = Settings()
