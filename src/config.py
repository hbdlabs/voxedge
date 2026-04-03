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
    corpus_dir: str = "/data/corpus"
    qdrant_dir: str = "/data/qdrant"
    max_tokens: int = 100
    local_language: str = "Norwegian"
    cache_dir: str = ""  # If set, FastEmbed/reranker models cached here
    host: str = "0.0.0.0"
    port: int = 8080
    n_ctx: int = 4096
    n_threads: int = 4
    mode: str = "full"  # "full" = RAG + chat + translate, "chat" = chat + translate only
    api_key: str = ""  # If set, all requests require Bearer token. Empty = no auth.

    model_config = {"env_prefix": "EDGE_"}


settings = Settings()
