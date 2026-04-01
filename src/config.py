from langdetect import detect
from pydantic_settings import BaseSettings


def detect_language(text: str) -> str:
    """Detect language of text, returning 'unknown' on failure."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


class Settings(BaseSettings):
    model_path: str = "/data/models/tiny-aya-q4.gguf"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    score_threshold: float = 0.3
    corpus_dir: str = "/data/corpus"
    qdrant_dir: str = "/data/qdrant"
    max_tokens: int = 200
    host: str = "0.0.0.0"
    port: int = 8080
    n_ctx: int = 4096
    n_threads: int = 4

    model_config = {"env_prefix": "EDGE_"}


settings = Settings()
