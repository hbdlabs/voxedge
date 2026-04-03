from fastembed import TextEmbedding


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", cache_dir: str | None = None):
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into 384-dim vectors."""
        return [vec.tolist() for vec in self._model.embed(texts)]
