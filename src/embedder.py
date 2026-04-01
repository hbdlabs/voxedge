from fastembed import TextEmbedding


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into 384-dim vectors."""
        return [vec.tolist() for vec in self._model.embed(texts)]
