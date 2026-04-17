from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.profiles import ModelProfile


class Reranker:
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        cuda: bool = False,
    ):
        kwargs: dict = {"model_name": model_name, "cache_dir": cache_dir}
        if cuda:
            kwargs["cuda"] = True
        self._model = TextCrossEncoder(**kwargs)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """Rerank retrieved chunks by cross-encoder relevance score.

        Args:
            query: the user's question
            chunks: list of dicts with "payload" and "score" from vector search
            top_k: number of best chunks to return

        Returns:
            top_k chunks sorted by reranker score, with updated scores
        """
        if not chunks:
            return []

        texts = [c["payload"]["text"] for c in chunks]
        scores = list(self._model.rerank(query, texts))

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]

    def active_providers(self) -> list[str]:
        """Return the ONNX Runtime providers actually in use by this session."""
        sess = getattr(self._model, "model", None)
        if sess is None or not hasattr(sess, "get_providers"):
            return []
        return sess.get_providers()


_SUPPORTED_DEVICES = {"cpu", "cuda"}


def build(
    profile: ModelProfile,
    model_name: str,
    cache_dir: str | None = None,
) -> Reranker:
    """Construct a Reranker according to the profile's reranker_device."""
    if profile.reranker_device not in _SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported reranker_device '{profile.reranker_device}' "
            f"for profile '{profile.name}'. Supported: {sorted(_SUPPORTED_DEVICES)}"
        )
    return Reranker(
        model_name=model_name,
        cache_dir=cache_dir,
        cuda=(profile.reranker_device == "cuda"),
    )
