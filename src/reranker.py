from fastembed.rerank.cross_encoder import TextCrossEncoder


class Reranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self._model = TextCrossEncoder(model_name=model_name)

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
