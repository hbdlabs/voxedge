from pathlib import Path

from qdrant_edge import (
    CountRequest,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    FieldCondition,
    Filter,
    MatchValue,
    Point,
    QueryRequest,
    Query,
    ScrollRequest,
    UpdateOperation,
)


class VectorStore:
    def __init__(self, path: str, vector_size: int = 384):
        self._path = Path(path)
        self._vector_size = vector_size
        if self._path.exists() and any(self._path.iterdir()):
            try:
                self._shard = EdgeShard.load(str(self._path))
                return
            except Exception:
                pass
        self._path.mkdir(parents=True, exist_ok=True)
        config = EdgeConfig(
            vectors=EdgeVectorParams(
                size=vector_size,
                distance=Distance.Cosine,
            ),
        )
        self._shard = EdgeShard.create(str(self._path), config)

    def upsert(self, point_id: int, vector: list[float], payload: dict) -> None:
        self._shard.update(
            UpdateOperation.upsert_points([Point(point_id, vector, payload)])
        )

    def upsert_batch(self, points: list[tuple[int, list[float], dict]]) -> None:
        self._shard.update(
            UpdateOperation.upsert_points(
                [Point(pid, vec, pay) for pid, vec, pay in points]
            )
        )

    def query(self, vector: list[float], limit: int = 5, score_threshold: float = 0.0) -> list[dict]:
        results = self._shard.query(
            QueryRequest(
                query=Query.Nearest(vector),
                limit=limit,
                with_vector=False,
                with_payload=True,
            )
        )
        out = []
        for point in results:
            score = point.score if hasattr(point, "score") else 0.0
            if score >= score_threshold:
                out.append({
                    "id": point.id,
                    "score": score,
                    "payload": point.payload,
                })
        return out

    def count(self) -> int:
        return self._shard.count(CountRequest(exact=True))

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        offset = None
        while True:
            req = ScrollRequest(limit=100, offset=offset) if offset else ScrollRequest(limit=100)
            results, next_offset = self._shard.scroll(req)
            for point in results:
                payload = point.payload or {}
                source = payload.get("source_file", "unknown")
                if source not in docs:
                    docs[source] = {
                        "source_file": source,
                        "chunks": 0,
                        "language": payload.get("language", "unknown"),
                        "ingested_at": payload.get("ingested_at", ""),
                    }
                docs[source]["chunks"] += 1
            if next_offset is None:
                break
            offset = next_offset
        return list(docs.values())

    def delete_by_source(self, source_file: str) -> int:
        """Delete all points for a given source file. Returns count deleted."""
        count_before = self.count()
        self._shard.update(
            UpdateOperation.delete_points_by_filter(
                Filter(
                    must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
                )
            )
        )
        return count_before - self.count()

    def flush(self) -> None:
        self._shard.flush()

    def close(self) -> None:
        self._shard.close()
