import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from src.config import settings
from src.embedder import Embedder
from src.generator import Generator
from src.ingest import ingest_file
from src.query import query_brain
from src.store import VectorStore


class QueryRequest(BaseModel):
    question: str


_start_time: float = 0.0


def create_app(
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    generator: Generator | None = None,
) -> FastAPI:
    """Create the FastAPI app. Accepts optional pre-built components for testing."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _start_time
        _start_time = time.time()

        if app.state.embedder is None:
            app.state.embedder = Embedder(model_name=settings.embedding_model)
        if app.state.store is None:
            app.state.store = VectorStore(
                path=settings.qdrant_dir, vector_size=384
            )
        if app.state.generator is None:
            app.state.generator = Generator(
                model_path=settings.model_path,
                n_ctx=settings.n_ctx,
                n_threads=settings.n_threads,
            )

        # Ingest baked-in corpus (only new files)
        corpus_dir = Path(settings.corpus_dir)
        if corpus_dir.exists():
            existing_docs = {d["source_file"] for d in app.state.store.list_documents()}
            supported = {".txt", ".md", ".pdf", ".docx", ".doc", ".pptx", ".xlsx"}
            for f in sorted(corpus_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in supported and f.name not in existing_docs:
                    ingest_file(
                        f, app.state.embedder, app.state.store,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                    )

        yield

        app.state.store.flush()
        app.state.store.close()

    app = FastAPI(title="Edge RAG Brain", lifespan=lifespan)
    app.state.embedder = embedder
    app.state.store = store
    app.state.generator = generator

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": app.state.generator is not None,
            "corpus_chunks": app.state.store.count(),
            "uptime_seconds": int(time.time() - _start_time),
        }

    @app.get("/corpus")
    def corpus():
        docs = app.state.store.list_documents()
        return {
            "documents": docs,
            "total_chunks": app.state.store.count(),
        }

    @app.post("/query")
    def query(req: QueryRequest):
        result = query_brain(
            question=req.question,
            embedder=app.state.embedder,
            store=app.state.store,
            generator=app.state.generator,
            top_k=settings.top_k,
            score_threshold=settings.score_threshold,
            max_tokens=settings.max_tokens,
        )
        return {
            "answer": result.answer,
            "sources": result.sources,
            "language": result.language,
        }

    @app.post("/ingest")
    async def ingest(files: list[UploadFile]):
        results = []
        for upload in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.filename or "upload").suffix) as tmp:
                content = await upload.read()
                tmp.write(content)
                tmp_path = Path(tmp.name)
            result = ingest_file(
                path=tmp_path,
                embedder=app.state.embedder,
                store=app.state.store,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                source_name=upload.filename,
            )
            tmp_path.unlink()
            results.append({
                "file": upload.filename or tmp_path.name,
                "chunks": result.chunks,
                "language": result.language,
            })
        return {"ingested": results}

    return app


app = create_app()
