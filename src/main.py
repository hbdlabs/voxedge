import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import detect_language, settings
from src.embedder import Embedder
from src.generator import Generator
from src.profiles import get_profile
from src.ingest import ingest_file
from src.query import query_brain
from src.reranker import Reranker
from src.store import VectorStore


class QueryRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    message: str
    system: str = ""


class TranslateRequest(BaseModel):
    text: str
    source: str = ""
    target: str = ""


_start_time: float = 0.0


def create_app(
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    generator: Generator | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    """Create the FastAPI app. Accepts optional pre-built components for testing."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _start_time
        _start_time = time.time()

        if app.state.generator is None:
            profile = get_profile(settings.model_profile)
            app.state.generator = Generator(
                model_path=settings.model_path,
                profile=profile,
                n_ctx=settings.n_ctx,
                n_threads=settings.n_threads,
            )

        if settings.mode == "full":
            if app.state.embedder is None:
                app.state.embedder = Embedder(model_name=settings.embedding_model, cache_dir=settings.cache_dir or None)
            if app.state.store is None:
                app.state.store = VectorStore(
                    path=settings.qdrant_dir, vector_size=384
                )
            if app.state.reranker is None:
                app.state.reranker = Reranker(model_name=settings.reranker_model, cache_dir=settings.cache_dir or None)

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

        if app.state.store:
            app.state.store.flush()
            app.state.store.close()

    app = FastAPI(title="VoxEdge", lifespan=lifespan)
    app.state.embedder = embedder
    app.state.store = store
    app.state.generator = generator
    app.state.reranker = reranker

    if settings.api_key:
        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {settings.api_key}":
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
            return await call_next(request)

    def _require_rag():
        if settings.mode != "full":
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="RAG endpoints not available in chat mode. Set EDGE_MODE=full to enable.")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "mode": settings.mode,
            "model_loaded": app.state.generator is not None,
            "corpus_chunks": app.state.store.count() if app.state.store else 0,
            "uptime_seconds": int(time.time() - _start_time),
        }

    @app.get("/info")
    def info():
        result = {
            "mode": settings.mode,
            "model_profile": settings.model_profile,
            "models": {
                "llm": settings.model_path,
                "llm_context": settings.n_ctx,
                "llm_threads": settings.n_threads,
            },
            "config": {
                "local_language": settings.local_language,
                "max_tokens": settings.max_tokens,
            },
        }
        if settings.mode == "full" and app.state.store:
            store_info = app.state.store.info()
            docs = app.state.store.list_documents()
            result["models"]["embedding"] = settings.embedding_model
            result["models"]["reranker"] = settings.reranker_model
            result["config"]["chunk_size"] = settings.chunk_size
            result["config"]["chunk_overlap"] = settings.chunk_overlap
            result["config"]["top_k"] = settings.top_k
            result["config"]["score_threshold"] = settings.score_threshold
            result["store"] = store_info
            result["documents"] = docs
        return result

    @app.get("/corpus")
    def corpus():
        _require_rag()
        docs = app.state.store.list_documents()
        return {
            "documents": docs,
            "total_chunks": app.state.store.count(),
        }

    @app.delete("/corpus/{filename}")
    def delete_document(filename: str):
        _require_rag()
        deleted = app.state.store.delete_by_source(filename)
        return {
            "deleted": filename,
            "chunks_removed": deleted,
            "total_chunks": app.state.store.count(),
        }

    @app.post("/query")
    def query(req: QueryRequest):
        _require_rag()
        result = query_brain(
            question=req.question,
            embedder=app.state.embedder,
            store=app.state.store,
            generator=app.state.generator,
            reranker=app.state.reranker,
            top_k=settings.top_k,
            score_threshold=settings.score_threshold,
            max_tokens=settings.max_tokens,
        )
        return {
            "answer": result.answer,
            "sources": result.sources,
            "language": result.language,
        }

    @app.post("/chat")
    def chat(req: ChatRequest):
        response = app.state.generator.chat(
            message=req.message,
            system=req.system,
            max_tokens=settings.max_tokens,
        )
        return {"response": response}

    @app.post("/translate")
    def translate(req: TranslateRequest):
        local = settings.local_language
        # Auto-detect: if text looks like English, translate to local. Otherwise to English.
        if req.source:
            source = req.source
        else:
            detected = detect_language(req.text)
            source = "English" if detected == "en" else local
        target = req.target or ("English" if source != "English" else local)
        translation = app.state.generator.translate(
            text=req.text,
            source_lang=source,
            target_lang=target,
            max_tokens=min(len(req.text.split()) * 3, 200),
        )
        return {
            "translation": translation,
            "source": source,
            "target": target,
        }

    @app.post("/ingest")
    async def ingest(files: list[UploadFile]):
        _require_rag()
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
