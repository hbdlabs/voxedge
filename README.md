# Edge RAG Brain

## What is this

Multilingual AI that runs anywhere, without internet.

Load it with documents, and people can ask questions in their own language -- getting answers grounded in those documents, with source references. It translates between languages, explains concepts, and makes information accessible to people who would otherwise not have access to it.

This is made possible by a new generation of small but capable multilingual models that can run on consumer hardware. A 3B parameter model on a Raspberry Pi can now do what required cloud APIs and expensive infrastructure just a year ago. Combined with local vector search and document parsing, it becomes a self-contained knowledge system that works offline.

**Where it can be used:**

- **Schools and universities** -- students access educational material, get explanations, and translate content in their own language on a shared classroom device
- **Language learning stations** -- learners practice translation and conversation with an AI powered by local documents and phrasebooks
- **Tourist information kiosks** -- visitors ask about local attractions, transportation, and services in their own language
- **Field offices and NGOs** -- workers access operational manuals, training material, and policy documents in the field without connectivity
- **Community centers and libraries** -- a shared knowledge terminal where residents ask about government services, civic rights, or local information in their language
- **Employee onboarding** -- new hires query company handbooks and policies in their preferred language

Adding content is straightforward: upload a PDF, Word doc, or text file via the API and it becomes searchable immediately. No retraining, no restart. Documents can be added, replaced, and removed while the system is running.

### How it works

1. **You load documents** (PDF, text, Word) into the system -- either baked into the container image or uploaded via API at runtime.
2. **Documents are split into chunks**, converted to numerical vectors by a multilingual embedding model, and stored in a local vector database.
3. **When someone asks a question**, the question is also converted to a vector, the most relevant chunks are retrieved, a cross-encoder reranker selects the best matches, and a local language model generates an answer using only those chunks as context.
4. **The answer comes back with source references** so the user can verify where the information came from.

### Modes

The system runs in two modes:

**Full mode** (default) -- document Q&A with retrieval, plus chat and translation. Loads the LLM, embedding model, vector store, and reranker. All endpoints available.

**Chat mode** -- translation and conversation only, no document retrieval. Loads only the LLM. Faster startup, less RAM (~2.5 GB vs 4+ GB). Set `EDGE_MODE=chat`.

### Reranker options

The reranker is a precision filter that picks the best chunks from a larger set of candidates. Two models are available:

**English reranker** (80 MB, default) -- `Xenova/ms-marco-MiniLM-L-6-v2`. Fast and lightweight. Works well when your documents are in English. This is the default because it keeps the footprint small and handles the most common use case.

**Multilingual reranker** (1.1 GB) -- `jinaai/jina-reranker-v2-base-multilingual`. Switch to this when your documents are in non-English languages. The English reranker can pick the wrong chunk when multiple similar chunks compete in a non-English document. Set `EDGE_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual`.

The embedding model and the LLM are multilingual regardless of which reranker you choose. The reranker is the only component where language matters for the choice.

### What makes it different

- **Fully offline** -- runs on a Raspberry Pi, laptop, or kiosk with no internet
- **Multilingual** -- questions and documents can be in different languages (50+ for retrieval, 70+ for generation). Ask in Vietnamese, get answers from English documents.
- **Self-contained** -- single Docker container with all models, vector database, document parser, and API server
- **Grounded** -- answers cite their sources and the system refuses to answer when it has no relevant information
- **Configurable** -- swap rerankers, adjust chunk size, tune retrieval thresholds, switch between full and chat mode

### About Tiny Aya

The language model is [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B parameters) by [Cohere Labs](https://cohere.com/research/aya). It supports 70+ languages and is part of the Aya open-science initiative. We run the Q4_K_M quantized version (2.1 GB) on CPU via llama-cpp-python.

It comes in regional variants (global, earth, water, fire). We use **global** for balanced performance across all languages. Strong at translation and generation, weaker at math and reasoning — well suited for RAG where answers come from retrieved context.

Licensed CC-BY-NC (non-commercial). Commercial use requires contacting Cohere.

### Stack

| Component | Library | Role |
|---|---|---|
| Document parsing | LiteParse (Bun) | Extract text from PDF, DOCX, images |
| Text embedding | FastEmbed (ONNX) | Convert text to 384-dim multilingual vectors |
| Vector storage | Qdrant Edge | Store and search vectors locally on disk |
| Reranking | FastEmbed Cross-Encoder (ONNX) | Precision-filter retrieved chunks |
| Generation | llama-cpp-python + Tiny Aya 3.35B (GGUF) | Generate multilingual answers on CPU |
| API | FastAPI | HTTP endpoints for query, ingest, chat, translate, info, health |

For detailed component descriptions and data flow diagrams, see [docs/architecture.md](docs/architecture.md).

Sample documents for testing are in `examples/corpus/`.

## Requirements

- Python 3.11+
- Bun (for LiteParse document parsing, not needed in chat mode)
- ~4 GB RAM minimum (full mode), ~2.5 GB (chat mode)
- ~4 GB disk (model weights + vector storage)

## Local Development Setup

### 1. Create virtual environment

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate

# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

This installs all runtime dependencies (FastEmbed, qdrant-edge-py, llama-cpp-python, LiteParse, FastAPI, etc.) plus dev dependencies (pytest, httpx).

llama-cpp-python builds from source and requires `cmake` and a C++ compiler:
- **macOS**: Xcode Command Line Tools (`xcode-select --install`)
- **Linux**: `apt-get install build-essential cmake`
- **Windows**: Visual Studio Build Tools with "C++ build tools" workload, plus CMake (`winget install cmake`)

### 3. Install Bun

```bash
# macOS / Linux
curl -fsSL https://bun.sh/install | bash

# Windows
powershell -c "irm bun.sh/install.ps1 | iex"
```

Bun is needed for LiteParse (PDF/DOCX parsing). If you only use `.txt` and `.md` files or run in chat mode, Bun is not required.

### 4. Download the GGUF model

```bash
mkdir -p ~/models
curl -L -o ~/models/tiny-aya-global-q4_k_m.gguf \
  "https://huggingface.co/CohereLabs/tiny-aya-global-GGUF/resolve/main/tiny-aya-global-q4_k_m.gguf"
```

This is ~2.1 GB. The model is stored in `~/models/` so it can be shared across projects and easily managed.

### 5. Run tests

```bash
python -m pytest tests/ -v
```

Unit and integration tests run without the GGUF model (they mock the generator). The end-to-end test (`test_e2e.py`) requires the model and is skipped automatically if the file is not found.

### 6. Start the server

```bash
# Full mode (RAG + chat + translate)
EDGE_MODEL_PATH=~/models/tiny-aya-global-q4_k_m.gguf \
EDGE_CORPUS_DIR=./data/corpus \
EDGE_QDRANT_DIR=./data/qdrant \
uvicorn src.main:app --host 127.0.0.1 --port 8080

# Chat mode (chat + translate only)
EDGE_MODE=chat \
EDGE_MODEL_PATH=~/models/tiny-aya-global-q4_k_m.gguf \
uvicorn src.main:app --host 127.0.0.1 --port 8080
```

Startup takes 15-30 seconds in full mode (loading all models + indexing corpus), 5-10 seconds in chat mode (loading LLM only).

## Docker Deployment

### Build

```bash
# Optionally place documents in data/corpus/ before building
cp your_documents/*.pdf data/corpus/

# Build the image (~10 min first time, downloads 2.1 GB model)
docker build -t edge-brain .
```

### Run

```bash
# Full mode
docker run -d \
  --name edge-brain \
  -p 8080:8080 \
  -v edge-brain-data:/data/qdrant \
  edge-brain

# Chat mode
docker run -d \
  --name edge-brain \
  -p 8080:8080 \
  -e EDGE_MODE=chat \
  edge-brain
```

The volume mount (`-v edge-brain-data:/data/qdrant`) persists the vector index across container restarts. Not needed in chat mode.

### Verify

```bash
curl http://localhost:8080/health
```

## Fly.io Deployment

Two fly configs are provided. Performance CPUs are required — shared CPUs cannot run the 3.35B LLM within HTTP timeouts.

**Full mode** (performance 4-core, 8 GB, persistent volume for vector storage):
```bash
fly apps create edge-rag-brain
fly volumes create edge_brain_data --region arn --size 5
fly deploy --config fly.full.toml --remote-only
```

**Chat mode** (performance 2-core, 8 GB, no volume needed):
```bash
fly apps create edge-rag-brain
fly deploy --config fly.chat.toml --remote-only
```

First startup takes 3-5 minutes (downloading embedding and reranker models). Subsequent starts are faster if using a persistent volume with `HF_HOME` set (models are cached).

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-rag-kiosk
spec:
  replicas: 1
  selector:
    matchLabels:
      app: edge-rag-kiosk
  template:
    metadata:
      labels:
        app: edge-rag-kiosk
    spec:
      containers:
      - name: brain
        image: edge-brain:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "4Gi"
            cpu: "4"
          limits:
            memory: "6Gi"
        volumeMounts:
        - name: qdrant-storage
          mountPath: /data/qdrant
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
      volumes:
      - name: qdrant-storage
        persistentVolumeClaim:
          claimName: edge-brain-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: edge-rag-kiosk
spec:
  ports:
  - port: 8080
  selector:
    app: edge-rag-kiosk
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: edge-brain-pvc
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 1Gi
```

Set `initialDelaySeconds: 60` on probes to allow time for model loading and corpus ingestion.

## Raspberry Pi Deployment

The Docker image is built for `arm64`, compatible with Raspberry Pi 4/5 (64-bit OS).

**Install Docker on the Pi** (one-time):
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in
```

**Pull from registry:**
```bash
docker pull ghcr.io/YOUR_USER/edge-brain:latest
docker run -d -p 8080:8080 -v edge-data:/data/qdrant ghcr.io/YOUR_USER/edge-brain:latest
```

**Transfer image file (air-gapped):**
```bash
# On build machine:
docker save edge-brain | gzip > edge-brain.tar.gz
# Transfer to Pi via USB stick or scp, then:
docker load < edge-brain.tar.gz
docker run -d -p 8080:8080 -v edge-data:/data/qdrant edge-brain
```

**Hardware notes:**
- Raspberry Pi 5 (8 GB): runs well, queries take 5-15 seconds
- Raspberry Pi 4 (4 GB): functional but tight on RAM, use swap
- Raspberry Pi 4 (2 GB): not recommended

## API Reference

### POST /query

Ask a question. Returns a grounded answer with source references. Only available in full mode.

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I prevent malaria?"}'
```

```json
{
  "answer": "Prevention includes sleeping under insecticide-treated bed nets, using mosquito repellent, and taking antimalarial medication.",
  "sources": [{"file": "health_basics.txt", "chunk": 0, "score": 7.4985}],
  "language": "en"
}
```

Questions can be asked in any supported language. The system retrieves relevant chunks regardless of the document's language and generates an answer in the question's language.

If no relevant context is found:
```json
{"answer": "I don't have information about that.", "sources": [], "language": "en"}
```

### POST /ingest

Upload documents to index at runtime. Only available in full mode.

```bash
curl -X POST http://localhost:8080/ingest \
  -F "files=@guide.pdf" \
  -F "files=@manual.txt"
```

**Supported formats:**
- `.txt`, `.md` — read directly, best quality
- `.pdf` — parsed by LiteParse with spatial layout reconstruction and built-in OCR (Tesseract). Text-based PDFs work well. Scanned PDFs depend on image quality.
- `.docx`, `.doc`, `.pptx`, `.xlsx` — LiteParse converts to PDF via LibreOffice (included in the Docker image), then parses. For local development, install LibreOffice separately.

Re-ingesting the same file overwrites existing chunks.

### POST /chat

Direct chat with the language model, no RAG retrieval. Available in both modes.

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hva betyr feriepenger?", "system": "Svar kort på norsk."}'
```

```json
{"response": "Feriepenger er en type lønn som ansatte får under sin ferie..."}
```

The `system` field is optional. Without it, the model responds in whatever language the message is in.

### POST /translate

Translate text between languages. Available in both modes.

Auto-detects direction based on `EDGE_LOCAL_LANGUAGE` (default: Norwegian). English input translates to the local language, non-English input translates to English.

```bash
# Auto direction
curl -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "The vaccine is free at all health centers."}'

# Explicit
curl -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Xin chào", "source": "Vietnamese", "target": "Norwegian"}'
```

```json
{
  "translation": "Vaksinen er gratis på alle helsestasjoner.",
  "source": "English",
  "target": "Norwegian"
}
```

### GET /info

System information: loaded models, configuration, vector store stats, and indexed documents.

```bash
curl http://localhost:8080/info
```

In full mode:
```json
{
  "mode": "full",
  "models": {
    "llm": "/data/models/tiny-aya-global-q4_k_m.gguf",
    "llm_context": 4096,
    "llm_threads": 4,
    "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "reranker": "jinaai/jina-reranker-v2-base-multilingual"
  },
  "config": {
    "local_language": "Norwegian",
    "max_tokens": 100,
    "chunk_size": 250,
    "chunk_overlap": 30,
    "top_k": 5,
    "score_threshold": 0.3
  },
  "store": {
    "segments": 1,
    "points": 78,
    "indexed_vectors": 78,
    "vector_size": 384
  },
  "documents": [
    {"source_file": "Ferie.pdf", "chunks": 65, "language": "no", "ingested_at": "..."}
  ]
}
```

In chat mode, the store, documents, and RAG config fields are omitted.

### GET /health

```json
{"status": "ok", "mode": "full", "model_loaded": true, "corpus_chunks": 78, "uptime_seconds": 3600}
```

### GET /corpus

List all indexed documents. Only available in full mode.

### DELETE /corpus/{filename}

Remove a document and all its vectors from the index. Only available in full mode.

```bash
curl -X DELETE http://localhost:8080/corpus/Ferie.pdf
```

## Configuration

All settings are configurable via environment variables with the `EDGE_` prefix:

| Setting | Default | Purpose |
|---|---|---|
| `EDGE_MODE` | `full` | `full` = RAG + chat + translate, `chat` = chat + translate only |
| `EDGE_API_KEY` | *(empty)* | If set, all requests require `Authorization: Bearer <key>`. Empty = no auth. |
| `EDGE_MODEL_PATH` | `/data/models/tiny-aya-global-q4_k_m.gguf` | GGUF model location |
| `EDGE_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `EDGE_RERANKER_MODEL` | `jinaai/jina-reranker-v2-base-multilingual` | Reranker model |
| `EDGE_LOCAL_LANGUAGE` | `Norwegian` | Default language for auto-translate direction |
| `EDGE_CHUNK_SIZE` | `250` | Characters per chunk |
| `EDGE_CHUNK_OVERLAP` | `30` | Overlap between chunks |
| `EDGE_TOP_K` | `5` | Chunks passed to LLM after reranking |
| `EDGE_SCORE_THRESHOLD` | `0.3` | Minimum similarity for initial retrieval |
| `EDGE_MAX_TOKENS` | `100` | Max generation tokens for query/chat |
| `EDGE_N_THREADS` | `4` | CPU threads for LLM |
| `EDGE_N_CTX` | `4096` | LLM context window |
| `EDGE_CORPUS_DIR` | `/data/corpus` | Baked-in documents directory |
| `EDGE_QDRANT_DIR` | `/data/qdrant` | Vector storage path |

### Deployment modes

**Full mode** (`EDGE_MODE=full`, default) — loads all components. All endpoints available.

**Chat mode** (`EDGE_MODE=chat`) — loads only the LLM. Only `/chat`, `/translate`, `/info`, `/health` available. Faster startup, less RAM.

### Choosing a reranker

| Model | Size | Speed | Best for |
|---|---|---|---|
| `Xenova/ms-marco-MiniLM-L-6-v2` | 80 MB | ~5ms/chunk | English-only corpus |
| `jinaai/jina-reranker-v2-base-multilingual` | 1.1 GB | ~50ms/chunk | Non-English or mixed corpus |

The English reranker struggles with non-English content — it may score the wrong chunk higher when multiple similar chunks compete in a non-English document.

### Tuning guidelines

**EDGE_CHUNK_SIZE**: Smaller chunks (200-300) give more precise retrieval but may split context. Larger chunks (500-800) preserve more context but dilute relevance.

**EDGE_TOP_K**: More chunks = more context for the LLM but also more noise. 5 is a good default.

**EDGE_SCORE_THRESHOLD**: Lower values (0.2) cast a wider net. Higher values (0.5) are stricter but may miss chunks.

**EDGE_MAX_TOKENS**: 100 tokens produces 1-3 sentence answers. Increase for longer explanations.

**EDGE_N_THREADS**: Match your CPU core count.

## Operations

### Managing the corpus

```bash
# Add documents
curl -X POST http://localhost:8080/ingest -F "files=@new_guide.pdf"

# List indexed documents
curl http://localhost:8080/corpus

# Remove a document
curl -X DELETE http://localhost:8080/corpus/new_guide.pdf

# Replace a document (re-ingesting same filename overwrites)
curl -X POST http://localhost:8080/ingest -F "files=@updated_guide.pdf"

# Full system info with store stats
curl http://localhost:8080/info
```

No restart needed. Changes take effect immediately.

### Monitoring

```bash
# Health check
curl http://localhost:8080/health

# Full system status
curl http://localhost:8080/info

# Watch corpus
watch -n 10 'curl -s http://localhost:8080/corpus | python3 -m json.tool'
```

Key indicators:
- `model_loaded: false` — the LLM failed to load. Check logs for memory issues.
- `corpus_chunks: 0` — no documents indexed. Check `EDGE_CORPUS_DIR` path and file formats.

### Clearing the index

```bash
# Docker: remove the volume
docker stop edge-brain && docker rm edge-brain
docker volume rm edge-brain-data
docker run -d --name edge-brain -p 8080:8080 -v edge-brain-data:/data/qdrant edge-brain

# Local: delete the qdrant directory
rm -rf data/qdrant
```

### API authentication

By default, the API has no authentication (suitable for local kiosks and air-gapped devices). For network deployments, set an API key:

```bash
# Docker
docker run -d -p 8080:8080 -e EDGE_API_KEY=your-secret-key edge-brain

# Fly.io: set as a secret
fly secrets set EDGE_API_KEY=your-secret-key -a edge-rag-brain
```

All requests (except `/health`) must include the key:

```bash
curl -H "Authorization: Bearer your-secret-key" \
  http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I prevent malaria?"}'
```

Requests without a valid key receive a 401 response. The `/health` endpoint is always accessible (for load balancer and orchestrator health checks).

### Remote access with ngrok

```bash
curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok-v3-stable-linux-arm64.tgz | tar xz
./ngrok http 8080
```

### Backup and restore

```bash
# Backup
docker cp edge-brain:/data/qdrant ./qdrant-backup

# Restore
docker run -d --name edge-brain-new -p 8080:8080 -v $(pwd)/qdrant-backup:/data/qdrant edge-brain
```

## Retrieval Quality Guide

The quality of answers depends on how well the system retrieves the right chunks from your documents.

### How LiteParse handles documents

LiteParse preserves spatial structure rather than converting it away. Instead of transforming a PDF into markdown (which loses layout information), LiteParse extracts text with its original positioning — each text item comes with x/y coordinates, font size, and a confidence score.

We use this spatial data in `parser.py` to reconstruct clean text. The parser groups text items into lines by Y position, identifies garbled regions (OCR artifacts from screenshots, tiny mixed font sizes) and filters them out, and strips headers and footers by their position on the page.

```bash
# Inspect spatial output
bunx @llamaindex/liteparse parse your_document.pdf --format json | python3 -m json.tool

# Compare with plain text extraction
bunx @llamaindex/liteparse parse your_document.pdf --format text
```

### Know your documents

Before ingesting, review what gets extracted:

- **Garbled text from screenshots or tables** — the spatial parser filters most of this, but complex layouts may still produce noise
- **Important facts buried in tables** — consider adding a plain text summary alongside the PDF
- **Contradictory information** — if the same topic appears with different numbers, the model may pick the wrong one
- **Information split across pages** — may get split into different chunks

### How retrieval can fail

**1. Embedding search** — the question and relevant chunk use very different vocabulary. Diagnosis: system says "I don't have information" for a question you know is covered.

**2. Cross-encoder reranking** — a related but wrong chunk scores higher than the correct one. Diagnosis: wrong answer but correct source file.

**3. LLM generation** — right chunks retrieved but the 3B model misinterprets or conflates facts. Diagnosis: correct source and chunk, wrong answer.

### Improving retrieval

- **Phrasing matters.** Questions using the same vocabulary as source documents get better results.
- **Add plain text summaries** alongside PDFs with complex layouts.
- **Adjust chunk size.** 200 for fact-dense documents, 400-500 for narrative text.
- **Adjust score threshold.** Lower if too many misses, raise if too many wrong answers.
- **Test with known answers.** Compare system output against actual document content.

## Licensing

- **Tiny Aya Global** (CohereLabs): CC-BY-NC-4.0 (non-commercial)
- **Jina Reranker v2** (Jina AI): CC-BY-NC-4.0 (non-commercial)
- **FastEmbed models**: Apache 2.0
- **Qdrant Edge**: Apache 2.0
- **LiteParse**: MIT

The code in this repository is MIT licensed. When deploying with the default models, the CC-BY-NC restriction applies to the overall system. See [LICENSE](LICENSE) for details.
