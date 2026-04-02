# Edge RAG Brain

## What is this

A knowledge kiosk in a box. Load it with documents -- health guides, farming manuals, government policies, employee handbooks, whatever -- and people can ask questions in their own language and get answers grounded in those documents.

It is built for places where there is no reliable internet: rural health posts, community centers, field offices, schools. Everything runs locally on a single device. No cloud APIs, no network calls, no data leaving the machine.

The system uses RAG (Retrieval-Augmented Generation), which means it does not just generate text from a language model -- it first searches the loaded documents for relevant passages, then generates an answer based only on what it found. This keeps answers factual and traceable to source documents.

### How it works

1. **You load documents** (PDF, text, Word) into the system -- either baked into the container image or uploaded via API at runtime.
2. **Documents are split into chunks**, converted to numerical vectors by a multilingual embedding model, and stored in a local vector database.
3. **When someone asks a question**, the question is also converted to a vector, the most relevant chunks are retrieved, a cross-encoder reranker selects the best matches, and a local language model generates an answer using only those chunks as context.
4. **The answer comes back with source references** so the user can verify where the information came from.

### What makes it different

- **Fully offline** -- runs on a Raspberry Pi, laptop, or kiosk with no internet
- **Multilingual** -- questions and documents can be in different languages (50+ supported for retrieval, 70+ for generation). Ask in Vietnamese, get answers from English documents.
- **Self-contained** -- single Docker container, ~3.2 GB, includes the language model, embedding model, reranker, vector database, document parser, and API server
- **Grounded** -- answers cite their sources and the system refuses to answer when it has no relevant information

### Stack

| Component | Library | Role |
|---|---|---|
| Document parsing | LiteParse (Bun) | Extract text from PDF, DOCX, images |
| Text embedding | FastEmbed (ONNX) | Convert text to 384-dim multilingual vectors |
| Vector storage | Qdrant Edge | Store and search vectors locally on disk |
| Reranking | FastEmbed Cross-Encoder (ONNX) | Precision-filter retrieved chunks |
| Generation | llama-cpp-python + Tiny Aya 3.35B (GGUF) | Generate multilingual answers on CPU |
| API | FastAPI | HTTP endpoints for query, ingest, health, corpus |

For detailed component descriptions and data flow diagrams, see [docs/architecture.md](docs/architecture.md).

## Requirements

- Python 3.11+
- Bun (for LiteParse document parsing)
- ~4 GB RAM minimum
- ~4 GB disk (model weights + vector storage)

## Local Development Setup

### 1. Create virtual environment

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

This installs all runtime dependencies (FastEmbed, qdrant-edge-py, llama-cpp-python, LiteParse, FastAPI, etc.) plus dev dependencies (pytest, httpx).

llama-cpp-python builds from source and requires `cmake` and a C++ compiler. On macOS these come with Xcode Command Line Tools. On Linux, install `build-essential cmake`.

### 3. Install Bun

```bash
curl -fsSL https://bun.sh/install | bash
```

Bun is needed for LiteParse (PDF/DOCX parsing). If you only use `.txt` and `.md` files, Bun is not required.

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
EDGE_MODEL_PATH=~/models/tiny-aya-global-q4_k_m.gguf \
EDGE_CORPUS_DIR=./data/corpus \
EDGE_QDRANT_DIR=./data/qdrant \
uvicorn src.main:app --host 127.0.0.1 --port 8080
```

Startup takes 15-30 seconds (loading embedding model, LLM, reranker, and indexing any baked-in corpus).

## Docker Deployment

### Build

```bash
# Place documents in data/corpus/ before building
cp your_documents/*.pdf data/corpus/
cp your_documents/*.txt data/corpus/

# Build the image (~10 min first time, downloads 2.1 GB model)
docker build -t edge-brain .
```

The image is ~3.2 GB. The GGUF model is downloaded during build and baked into the image.

### Run

```bash
docker run -d \
  --name edge-brain \
  -p 8080:8080 \
  -v edge-brain-data:/data/qdrant \
  edge-brain
```

The volume mount (`-v edge-brain-data:/data/qdrant`) persists the vector index across container restarts. Without it, the baked-in corpus is re-indexed on every cold start.

### Verify

```bash
# Wait ~30 seconds for startup, then:
curl http://localhost:8080/health
```

Expected response:
```json
{"status": "ok", "model_loaded": true, "corpus_chunks": 15, "uptime_seconds": 45}
```

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

The Docker image is built for `arm64`, which is compatible with Raspberry Pi 4/5 (64-bit OS).

### Option A: Pull from registry

```bash
# On your build machine: push to GitHub Container Registry
docker tag edge-brain ghcr.io/YOUR_USER/edge-brain:latest
docker push ghcr.io/YOUR_USER/edge-brain:latest

# On the Pi:
curl -fsSL https://get.docker.com | sh
docker pull ghcr.io/YOUR_USER/edge-brain:latest
docker run -d -p 8080:8080 -v edge-data:/data/qdrant ghcr.io/YOUR_USER/edge-brain:latest
```

### Option B: Transfer image file (air-gapped)

```bash
# On your build machine:
docker save edge-brain | gzip > edge-brain.tar.gz
# Transfer edge-brain.tar.gz to Pi via USB stick, scp, etc.

# On the Pi:
docker load < edge-brain.tar.gz
docker run -d -p 8080:8080 -v edge-data:/data/qdrant edge-brain
```

### Option C: Build on Pi

```bash
git clone https://github.com/YOUR_USER/ll-edge-brain.git
cd ll-edge-brain
docker build -t edge-brain .
docker run -d -p 8080:8080 -v edge-data:/data/qdrant edge-brain
```

Building on Pi is slower (llama-cpp-python compiles from source). Expect 20-30 minutes.

### Hardware notes

- **Raspberry Pi 5 (8 GB)**: runs well. Queries take 5-15 seconds depending on answer length.
- **Raspberry Pi 4 (4 GB)**: functional but tight on RAM. Use swap if needed (`sudo dphys-swapfile swapoff && sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile && sudo dphys-swapfile setup && sudo dphys-swapfile swapon`).
- **Raspberry Pi 4 (2 GB)**: not recommended. Model loading will likely fail.

## API Reference

### POST /query

Ask a question. Returns a grounded answer with source references.

**Request:**
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I prevent malaria?"}'
```

**Response:**
```json
{
  "answer": "Prevention includes sleeping under insecticide-treated bed nets, using mosquito repellent, and taking antimalarial medication.",
  "sources": [
    {"file": "health_basics.txt", "chunk": 0, "score": 4.9559}
  ],
  "language": "en"
}
```

The `score` is the cross-encoder reranker score (higher is more relevant). The `language` is auto-detected from the question.

Questions can be asked in any supported language. The system retrieves relevant chunks regardless of the document's language and generates an answer in the question's language:

```bash
# Norwegian question on English documents
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Hvordan kan jeg beskytte barnet mitt mot sykdommer?"}'

# Vietnamese question on Norwegian documents
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Toi duoc nghi phep bao nhieu ngay?"}'
```

If no relevant context is found, the response is:
```json
{
  "answer": "I don't have information about that.",
  "sources": [],
  "language": "en"
}
```

### POST /ingest

Upload documents to index at runtime.

**Request:**
```bash
# Single file
curl -X POST http://localhost:8080/ingest \
  -F "files=@document.pdf"

# Multiple files
curl -X POST http://localhost:8080/ingest \
  -F "files=@guide.pdf" \
  -F "files=@manual.txt" \
  -F "files=@policy.docx"
```

**Response:**
```json
{
  "ingested": [
    {"file": "guide.pdf", "chunks": 42, "language": "en"},
    {"file": "manual.txt", "chunks": 8, "language": "no"},
    {"file": "policy.docx", "chunks": 15, "language": "es"}
  ]
}
```

Supported formats: `.txt`, `.md`, `.pdf`, `.docx`, `.doc`, `.pptx`, `.xlsx`. PDF and Office formats require Bun and LiteParse.

Re-ingesting the same file overwrites existing chunks (deterministic IDs based on filename + chunk index).

### GET /health

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "corpus_chunks": 78,
  "uptime_seconds": 3600
}
```

### GET /corpus

**Response:**
```json
{
  "documents": [
    {"source_file": "health_basics.txt", "chunks": 5, "language": "en", "ingested_at": "2026-04-01T22:23:59Z"},
    {"source_file": "Ferie.pdf", "chunks": 65, "language": "no", "ingested_at": "2026-04-01T23:25:58Z"}
  ],
  "total_chunks": 70
}
```

## Configuration

All settings are configurable via environment variables with the `EDGE_` prefix:

```bash
# Override any default
EDGE_MODEL_PATH=~/models/tiny-aya-global-q4_k_m.gguf  # GGUF model location
EDGE_CHUNK_SIZE=300          # smaller chunks for more precise retrieval
EDGE_CHUNK_OVERLAP=30        # overlap between chunks
EDGE_TOP_K=5                 # number of chunks passed to LLM
EDGE_SCORE_THRESHOLD=0.4     # minimum similarity for retrieval
EDGE_MAX_TOKENS=150          # max generation length
EDGE_N_THREADS=8             # CPU threads for LLM (match your core count)
EDGE_N_CTX=2048              # LLM context window (lower = less RAM)
```

### Tuning guidelines

**EDGE_CHUNK_SIZE**: Smaller chunks (200-300) give more precise retrieval but may split important context across chunks. Larger chunks (500-800) preserve more context per chunk but may dilute relevance.

**EDGE_TOP_K**: More chunks give the LLM more context to work with but increase generation time and the chance of irrelevant information leaking in. 3 is a good default.

**EDGE_SCORE_THRESHOLD**: This is the initial retrieval threshold before reranking. Lower values (0.2-0.3) cast a wider net, relying on the reranker to filter. Higher values (0.5+) are stricter but may miss relevant chunks.

**EDGE_MAX_TOKENS**: Controls answer length. 100 tokens produces 1-3 sentence answers. Increase for questions that need longer explanations.

**EDGE_N_THREADS**: Set to your CPU core count. More threads speed up LLM generation but increase CPU load.

**EDGE_N_CTX**: The LLM context window. 4096 is sufficient for most queries. Lowering to 2048 reduces RAM usage by ~200 MB.

## Operations

### Adding documents to the corpus

**At build time (baked-in):** Place files in `data/corpus/` before building the Docker image. These are indexed automatically on first startup.

**At runtime:** Use the `/ingest` endpoint to upload files. These are indexed immediately and available for queries.

**Persistence:** Runtime-ingested documents are stored in the Qdrant Edge shard at `/data/qdrant/`. If this directory is a volume mount, documents survive container restarts. Without a volume mount, only baked-in corpus documents are available after restart (they are re-indexed automatically).

### Monitoring

Use the `/health` endpoint for liveness and readiness checks:

```bash
# Basic check
curl http://localhost:8080/health

# Watch corpus growth
watch -n 10 'curl -s http://localhost:8080/corpus | python3 -m json.tool'
```

Key indicators:
- `model_loaded: false` -- the LLM failed to load. Check logs for memory issues.
- `corpus_chunks: 0` -- no documents indexed. Check `EDGE_CORPUS_DIR` path and file formats.

### Logs

```bash
# Docker
docker logs edge-brain

# Docker follow
docker logs -f edge-brain

# Kubernetes
kubectl logs deployment/edge-rag-kiosk
```

Startup logs show model loading progress and any ingestion errors. Runtime logs show HTTP requests via uvicorn.

### Clearing the index

To re-index all documents from scratch:

```bash
# Docker: remove the volume
docker stop edge-brain && docker rm edge-brain
docker volume rm edge-brain-data
docker run -d --name edge-brain -p 8080:8080 -v edge-brain-data:/data/qdrant edge-brain

# Local development: delete the qdrant directory
rm -rf data/qdrant
```

On next startup, all baked-in corpus files will be re-indexed.

### Remote access with ngrok

To expose the brain to the internet from a local device or Pi:

```bash
# Install ngrok
curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok-v3-stable-linux-arm64.tgz | tar xz

# Expose port 8080
./ngrok http 8080
```

This gives a public URL (e.g., `https://abc123.ngrok.io`) that forwards to your local brain. Use it to query from anywhere:

```bash
curl https://abc123.ngrok.io/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What vaccines do children need?"}'
```

### Updating the corpus without rebuilding

If the container is running with a volume mount:

```bash
# Upload new documents via API
curl -X POST http://localhost:8080/ingest -F "files=@new_guide.pdf"

# Verify
curl http://localhost:8080/corpus
```

No restart needed. New documents are indexed immediately.

To replace baked-in corpus documents, rebuild the image with updated files in `data/corpus/`.

### Backup and restore

The vector index is stored in `/data/qdrant/`. To back up:

```bash
# Docker
docker cp edge-brain:/data/qdrant ./qdrant-backup

# Restore to a new container
docker run -d --name edge-brain-new -p 8080:8080 -v $(pwd)/qdrant-backup:/data/qdrant edge-brain
```

## Licensing

- **Tiny Aya Global**: CC-BY-NC (non-commercial). Commercial use requires contacting Cohere Sales.
- **FastEmbed models**: Apache 2.0
- **Qdrant Edge**: Apache 2.0
- **LiteParse**: MIT
