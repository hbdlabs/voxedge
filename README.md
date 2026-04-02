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
| API | FastAPI | HTTP endpoints for query, ingest, chat, translate, health, corpus |

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

### POST /chat

Direct chat with the language model, no RAG retrieval. Useful for general questions, explanations, or conversation. Optional system prompt to set the tone or language.

**Request:**
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hva betyr feriepenger?", "system": "Svar kort på norsk."}'
```

**Response:**
```json
{
  "response": "Feriepenger er en type lønn som ansatte får under sin ferie..."
}
```

The `system` field is optional. Without it, the model responds in whatever language the message is in.

### POST /translate

Translate text between languages. Configurable via `EDGE_LOCAL_LANGUAGE` (default: Norwegian).

**Auto-detect direction** — if no source/target specified, English input translates to the local language and non-English input translates to English:

```bash
# English → Norwegian (auto)
curl -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "The vaccine is free at all health centers."}'

# Norwegian → English (auto)
curl -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Alle ansatte har rett på 5 uker ferie hvert år."}'
```

**Explicit source and target:**

```bash
# Vietnamese → Norwegian
curl -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Xin chào, tôi cần giúp đỡ.", "source": "Vietnamese", "target": "Norwegian"}'

# Spanish → English
curl -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "La vacuna es gratuita.", "source": "Spanish", "target": "English"}'
```

**Response:**
```json
{
  "translation": "Legen anbefaler at du tar denne medisinen to ganger i døgnet med mat.",
  "source": "English",
  "target": "Norwegian"
}
```

To change the default local language:
```bash
EDGE_LOCAL_LANGUAGE=Vietnamese  # or Spanish, French, etc.
```

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
EDGE_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual  # reranker model
EDGE_CHUNK_SIZE=300          # smaller chunks for more precise retrieval
EDGE_CHUNK_OVERLAP=30        # overlap between chunks
EDGE_TOP_K=5                 # number of chunks passed to LLM
EDGE_SCORE_THRESHOLD=0.4     # minimum similarity for retrieval
EDGE_MAX_TOKENS=150          # max generation length
EDGE_N_THREADS=8             # CPU threads for LLM (match your core count)
EDGE_N_CTX=2048              # LLM context window (lower = less RAM)
```

### Choosing a reranker

The reranker is the precision filter between vector search and the LLM. Pick based on your corpus language:

```bash
# English-only corpus (default for lightweight deployments)
EDGE_RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2

# Non-English or mixed-language corpus (default)
EDGE_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
```

| Model | Size | Speed | Best for |
|---|---|---|---|
| `Xenova/ms-marco-MiniLM-L-6-v2` | 80 MB | ~5ms/chunk | English-only corpus |
| `jinaai/jina-reranker-v2-base-multilingual` | 1.1 GB | ~50ms/chunk | Non-English or mixed corpus |

The multilingual reranker is the default because the system is designed for multilingual use. If you deploy with English-only content and want a smaller footprint, switch to the English model.

The English reranker struggles with non-English content — it may score the wrong chunk higher when multiple similar chunks compete in a non-English document. This leads to correct retrieval (right document found) but wrong answers (wrong chunk selected from that document).

### Tuning guidelines

**EDGE_CHUNK_SIZE**: Smaller chunks (200-300) give more precise retrieval but may split important context across chunks. Larger chunks (500-800) preserve more context per chunk but may dilute relevance.

**EDGE_TOP_K**: More chunks give the LLM more context to work with but increase generation time and the chance of irrelevant information leaking in. 5 is the default. Reduce to 3 if answers are unfocused, increase to 7 if the right information keeps getting excluded.

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

## Retrieval Quality Guide

The quality of answers depends entirely on how well the system retrieves the right chunks from your documents. Understanding your source material and how it gets processed is key to getting good results.

### How LiteParse handles documents

LiteParse's approach to document parsing is to preserve spatial structure rather than convert it away. Instead of transforming a PDF into markdown or HTML (which loses layout information), LiteParse extracts text with its original positioning — each text item comes with x/y coordinates, font size, and a confidence score. This means the physical structure of the document (headings, columns, tables, margins) is retained as data, not interpreted and flattened.

We use this spatial data in `parser.py` to reconstruct clean text. The parser groups text items into lines by Y position, identifies garbled regions (OCR artifacts from screenshots, tiny mixed font sizes) and filters them out, and strips headers and footers by their position on the page. The result is cleaner text than a naive `--format text` extraction, because the spatial information tells us which parts of the page are real content and which are noise.

You can inspect the raw spatial output to understand what LiteParse sees:

```bash
# Spatial JSON with bounding boxes
bunx @llamaindex/liteparse parse your_document.pdf --format json | python3 -m json.tool

# Plain text (what the default text mode produces, before our spatial filtering)
bunx @llamaindex/liteparse parse your_document.pdf --format text
```

### Know your documents

Before ingesting, review what LiteParse extracts from your documents:

```bash
# See what text gets extracted from a PDF
bunx @llamaindex/liteparse parse your_document.pdf --format text
```

Check for:
- **Garbled text from screenshots or tables** — the spatial parser filters most of this, but complex layouts may still produce noise
- **Important facts buried in tables** — tabular data often extracts poorly. If critical information lives in tables, consider adding a plain text summary document alongside the PDF
- **Contradictory information** — if the same topic (e.g., phone budget) appears in multiple places with different numbers, the model may pick the wrong one
- **Information split across pages** — a sentence that starts at the bottom of one page and continues at the top of the next may get split into different chunks

### Check how documents get chunked

The chunker splits text into 250-character overlapping windows. Short facts that land at chunk boundaries may end up in a chunk where they're surrounded by unrelated content:

```bash
# Preview how a document gets chunked
python -c "
from src.parser import parse_file
from src.chunker import chunk_text
from pathlib import Path

text = parse_file(Path('your_document.pdf'))
chunks = chunk_text(text, chunk_size=250, overlap=30)
for i, chunk in enumerate(chunks):
    print(f'--- Chunk {i} ---')
    print(chunk)
    print()
"
```

If important facts end up buried at the end of a chunk with unrelated content at the start, the reranker may not score that chunk highly enough.

### How retrieval works and where it can fail

The query pipeline has three stages, each with its own failure mode:

**1. Embedding search** (retrieve 20 candidates, threshold >= 0.3)
- Failure mode: the question and the relevant chunk use very different vocabulary. The multilingual embedding model handles synonyms and cross-lingual matching well, but highly domain-specific terminology may not match.
- Diagnosis: if the answer exists in your corpus but the system says "I don't have information about that", the embedding search didn't find it above the threshold.

**2. Cross-encoder reranking** (pick best 5 from the 20 candidates)
- Failure mode: a chunk about a related but wrong topic scores higher than the chunk with the correct answer. For example, a question about "PC budget" may match a chunk about "phone budget" because both discuss equipment pricing.
- Diagnosis: the answer is wrong but the source file is correct — the reranker picked a plausible but wrong chunk from the right document.

**3. LLM generation** (generate answer from the 5 chunks)
- Failure mode: the right chunks are retrieved but the 3B model misinterprets, conflates numbers, or picks the wrong fact from the context.
- Diagnosis: the source file and chunk are correct, but the answer contains a wrong number or mixes up details.

### Improving retrieval

**Phrasing matters.** Questions that use the same vocabulary as the source documents get better results. "Hvor mange dager kan jeg bruke egenmelding?" works better than "Hvor mange egenmeldingsdager har jeg?" if the document says "bruke egenmelding".

**Add plain text summaries.** If a PDF has important information locked in tables, screenshots, or complex layouts, add a companion `.txt` file that restates the key facts in clean prose. This gives the system clean, chunkable text to work with.

**Adjust chunk size.** The default 250 characters works well for documents with short, factual paragraphs. For documents with longer narrative sections, increasing `EDGE_CHUNK_SIZE` to 400-500 may help keep related facts together. For documents with many short facts (price lists, policy rules), 200 or even 150 may be better.

**Adjust score threshold.** If the system says "I don't have information" for questions you know are covered, lower `EDGE_SCORE_THRESHOLD` from 0.3 to 0.2. If it gives wrong answers from irrelevant chunks, raise it to 0.4.

**Adjust top_k.** More chunks to the LLM (higher `EDGE_TOP_K`) means the correct chunk is more likely to be included, but also gives the model more noise to sort through. 5 is a good default. Reduce to 3 if answers are unfocused, increase to 7 if the right information keeps getting excluded.

### Testing your corpus

After ingesting documents, test with questions where you know the expected answer. Compare the system's answer and source references against the actual document content. This is the most reliable way to identify retrieval gaps and tune the configuration for your specific documents.

## Licensing

- **Tiny Aya Global**: CC-BY-NC (non-commercial). Commercial use requires contacting Cohere Sales.
- **FastEmbed models**: Apache 2.0
- **Qdrant Edge**: Apache 2.0
- **LiteParse**: MIT
