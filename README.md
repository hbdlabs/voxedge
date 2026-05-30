# VoxEdge

*Built by hbdlabs — making smart systems available for everyone, everywhere.*

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

The reranker is a precision filter that picks the best chunks from a larger candidate set. Two models, selected via `EDGE_RERANKER_MODEL`:

- **`jinaai/jina-reranker-v2-base-multilingual`** (1.1 GB, **default**) — ranks non-English and mixed-language corpora correctly.
- **`Xenova/ms-marco-MiniLM-L-6-v2`** (80 MB) — English-only and much smaller. Use it when your corpus is English to shrink the footprint. On non-English text it flatlines — scoring every chunk near-equal, so it can't tell good chunks from bad.

The embedding model and the LLM are multilingual regardless. The reranker is the only component where corpus language drives the choice.

### What makes it different

- **Fully offline** -- runs on a Raspberry Pi, laptop, or kiosk with no internet
- **Multilingual** -- questions and documents can be in different languages (50+ for retrieval, 70+ for generation). Ask in Vietnamese, get answers from English documents.
- **Self-contained** -- single Docker container with all models, vector database, document parser, and API server
- **Grounded** -- answers cite their sources and the system refuses to answer when it has no relevant information
- **Scales from Pi to GPU** -- same API and same model weights run on a $75 Raspberry Pi (CPU), a Mac (Metal), or an NVIDIA GPU / DGX Spark (CUDA). Only the deployment image changes.
- **Configurable** -- swap rerankers, adjust chunk size, tune retrieval thresholds, switch between full and chat mode

### Model profiles

A **profile** bundles one model's prompt templates, generation parameters, and runtime placement (CPU, Metal, or CUDA) under one name. Switch profiles by setting `EDGE_MODEL_PROFILE` — one env var picks which model runs and where.

Five profiles are included:

| Profile | Model | Hardware | Languages | License | Image |
|---|---|---|---|---|---|
| `gemma` | Gemma 4 E2B, Q4 GGUF (3.1 GB) | CPU | 35+ | Apache 2.0 (commercial OK) | `Dockerfile.gemma` (Pi) |
| `aya` | Tiny Aya Global, Q4 GGUF (2.1 GB) | CPU | 70+ | CC-BY-NC (non-commercial) | `Dockerfile.aya` (Pi) |
| `gemma-metal` | Gemma 4 E2B, Q4 GGUF | Apple GPU (Metal) | 35+ | Apache 2.0 | dev only, no Docker image |
| `gemma-cuda` | Gemma 4 E2B, Q4 GGUF | NVIDIA GPU (CUDA) | 35+ | Apache 2.0 | `Dockerfile.cuda` |
| `aya-cuda` | Tiny Aya Global, Q4 GGUF (2.1 GB) | NVIDIA GPU (CUDA) | 70+ | CC-BY-NC | `Dockerfile.cuda.aya` |

**All four profiles share the same GGUF model weights for their respective models — only the runtime placement differs.** A Pi deployment and a DGX Spark deployment run the same Gemma 4 E2B file; the Pi offloads nothing, the Spark offloads every layer.

**`gemma` and `aya`** — the Pi path. Pure CPU inference via `llama-cpp-python`. Defaults to `EDGE_MAX_TOKENS=100` for snappy kiosk UX under slow CPU generation.

**`gemma-metal`** — Apple Silicon dev dress rehearsal. Offloads every layer of Gemma 4 to the Apple GPU via llama.cpp's Metal backend. The default `llama-cpp-python` wheel on Apple Silicon already bundles Metal support; if yours doesn't, reinstall with `CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python`. No Docker image — this is a local-venv path for developers.

**`gemma-cuda`** — production GPU path. The LLM, embedder, *and* reranker all run on the NVIDIA GPU. Built for NVIDIA DGX Spark but runs on any CUDA GPU with ≥ 6 GB VRAM (RTX 3090/4090, A100, T4‑16GB, etc.). Uses the Jina multilingual reranker by default (VRAM is plentiful) and bumps `EDGE_MAX_TOKENS` to 800 so answers have room to breathe. See [`docs/gpu-profile-lessons.md`](docs/gpu-profile-lessons.md) for runtime gotchas (cuDNN, `onnxruntime-gpu`).

**`aya-cuda`** — same GPU placement as `gemma-cuda` but runs the smaller Tiny Aya model (70+ languages, CC-BY-NC). Built via `Dockerfile.cuda.aya`.

Separate Dockerfiles for each image live in `deploy/docker/`. The root Dockerfile defaults to Aya for backward compatibility.

### Stack

| Component | Library | Role |
|---|---|---|
| Document parsing | LiteParse (Python binding) | Extract text from PDF, DOCX, images |
| Text embedding | FastEmbed (ONNX, CPU or CUDA) | Convert text to 384-dim multilingual vectors |
| Vector storage | Qdrant Edge | Store and search vectors locally on disk |
| Reranking | FastEmbed Cross-Encoder (ONNX, CPU or CUDA) | Precision-filter retrieved chunks |
| Generation | llama-cpp-python (GGUF, CPU / Metal / CUDA) | Run LLM locally — CPU on Pi, Apple GPU on Mac dev, NVIDIA GPU on Spark / rental |
| API | FastAPI | HTTP endpoints for query, ingest, chat, translate, info, health |

For detailed component descriptions and data flow diagrams, see [docs/architecture.md](docs/architecture.md).

Sample documents for testing are in `examples/corpus/`.

## Requirements

Depends on which profile you deploy:

**CPU (Pi, laptop — `aya` / `gemma` profiles)**
- Python 3.11+
- LibreOffice — only to parse Office docs (`.docx`/`.xlsx`/`.pptx`); not needed for PDF or text
- ~4 GB RAM minimum (full mode), ~2.5 GB (chat mode)
- ~4 GB disk (model weights + vector storage)

**Apple Silicon dev (`gemma-metal` profile)**
- Same as CPU, plus llama-cpp-python with Metal enabled (default on Apple Silicon wheels)

**NVIDIA GPU (`gemma-cuda` profile — Spark, 4090, A100, T4, etc.)**
- NVIDIA driver ≥ 550
- CUDA 12.x runtime + cuDNN 9 (bundled in `Dockerfile.cuda` via the `cudnn-runtime` base image)
- ≥ 6 GB VRAM (with Jina multilingual reranker), or ≥ 4 GB (with MiniLM English reranker)
- `--gpus all` when running the container

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

PDF/DOCX parsing uses the in-process `liteparse` Python binding (installed above) — no Bun or Node needed. For Office docs (`.docx`/`.xlsx`/`.pptx`), also install LibreOffice (`brew install --cask libreoffice` / `apt-get install libreoffice`).

### 3. Download the GGUF model

```bash
mkdir -p ~/models
curl -L -o ~/models/tiny-aya-global-q4_k_m.gguf \
  "https://huggingface.co/CohereLabs/tiny-aya-global-GGUF/resolve/main/tiny-aya-global-q4_k_m.gguf"
```

This is ~2.1 GB. The model is stored in `~/models/` so it can be shared across projects and easily managed.

### 4. Run tests

```bash
python -m pytest tests/ -v
```

Unit and integration tests run without the GGUF model (they mock the generator). The end-to-end test (`test_e2e.py`) requires the model and is skipped automatically if the file is not found.

### 5. Start the server

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

## Deployment

Four deployment paths, pick the one that fits:

| Path | Best for | Config |
|---|---|---|
| **Raspberry Pi / K3s** | Kiosks, edge devices, offline | [`deploy/k8s/`](deploy/k8s/) with `deployment-gemma.yaml` or `deployment-aya.yaml` |
| **NVIDIA GPU / Spark** | High throughput, large reranker, GPU-accelerated retrieval | [`deploy/k8s/deployment-gemma-cuda.yaml`](deploy/k8s/deployment-gemma-cuda.yaml) |
| **Fly.io** | Cloud, public-facing, auto-suspend | [`deploy/fly/`](deploy/fly/) |
| **Docker** | Local testing, laptops, simple setups | [`deploy/docker/`](deploy/docker/) |

Choose an image per deployment target:

```bash
# Pi — Gemma 4 (Apache 2.0, commercial OK, 3.1 GB)
docker build -f deploy/docker/Dockerfile.gemma -t voxedge:gemma .

# Pi — Tiny Aya (CC-BY-NC, 70+ languages, 2.1 GB)
docker build -f deploy/docker/Dockerfile.aya -t voxedge:aya .

# NVIDIA GPU / Spark — Gemma 4, full GPU offload (~6-8 GB image)
docker build -f deploy/docker/Dockerfile.cuda -t voxedge:cuda .
```

### Raspberry Pi / K3s (recommended for kiosks)

Self-healing, declarative config, health monitoring. See [`deploy/k8s/README.md`](deploy/k8s/README.md) for the full guide (air-gapped install, SSD storage, fleet management).

```bash
curl -sfL https://get.k3s.io | sh -
sudo cp deploy/k8s/traefik-config.yaml /var/lib/rancher/k3s/server/manifests/
kubectl apply -k deploy/k8s/
```

### NVIDIA GPU / DGX Spark

For deployments where CPU inference latency matters — shared terminals with many users, long-form answers, or Spark-class hardware waiting for work. The LLM, embedder, and reranker all run on the GPU, with `EDGE_MAX_TOKENS=800` baked into the image so answers aren't truncated by the Pi-kiosk default.

**Docker (any NVIDIA GPU — 4090, A100, T4, DGX Spark)**

```bash
docker build -f deploy/docker/Dockerfile.cuda -t voxedge:cuda .
docker run -d --gpus all -p 8080:8080 -v voxedge-data:/data/qdrant voxedge:cuda
```

The same `Dockerfile.cuda` builds for both x86_64 (workstations, rentals) and ARM64 (DGX Spark). The NVIDIA CUDA base image, Python packages, and `pip install` of `llama-cpp-python` and `onnxruntime-gpu` all resolve to the correct architecture automatically — Docker picks the right layer from the multi-arch manifest, and pip downloads the right wheel.

**DGX Spark (ARM64).** Build directly on the device — simplest path:

```bash
# on the Spark
git clone https://github.com/hbdlabs/voxedge.git && cd voxedge && git checkout v0.7.0
docker build -f deploy/docker/Dockerfile.cuda -t voxedge:cuda .
docker run -d --gpus all -p 8080:8080 -v voxedge-data:/data/qdrant voxedge:cuda
```

The build takes longer on first run than on a 4090 (compiling `llama-cpp-python` CUDA kernels on Grace ARM64), but produces a Spark-native image. No separate Dockerfile, no code changes.

**Central registry with multi-arch manifest.** If you publish voxedge from CI for a fleet that mixes x86_64 GPU boxes and Spark, build both architectures under one tag:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f deploy/docker/Dockerfile.cuda \
  -t yourregistry/voxedge:cuda \
  --push .
```

Then any host — x86_64 workstation or Spark — can `docker pull yourregistry/voxedge:cuda` and Docker fetches the right architecture.

**Kubernetes (GPU node with NVIDIA device plugin installed)**

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/deployment-gemma-cuda.yaml
```

The deployment requests `nvidia.com/gpu: 1` and a longer startup probe window for the CUDA cold-start.

**Verify the GPU is actually being used.** `/info` reports both the profile's declared intent *and* the ONNX Runtime providers actually in use:

```json
"runtime": {
  "backend": "llama_cuda",
  "n_gpu_layers": -1,
  "embedder_device": "cuda",
  "reranker_device": "cuda",
  "embedder_active_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "reranker_active_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
}
```

If `embedder_active_providers` or `reranker_active_providers` are `["CPUExecutionProvider"]` only, CUDA silently fell back to CPU — usually a missing cuDNN library. See [`docs/gpu-profile-lessons.md`](docs/gpu-profile-lessons.md) for the fix.

### Fly.io

Performance CPUs required -- shared CPUs cannot run LLM inference within HTTP timeouts.

```bash
fly apps create voxedge
fly volumes create voxedge_data --region arn --size 5
fly deploy --config deploy/fly/fly.full.toml --dockerfile deploy/docker/Dockerfile.gemma --remote-only
```

Chat mode (no RAG, smaller machine): `fly deploy --config deploy/fly/fly.chat.toml --remote-only`

First startup takes 6-8 minutes. Subsequent starts are faster with `EDGE_CACHE_DIR` on a persistent volume.

### Docker

```bash
docker run -d --name voxedge -p 8080:8080 -v voxedge-data:/data/qdrant voxedge:gemma
```

Chat mode: add `-e EDGE_MODE=chat`. Volume mount not needed in chat mode.

### Verify (all paths)

```bash
curl http://localhost:8080/health
curl http://localhost:8080/info
```

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
- `.pdf` — parsed by LiteParse with spatial layout reconstruction. VoxEdge runs LiteParse with `--no-ocr`, so **text-layer PDFs only** (digital exports from Word/LaTeX/etc.). Scanned or image-only PDFs yield little to no text. To OCR scans, drop `--no-ocr` in `src/parser.py` and install Tesseract language data in the image.
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
  "model_profile": "gemma",
  "runtime": {
    "backend": "llama_cpu",
    "n_gpu_layers": 0,
    "embedder_device": "cpu",
    "reranker_device": "cpu",
    "embedder_active_providers": ["CPUExecutionProvider"],
    "reranker_active_providers": ["CPUExecutionProvider"]
  },
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
| `EDGE_TRANSLATE_QUERIES` | `false` | Translate a query into `EDGE_CORPUS_LANGUAGE` before retrieval when its detected language differs. Fixes cross-lingual mis-routing (e.g. a Vietnamese query retrieving the wrong English doc); the answer still comes back in the user's language. Costs one extra short LLM call per translated query. Best for a single-language corpus. |
| `EDGE_CORPUS_LANGUAGE` | `English` | Language the documents are in. Queries in other languages are translated to this when `EDGE_TRANSLATE_QUERIES=true`. |
| `EDGE_MAX_TOKENS` | `100` | Max generation tokens for query/chat |
| `EDGE_N_THREADS` | `4` | CPU threads for LLM |
| `EDGE_N_CTX` | `4096` | LLM context window |
| `EDGE_CORPUS_DIR` | `/data/corpus` | Baked-in documents directory |
| `EDGE_QDRANT_DIR` | `/data/qdrant` | Vector storage path |

### Choosing a reranker

See [Reranker options](#reranker-options) above for the why. Quick reference:

| Model | Size | Speed | Best for |
|---|---|---|---|
| `jinaai/jina-reranker-v2-base-multilingual` (default) | 1.1 GB | ~50ms/chunk | Non-English or mixed corpus |
| `Xenova/ms-marco-MiniLM-L-6-v2` | 80 MB | ~5ms/chunk | English-only corpus |

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
docker stop voxedge && docker rm voxedge
docker volume rm voxedge-data
docker run -d --name voxedge -p 8080:8080 -v voxedge-data:/data/qdrant voxedge

# Local: delete the qdrant directory
rm -rf data/qdrant
```

### API authentication

By default, the API has no authentication (suitable for local kiosks and air-gapped devices). For network deployments, set an API key:

```bash
# Docker
docker run -d -p 8080:8080 -e EDGE_API_KEY=your-secret-key voxedge

# Fly.io: set as a secret
fly secrets set EDGE_API_KEY=your-secret-key -a voxedge
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
docker cp voxedge:/data/qdrant ./qdrant-backup

# Restore
docker run -d --name voxedge-new -p 8080:8080 -v $(pwd)/qdrant-backup:/data/qdrant voxedge
```

## Retrieval Quality Guide

The quality of answers depends on how well the system retrieves the right chunks from your documents.

### How LiteParse handles documents

LiteParse preserves spatial structure rather than converting it away. Instead of transforming a PDF into markdown (which loses layout information), LiteParse extracts text with its original positioning — each text item comes with x/y coordinates, font size, and a confidence score.

We use this spatial data in `parser.py` to reconstruct clean text. The parser groups text items into lines by Y position, identifies garbled regions (OCR artifacts from screenshots, tiny mixed font sizes) and filters them out, and strips headers and footers by their position on the page.

```python
# Inspect what liteparse extracts (in-process Python binding, no Bun)
import liteparse
r = liteparse.LiteParse(ocr_enabled=False).parse("your_document.pdf")
print("pages:", r.num_pages)
print(r.pages[0].text[:500])           # reconstructed text for page 1
print(r.pages[0].text_items[0])        # one spatial item: x/y/font_size/text
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

### Optimizing answer quality

When answers are weak, the fix is usually one of these — roughly in order of impact:

- **The generation model is often the ceiling, not retrieval.** If answers are on-topic but vague or wrong, the model is struggling to *synthesize* the context it was given, not failing to find it. A larger or higher-quality model (the Gemma profile, or a bigger GGUF) raises answer quality without changing retrieval. Small edge models trade quality for footprint — fine for direct Q&A, weaker on dense or technical material.

- **Chunk size, more than chunk count.** Small chunks retrieve precisely but hand the model fragmented snippets; larger chunks (`EDGE_CHUNK_SIZE` ≈ 500–600) give it coherent passages and usually produce fuller, more accurate answers on dense documents — often a bigger lever than raising `EDGE_TOP_K`. Trade-off: larger chunks are slightly less targeted for retrieval and a little slower to generate, and changing this requires re-ingesting. Returning *more* chunks rarely rescues a wrong answer and can add noise.

- **Match the reranker to the corpus language.** The English `ms-marco` reranker is small and fast but unreliable on non-English text. For any non-English or mixed corpus, use the multilingual `jina-reranker-v2` (`EDGE_RERANKER_MODEL`).

- **Multilingual and lower-resource languages.** Cross-lingual retrieval relies on the embedding model placing different languages in a shared space. This holds up well for high-resource languages but weakens for lower-resource ones — a query in such a language can pull the wrong document. Two levers:
  - **`EDGE_TRANSLATE_QUERIES`** — translate the query into the corpus language before retrieval. The most reliable option, works uniformly across languages, at the cost of one extra LLM call per foreign-language query.
  - **A stronger multilingual embedder** improves cross-lingual alignment for more languages, but larger models cost RAM and speed (a real constraint on edge hardware) and may still under-serve the lowest-resource languages.

## Licensing

- **Gemma 4 E2B** (Google): Apache 2.0 (commercial use OK)
- **Tiny Aya Global** (CohereLabs): CC-BY-NC-4.0 (non-commercial)
- **Jina Reranker v2** (Jina AI): CC-BY-NC-4.0 (non-commercial)
- **FastEmbed models**: Apache 2.0
- **Qdrant Edge**: Apache 2.0
- **LiteParse**: MIT

The code in this repository is MIT licensed. Deploying with Gemma 4 + the English reranker gives a fully permissive commercial stack. Deploying with Tiny Aya or the Jina multilingual reranker adds CC-BY-NC restrictions. See [LICENSE](LICENSE) for details.
