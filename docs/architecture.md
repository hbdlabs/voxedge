# Edge RAG Brain - Technical Architecture

## Overview

A self-contained, offline RAG (Retrieval-Augmented Generation) system that runs entirely in a single process. No external APIs, no cloud services, no internet required. Designed for multilingual knowledge access on edge devices -- kiosks, laptops, Raspberry Pi, or Kubernetes pods.

The system ingests documents, splits them into chunks, embeds them as vectors, stores them locally, and answers questions by retrieving relevant chunks and generating grounded responses using a local LLM.

## System Diagram

```
                         POST /query { "question": "..." }
                                    |
                                    v
+-------------------------------------------------------------------+
|  main.py (FastAPI)                                                |
|                                                                   |
|  Startup:                          Endpoints:                     |
|  - Load Embedder                   - POST /query  --> query.py    |
|  - Load VectorStore                - POST /ingest --> ingest.py   |
|  - Load Generator                  - GET  /health                 |
|  - Load Reranker                   - GET  /corpus                 |
|  - Ingest baked-in corpus                                         |
+-------------------------------------------------------------------+
         |                |                |               |
         v                v                v               v
   +-----------+   +------------+   +------------+   +-----------+
   | embedder  |   |   store    |   | generator  |   | reranker  |
   | .py       |   |   .py      |   | .py        |   | .py       |
   |           |   |            |   |            |   |           |
   | FastEmbed |   | Qdrant     |   | llama-cpp  |   | FastEmbed |
   | (ONNX)    |   | Edge       |   | -python    |   | Cross-    |
   | 220 MB    |   | (mmap)     |   | Tiny Aya   |   | Encoder   |
   |           |   |            |   | 2.1 GB     |   | 80 MB     |
   +-----------+   +------------+   +------------+   +-----------+
```

## Components

### config.py -- Settings

Central configuration using Pydantic Settings. All values load from environment variables with the `EDGE_` prefix, with sensible defaults for container deployment.

Pydantic Settings is a library that maps environment variables to typed Python attributes with validation. It allows the same code to run in development (with defaults) and production (with env vars) without any code changes.

| Setting | Default | Purpose |
|---|---|---|
| `EDGE_MODEL_PATH` | `/data/models/tiny-aya-global-q4_k_m.gguf` | Path to the GGUF language model |
| `EDGE_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | FastEmbed model for text-to-vector |
| `EDGE_RERANKER_MODEL` | `jinaai/jina-reranker-v2-base-multilingual` | Cross-encoder reranker model |
| `EDGE_CHUNK_SIZE` | `250` | Characters per chunk |
| `EDGE_CHUNK_OVERLAP` | `30` | Overlap between adjacent chunks |
| `EDGE_TOP_K` | `5` | Final number of chunks passed to the LLM |
| `EDGE_SCORE_THRESHOLD` | `0.3` | Minimum cosine similarity for initial retrieval |
| `EDGE_MAX_TOKENS` | `100` | Maximum tokens the LLM generates per answer |
| `EDGE_CORPUS_DIR` | `/data/corpus` | Directory for baked-in documents |
| `EDGE_QDRANT_DIR` | `/data/qdrant` | Qdrant Edge on-disk storage path |
| `EDGE_N_CTX` | `4096` | LLM context window size |
| `EDGE_N_THREADS` | `4` | CPU threads for LLM inference |

Also provides `detect_language(text)` -- a shared utility wrapping langdetect with a fallback to `"unknown"`. langdetect is a port of Google's language-detection library that identifies the language of a text string from its character n-grams.

**Used by:** main.py (reads settings to initialize all components), ingest.py and query.py (detect_language)

---

### parser.py -- Document Text Extraction

Extracts raw text from document files. Two code paths:

- **Plain text** (`.txt`, `.md`): read directly with `path.read_text()`. No external dependencies.
- **Rich documents** (`.pdf`, `.docx`, `.pptx`, `.xlsx`, images): shells out to LiteParse via Bun as a subprocess: `bunx @llamaindex/liteparse parse <file> --format text`.

**LiteParse** is a document parser by LlamaIndex that extracts text from PDFs using PDF.js, handles Office documents via LibreOffice conversion, and supports OCR via Tesseract.js. It runs on Bun (a fast JavaScript runtime, alternative to Node.js) and is invoked as a CLI subprocess from Python. The `--format text` flag returns plain extracted text; LiteParse also supports `--format json` for spatial layout with bounding boxes.

**Input:** file path
**Output:** extracted text as a single string
**Used by:** ingest.py

---

### chunker.py -- Text Splitting

Splits extracted text into overlapping windows of fixed character length. The overlap ensures that information at chunk boundaries isn't lost during retrieval -- a sentence that spans two chunks will appear in both, so it can be found regardless of which chunk the search matches.

With defaults (500 chars, 50 overlap), a 1200-character document produces:
```
Chunk 0: chars 0-499
Chunk 1: chars 450-949     (50 char overlap with chunk 0)
Chunk 2: chars 900-1199    (50 char overlap with chunk 1)
```

This is a character-based sliding window approach. Alternative strategies include semantic chunking (split at sentence or paragraph boundaries) and recursive chunking (split at decreasing granularity), which may produce more coherent chunks but add complexity.

**Input:** text string, chunk_size, overlap
**Output:** list of text chunks
**Used by:** ingest.py

---

### embedder.py -- Text-to-Vector Conversion

Wraps FastEmbed's `TextEmbedding` class using the `paraphrase-multilingual-MiniLM-L12-v2` model.

**FastEmbed** is a Python library by Qdrant for generating text embeddings. It uses ONNX Runtime for inference, which means models run on CPU without PyTorch or GPU dependencies. Models are downloaded from HuggingFace on first use and cached locally.

**paraphrase-multilingual-MiniLM-L12-v2** is a sentence-transformer model trained on parallel data across 50+ languages. It is a bi-encoder: it processes each text independently and produces a fixed 384-dimensional vector. Texts with similar meaning land near each other in this vector space regardless of language. This is what enables cross-lingual retrieval -- a Norwegian question and an English document about the same topic produce vectors with high cosine similarity.

- **Runtime:** ONNX Runtime (no GPU, no PyTorch)
- **Model size:** ~220 MB
- **Output:** 384-dimensional float vector per input text
- **Latency:** ~20-50 ms per embed call
- **Languages:** 50+ including en, es, fr, de, no, vi, zh, ja, ko, ar, hi, th, tr, uk, ...

**Input:** list of text strings
**Output:** list of 384-float vectors
**Used by:** ingest.py (embed chunks for storage), query.py (embed question for search)

---

### store.py -- Vector Storage and Search

Wraps Qdrant Edge (`qdrant-edge-py`), an embedded vector database.

**Qdrant Edge** is a lightweight, in-process vector search engine by Qdrant, designed for edge devices. It operates similarly to SQLite -- data is stored on local disk and accessed in-process, with no server or network required. Vectors are stored in memory-mapped files, so the OS pages data in and out of RAM as needed. For small datasets (thousands of chunks), search is brute-force cosine similarity. For larger datasets, Qdrant Edge supports HNSW (Hierarchical Navigable Small World) indexing for approximate nearest neighbor search.

Key operations:

| Method | Purpose |
|---|---|
| `upsert(id, vector, payload)` | Store a single chunk with its vector and metadata |
| `upsert_batch(points)` | Store multiple chunks at once |
| `query(vector, limit, threshold)` | Find nearest vectors by cosine similarity |
| `count()` | Total number of stored chunks |
| `list_documents()` | Scroll all chunks, aggregate by source file |
| `flush()` / `close()` | Persist to disk and shut down |

Data is stored on disk at `EDGE_QDRANT_DIR`. The shard auto-loads on restart if data exists, or creates a fresh shard if not.

Each stored point contains:
```
{
    id:      <deterministic hash of filename + chunk_index>,
    vector:  [384 floats],
    payload: {
        "text":        "the actual chunk text",
        "source_file": "farming_guide.txt",
        "chunk_index": 0,
        "language":    "en",
        "ingested_at": "2026-04-01T22:23:59Z"
    }
}
```

Point IDs are generated via SHA-256 hash of `source_file:chunk_index`, making re-ingestion of the same file idempotent -- upserting the same ID overwrites the existing point.

**Used by:** ingest.py (store vectors), query.py (search vectors), main.py (count, list, lifecycle)

---

### reranker.py -- Cross-Encoder Reranking

Wraps FastEmbed's `TextCrossEncoder`. The model is configurable via `EDGE_RERANKER_MODEL`.

**Cross-encoder reranking** is a second-stage retrieval technique. The embedder (bi-encoder) is fast but approximate: it encodes the question and each chunk independently, then compares their vectors. A cross-encoder is slower but more precise: it takes the question and a chunk as a single concatenated input, allowing full token-level attention between them. This means it can understand the relationship between question and chunk more accurately than vector similarity alone.

Two models are supported:

- **Xenova/ms-marco-MiniLM-L-6-v2** (80 MB, English) — lightweight, fast, suitable for English-only corpora
- **jinaai/jina-reranker-v2-base-multilingual** (1.1 GB, 100+ languages) — built on XLM-RoBERTa, understands non-English text natively. Default for multilingual deployments.

The pipeline: retrieve 10 candidates loosely with the bi-encoder (score threshold 0.3), then rerank to pick the best 3 with the cross-encoder.

**Input:** question string + list of candidate chunks from vector search
**Output:** top_k chunks sorted by cross-encoder relevance score
**Used by:** query.py (between retrieval and generation)

---

### generator.py -- Answer Generation

Wraps `llama-cpp-python` to load and run a Tiny Aya GGUF model locally on CPU.

**llama-cpp-python** is a Python binding for llama.cpp, a C++ library for running LLM inference on consumer hardware. It loads models in GGUF format (a binary format for quantized model weights) and runs inference on CPU with optional GPU offloading. Quantization (Q4_K_M in our case) reduces the model from ~6.7 GB (full precision) to ~2.1 GB with minimal quality loss.

**Tiny Aya Global** is a 3.35B parameter multilingual language model by Cohere Labs. It is instruction-tuned for general-purpose multilingual generation across 70+ languages. The "global" variant is the general-purpose instruction-tuned version (as opposed to "earth", "water", "fire" which are specialized variants).

The prompt template enforces grounded generation:
```
You are a helpful assistant at a community knowledge kiosk.

Rules:
- ONLY use information from the Context below
- If the Context does not answer the question, reply ONLY with:
  "I don't have information about that."
- Do NOT make up information
- Do NOT add information from your own knowledge
- Answer in the same language as the question
- Keep your answer short and direct

Context:
{retrieved chunks joined by newlines}

Question: {user's question}

Answer:
```

Key generation parameters:
- `temperature=0.3` -- low randomness for factual answers
- `repeat_penalty=1.3` -- penalizes tokens that have already appeared, preventing repetitive loops
- `stop=["\nQuestion:", "\n\n\n", "\nNote:", "(Note:", "\nAnswer:"]` -- stops generation at natural boundaries, prevents the model from generating multiple answer blocks
- `max_tokens=100` -- keeps answers concise

Also includes a Jinja2 monkey-patch to add the `loopcontrols` extension, which is needed because the Tiny Aya GGUF embeds a chat template that uses `{% break %}` -- a Jinja2 tag not available without this extension.

**Input:** list of context chunks + question
**Output:** generated answer string
**Used by:** query.py

---

### ingest.py -- Ingestion Pipeline

Orchestrates the full document-to-vectors pipeline:

```
Document file
    --> parser.py      (extract text)
    --> chunker.py     (split into ~500 char overlapping segments)
    --> detect_language (identify language of first chunk)
    --> embedder.py    (convert each chunk to 384-dim vector)
    --> store.py       (store vectors + metadata in Qdrant Edge)
```

Two entry points:
- `ingest_file(path)` -- ingest a single document. Accepts an optional `source_name` parameter to store the original filename when ingesting from temp files (e.g., uploads).
- `ingest_directory(directory)` -- ingest all supported files (`.txt`, `.md`, `.pdf`, `.docx`, `.doc`, `.pptx`, `.xlsx`) in a folder.

Returns `IngestResult(file, chunks, language)` for each document.

**Used by:** main.py (startup corpus ingestion + `/ingest` endpoint)

---

### query.py -- Query Pipeline

Orchestrates the full question-to-answer pipeline:

```
User question
    --> detect_language     (identify question language)
    --> embedder.py         (question --> 384-dim vector)
    --> store.py            (retrieve top 10 candidates, threshold >= 0.3)
    --> [empty guard]       (if 0 results: return canned response, skip LLM)
    --> reranker.py         (cross-encoder picks best 3 chunks)
    --> generator.py        (LLM generates grounded answer from context)
    --> QueryResult         (answer + sources + language)
```

The empty-sources guard is a code-level check: if vector search returns zero results above the threshold, a static "I don't have information about that." response is returned immediately without calling the LLM. This prevents the model from answering questions that have no basis in the corpus.

**Used by:** main.py (`/query` endpoint)

---

### main.py -- FastAPI Application

The entry point that wires everything together.

**FastAPI** is a Python web framework for building HTTP APIs. It uses Pydantic for request/response validation and supports async endpoints. Uvicorn is the ASGI server that runs the application.

**Startup (lifespan handler):**
1. Initialize Embedder (loads ONNX model, ~220 MB)
2. Initialize VectorStore (opens or creates Qdrant Edge shard)
3. Initialize Generator (loads GGUF model into RAM, ~2.1 GB)
4. Initialize Reranker (loads cross-encoder, ~80 MB)
5. Ingest baked-in corpus -- compares files in `EDGE_CORPUS_DIR` against already-indexed documents (via `store.list_documents()`), ingests only files not already present

**Shutdown:** flush and close Qdrant Edge shard.

**Endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/query` | POST | Accept a question (JSON), return a grounded answer with source references |
| `/ingest` | POST | Accept document files (multipart upload), parse, chunk, embed, and store them |
| `/health` | GET | Return status, whether the model is loaded, total chunk count, and uptime |
| `/corpus` | GET | List all indexed documents with their chunk counts, languages, and ingestion timestamps |

Uses a factory pattern (`create_app()`) that accepts pre-built components for testing. In production, components are `None` and initialized during the lifespan startup phase.

---

## Data Flow

### Ingestion

```
PDF/TXT/DOCX file
     |
     v
[parser.py] -- LiteParse (Bun) for PDFs, direct read for .txt/.md
     |
     v
Raw text string
     |
     v
[chunker.py] -- 500 char windows, 50 char overlap
     |
     v
["chunk 0", "chunk 1", "chunk 2", ...]
     |
     v
[embedder.py] -- FastEmbed multilingual MiniLM (ONNX)
     |
     v
[[0.02, -0.04, ...], [0.01, 0.08, ...], ...]   (384 floats each)
     |
     v
[store.py] -- Qdrant Edge upsert with metadata
     |
     v
On-disk shard at /data/qdrant/
```

### Query

```
"How do I prevent malaria?"
     |
     v
[embedder.py] -- same model, question --> 384-dim vector
     |
     v
[store.py] -- cosine similarity, top 10, threshold >= 0.3
     |
     v
10 candidate chunks (loose matches)
     |
     v
[reranker.py] -- cross-encoder scores each (question, chunk) pair
     |
     v
3 best chunks (precise matches)
     |
     v
[generator.py] -- Tiny Aya reads context + question, generates answer
     |
     v
{ "answer": "...", "sources": [...], "language": "en" }
```

## Deployment

### Container Image (3.24 GB)

```
Python 3.11-slim base
  + Bun runtime (for LiteParse)
  + LiteParse CLI
  + Python dependencies (FastEmbed, qdrant-edge-py, llama-cpp-python, ...)
  + Tiny Aya GGUF model (2.1 GB, downloaded at build time)
  + Baked-in corpus documents
```

### Runtime Requirements

| Resource | Minimum |
|---|---|
| RAM | 4 GB |
| CPU | 4 cores (ARM or x86) |
| Disk | 4 GB (image) + storage for vectors |
| Network | None (fully offline after deployment) |

### Startup Time

| Step | Duration |
|---|---|
| FastEmbed model load | ~2s (cached) or ~10s (first download) |
| Qdrant Edge shard open | <1s |
| Tiny Aya GGUF load | ~5s |
| Reranker model load | ~2s |
| Corpus ingestion | depends on document count |
| **Total cold start** | **~15-30s** |

## Models

| Model | Type | Size | Purpose | Languages |
|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | Bi-encoder (ONNX) | 220 MB | Text embedding | 50+ |
| jinaai/jina-reranker-v2-base-multilingual | Cross-encoder (ONNX) | 1.1 GB | Reranking (default) | 100+ |
| Xenova/ms-marco-MiniLM-L-6-v2 | Cross-encoder (ONNX) | 80 MB | Reranking (English alt) | English |
| CohereLabs/tiny-aya-global (Q4_K_M) | Generative LLM (GGUF) | 2.1 GB | Answer generation | 70+ |

## Quality Controls

| Mechanism | What it prevents |
|---|---|
| Score threshold (0.3) | Irrelevant chunks reaching the reranker |
| Cross-encoder reranker | Wrong chunks reaching the LLM |
| Empty-sources guard | LLM being called when no relevant context exists |
| Grounding prompt | LLM adding information from its own knowledge |
| repeat_penalty (1.3) | Repetitive or looping output |
| Stop sequences | Multi-answer rambling |
| max_tokens (100) | Overly long, unfocused answers |
| Deterministic point IDs | Duplicate chunks on re-ingestion |
| Startup dedup check | Re-indexing already-ingested corpus files |

## Key Technologies

### ONNX and ONNX Runtime

ONNX (Open Neural Network Exchange) is a standard file format for machine learning models. It allows a model trained in one framework (e.g., PyTorch) to be exported as a `.onnx` file and run in a different runtime. ONNX Runtime is a lightweight C++ inference engine by Microsoft that executes these models. It supports CPU-optimized kernels for both ARM and x86 architectures.

The embedder and reranker both use ONNX Runtime through FastEmbed. This avoids a PyTorch dependency (~2 GB), requires no GPU, and keeps the runtime footprint small -- critical for edge deployment.

### GGUF

GGUF (GPT-Generated Unified Format) is a binary file format for storing quantized LLM weights, used by llama.cpp. Quantization reduces model precision (e.g., from 16-bit floats to 4-bit integers) to shrink model size and memory usage. The Q4_K_M quantization used here reduces Tiny Aya from ~6.7 GB (full precision) to ~2.1 GB with minimal quality loss. llama-cpp-python loads GGUF files directly into memory for inference.

### Cosine Similarity

The distance metric used for vector search. It measures the angle between two vectors, returning a value from -1 (opposite) to 1 (identical). Two texts about the same topic produce vectors with high cosine similarity regardless of language, because the multilingual embedding model maps semantically similar text to nearby regions in vector space.

## Test Coverage

27 unit/integration tests + 1 end-to-end test with real models.
