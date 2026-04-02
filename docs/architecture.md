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

Central configuration loaded from environment variables with the `EDGE_` prefix.

| Setting | Default | Purpose |
|---|---|---|
| `EDGE_MODEL_PATH` | `/data/models/tiny-aya-global-q4_k_m.gguf` | Path to the GGUF language model |
| `EDGE_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | FastEmbed model for text-to-vector |
| `EDGE_CHUNK_SIZE` | `500` | Characters per chunk |
| `EDGE_CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `EDGE_TOP_K` | `3` | Final number of chunks passed to the LLM |
| `EDGE_SCORE_THRESHOLD` | `0.3` | Minimum cosine similarity for initial retrieval |
| `EDGE_MAX_TOKENS` | `100` | Maximum tokens the LLM generates per answer |
| `EDGE_CORPUS_DIR` | `/data/corpus` | Directory for baked-in documents |
| `EDGE_QDRANT_DIR` | `/data/qdrant` | Qdrant Edge on-disk storage path |
| `EDGE_N_CTX` | `4096` | LLM context window size |
| `EDGE_N_THREADS` | `4` | CPU threads for LLM inference |

Also provides `detect_language(text)` -- a shared utility wrapping `langdetect` with a fallback to `"unknown"`.

**Used by:** main.py (reads settings to initialize all components)

---

### parser.py -- Document Text Extraction

Extracts raw text from document files. Two code paths:

- **Plain text** (`.txt`, `.md`): read directly with `path.read_text()`. No external dependencies.
- **Rich documents** (`.pdf`, `.docx`, `.pptx`, `.xlsx`, images): shells out to LiteParse via Bun as a subprocess: `bunx @llamaindex/liteparse parse <file> --format text`. LiteParse handles PDF parsing, OCR, and Office document conversion.

**Input:** file path  
**Output:** extracted text as a single string  
**Used by:** ingest.py

---

### chunker.py -- Text Splitting

Splits extracted text into overlapping windows of fixed character length. The overlap ensures that information at chunk boundaries isn't lost during retrieval.

With defaults (500 chars, 50 overlap), a 1200-character document produces:
```
Chunk 0: chars 0-499
Chunk 1: chars 450-949     (50 char overlap with chunk 0)
Chunk 2: chars 900-1199    (50 char overlap with chunk 1)
```

**Input:** text string, chunk_size, overlap  
**Output:** list of text chunks  
**Used by:** ingest.py

---

### embedder.py -- Text-to-Vector Conversion

Wraps FastEmbed's `TextEmbedding` class. Converts text strings into 384-dimensional float vectors using the `paraphrase-multilingual-MiniLM-L12-v2` model.

This model is the key to cross-lingual retrieval. It maps text from 50+ languages into a shared vector space, so a Norwegian question and an English document about the same topic land near each other in 384-dimensional space.

- **Runtime:** ONNX (no GPU, no PyTorch)
- **Model size:** ~220 MB (downloaded and cached on first use)
- **Latency:** ~20-50 ms per embed call
- **Languages:** 50+ including en, es, fr, de, no, vi, zh, ja, ko, ar, hi, ...

**Input:** list of text strings  
**Output:** list of 384-float vectors  
**Used by:** ingest.py (embed chunks for storage), query.py (embed question for search)

---

### store.py -- Vector Storage and Search

Wraps Qdrant Edge (`qdrant-edge-py`), an embedded vector database that runs in-process. Similar to SQLite but for vectors.

Key operations:

| Method | Purpose |
|---|---|
| `upsert(id, vector, payload)` | Store a single chunk with its vector and metadata |
| `upsert_batch(points)` | Store multiple chunks at once |
| `query(vector, limit, threshold)` | Find nearest vectors by cosine similarity |
| `count()` | Total number of stored chunks |
| `list_documents()` | Scroll all chunks, aggregate by source file |
| `flush()` / `close()` | Persist to disk and shut down |

Data is stored on disk at `EDGE_QDRANT_DIR` using memory-mapped files. The shard auto-loads on restart if data exists, or creates a fresh shard if not.

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

Point IDs are generated via SHA-256 hash of `source_file:chunk_index`, making re-ingestion of the same file idempotent.

**Used by:** ingest.py (store vectors), query.py (search vectors), main.py (count, list, lifecycle)

---

### reranker.py -- Cross-Encoder Reranking

Wraps FastEmbed's `TextCrossEncoder` using the `Xenova/ms-marco-MiniLM-L-6-v2` model (80 MB).

The embedder performs fast but approximate matching (bi-encoder: question and chunks are embedded independently). The reranker performs slower but precise matching (cross-encoder: question and each chunk are processed together, allowing token-level attention).

Pipeline: retrieve 10 candidates loosely with the embedder, then rerank to pick the best 3 with the cross-encoder.

This was the single biggest quality improvement to the system, taking retrieval accuracy from 6/10 to 10/10 on known-answer tests.

**Input:** question string + list of candidate chunks from vector search  
**Output:** top_k chunks sorted by cross-encoder relevance score  
**Used by:** query.py (between retrieval and generation)

---

### generator.py -- Answer Generation

Wraps `llama-cpp-python` to load and run a Tiny Aya 3.35B GGUF model locally on CPU.

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
- `repeat_penalty=1.3` -- prevents repetitive output
- `stop=["\nQuestion:", "\n\n\n", "\nNote:", "(Note:", "\nAnswer:"]` -- stops generation at natural boundaries, prevents multi-answer rambling
- `max_tokens=100` -- keeps answers concise

Also includes a Jinja2 monkey-patch to add `loopcontrols` extension support, which is needed because the Tiny Aya GGUF embeds a chat template that uses `{% break %}`.

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
- `ingest_file(path)` -- ingest a single document
- `ingest_directory(directory)` -- ingest all supported files in a folder

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
    --> [empty guard]       (if 0 results: return "I don't have information")
    --> reranker.py         (cross-encoder picks best 3 chunks)
    --> generator.py        (LLM generates grounded answer from context)
    --> QueryResult         (answer + sources + language)
```

The empty-sources guard prevents the LLM from being called when no relevant context exists. Without it, the model would hallucinate an answer from its training data.

**Used by:** main.py (`/query` endpoint)

---

### main.py -- FastAPI Application

The entry point that wires everything together.

**Startup (lifespan handler):**
1. Initialize Embedder (loads ONNX model, ~220 MB)
2. Initialize VectorStore (opens or creates Qdrant Edge shard)
3. Initialize Generator (loads GGUF model into RAM, ~2.1 GB)
4. Initialize Reranker (loads cross-encoder, ~80 MB)
5. Ingest baked-in corpus -- compares files in `EDGE_CORPUS_DIR` against already-indexed documents, ingests only new files

**Shutdown:** flush and close Qdrant Edge shard.

**Endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/query` | POST | Ask a question, get a grounded answer with sources |
| `/ingest` | POST | Upload document files (multipart), index them |
| `/health` | GET | Status, model loaded, chunk count, uptime |
| `/corpus` | GET | List all indexed documents with chunk counts |

Uses a factory pattern (`create_app()`) that accepts pre-built components for testing. In production, components are `None` and initialized during startup.

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
| Network | None (fully offline) |

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
| Xenova/ms-marco-MiniLM-L-6-v2 | Cross-encoder (ONNX) | 80 MB | Reranking | English-focused |
| CohereLabs/tiny-aya-global (Q4_K_M) | Generative LLM (GGUF) | 2.1 GB | Answer generation | 70+ |

## Quality Controls

| Mechanism | What it prevents |
|---|---|
| Score threshold (0.3) | Irrelevant chunks reaching the reranker |
| Cross-encoder reranker | Wrong chunks reaching the LLM |
| Empty-sources guard | Hallucination when no context exists |
| Grounding prompt | LLM adding information from its own knowledge |
| repeat_penalty (1.3) | Repetitive / looping output |
| Stop sequences | Multi-answer rambling |
| max_tokens (100) | Overly long, unfocused answers |
| Deterministic point IDs | Duplicate chunks on re-ingestion |
| Startup dedup check | Re-indexing already-ingested corpus files |

## Test Results

27 unit/integration tests + 1 end-to-end test with real models. Known-answer evaluation on 10 factual questions from the corpus achieved 10/10 accuracy after adding the cross-encoder reranker.
