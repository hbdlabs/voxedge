# GPU-accelerated profile

## Goal

Add an optional GPU-accelerated deployment path to VoxEdge that uses the same container shape as the Pi, the same API, the same corpus, and the same Gemma 4 E2B Q4 model — but runs embedder, reranker, and LLM on the GPU. Target hardware: NVIDIA DGX Spark (primary), any CUDA GPU ≥ 6 GB VRAM for testing (Gemma 4 + Jina reranker + embedder + KV cache). Apple Silicon (Metal) is supported as a local dress rehearsal on Mac.

Pi deployment must remain bit-identical — no regressions, no new required dependencies on the Pi image.

## Non-goals

- No client/server split. One container, one box, same as today.
- No change to API surface (`/query`, `/ingest`, `/chat`, `/translate`, `/info`, `/health`, `/corpus`).
- No change to Qdrant, LiteParse, or ingestion pipeline.
- No new model sizes. Gemma 4 E2B Q4_K_M GGUF everywhere — only the runtime changes.
- No vLLM, no TensorRT-LLM, no HF downloads, no NVFP4. All backends are llama.cpp with different compile flags.

## Design

### 1. Profile as single source of truth

`ModelProfile` gains four runtime fields:

```
backend:          "llama_cpu" | "llama_metal" | "llama_cuda"
embedder_device:  "cpu" | "cuda"
reranker_device:  "cpu" | "cuda"
n_gpu_layers:     int | None   # -1 for GPU backends, None for cpu
```

All other fields (templates, stops, temperature, `use_chat_api`, `n_ctx_default`, `patches`) stay exactly as they are today. `EDGE_MODEL_PROFILE` remains the only selector. No new env vars.

### 2. Three pluggable components behind protocols

```
generator.py:  Generator   { LlamaCpuGenerator, LlamaMetalGenerator, LlamaCudaGenerator }
embedder.py:   Embedder    { FastEmbedCpu, FastEmbedCuda }
reranker.py:   Reranker    { FastEmbedCrossEncoderCpu, FastEmbedCrossEncoderCuda }
```

Each module exports `build(profile) -> Component`. `main.py` calls the three builders at startup and is otherwise unchanged. Existing CPU classes keep their current behaviour and are the default for `aya` and `gemma` profiles.

**Constructor drift to resolve in PR 1.** Current `Embedder` and `Reranker` constructors take `(model_name, cache_dir)`. CUDA variants need to pass `providers=["CUDAExecutionProvider"]` through to FastEmbed. Two paths:

- If the pinned FastEmbed version forwards `providers=` to ONNX Runtime, add a `providers` kwarg and done.
- If it doesn't, the CUDA impl bypasses FastEmbed and calls `onnxruntime` directly (loading the same ONNX model from FastEmbed's cache), or swaps to `sentence-transformers` on torch/CUDA.

A short spike (throwaway branch, not PR 1 itself) confirms which path is viable before PR 1 lands. The spike installs `onnxruntime-gpu` in a scratch venv and tries both approaches; its result pins the kwarg signature in PR 1's protocol.

### 3. Profile table

| Profile | Backend | Embedder | Reranker | Target |
|---|---|---|---|---|
| `aya` | llama_cpu | CPU | CPU | Pi (unchanged) |
| `gemma` | llama_cpu | CPU | CPU | Pi (unchanged) |
| `gemma-metal` | llama_metal | CPU | CPU | Mac dev dress rehearsal |
| `gemma-cuda` | llama_cuda | CUDA | CUDA | Rental GPU / Spark |

All profiles share the same Gemma 4 E2B Q4_K_M GGUF file (3.1 GB). GPU profiles set `n_gpu_layers=-1` to offload every layer. `gemma-cuda` uses the Jina v2 multilingual reranker by default — VRAM is plentiful and it handles non-English corpora correctly.

### 4. GPU memory layout

All three components load at startup and stay resident. Approximate footprint on the GPU with `gemma-cuda`:

| Component | Footprint |
|---|---|
| Embedder (multilingual MiniLM L12) | ~0.5 GB |
| Reranker (jina-reranker-v2 multilingual) | ~1.2 GB |
| LLM (Gemma 4 E2B Q4) | ~3.1 GB |
| KV cache (8 K ctx) | small, well under 1 GB |

Minimum ~6 GB VRAM with Jina reranker; drops to ~4 GB if you swap to the MiniLM English reranker. Trivial on Spark's 128 GB unified pool. Components take turns using the compute but never unload.

### 5. Container images

```
deploy/docker/
  Dockerfile.aya     # unchanged
  Dockerfile.gemma   # unchanged
  Dockerfile.cuda    # new
```

`Dockerfile.cuda`:

- Base `nvcr.io/nvidia/cuda:12.6-runtime-ubuntu24.04`
- Install `onnxruntime-gpu` for FastEmbed CUDA EP
- Rebuild `llama-cpp-python` with `CMAKE_ARGS="-DGGML_CUDA=on"` — this compiles CUDA kernels from source, multi-minute build, large intermediate layer. Use a multi-stage build to keep the final image lean and cache the build layer aggressively.
- Bakes the same Gemma 4 GGUF as `Dockerfile.gemma`
- Defaults `EDGE_MODEL_PROFILE=gemma-cuda`

Metal support is a dev-extra only (`pip install -e ".[metal]"` with llama-cpp-python rebuilt `-DGGML_METAL=on`). No Docker image for Metal — it's a local dev path.

### 6. Kubernetes overlay

```
deploy/k8s/
  ...                    # existing layout unchanged
  overlays/gpu/          # new — adds resources.limits."nvidia.com/gpu": 1
                         #       larger memory/storage requests
                         #       env EDGE_MODEL_PROFILE=gemma-cuda
```

### 7. Config changes

`config.py` gains nothing new. Profile fields reached via `get_profile(settings.model_profile)`. `EDGE_MODEL_PATH` is honoured by all backends.

### 8. `/info` reports backend

`/info` gains a small block so an operator can verify on the running container where each stage actually runs:

```
"runtime": {
  "backend": "llama_cuda",
  "n_gpu_layers": -1,
  "embedder_device": "cuda",
  "reranker_device": "cuda"
}
```

Values come straight from the active profile.

## Test plan

### Tier 1 — no hardware (Mac, CI)

- Unit: each `build(profile)` returns an instance of the declared backend class.
- Unit: `Generator` / `Embedder` / `Reranker` protocol conformance for every implementation.
- Regression: existing test suite passes unchanged under `aya` and `gemma` profiles.
- Lint: every `backend` value in the profile table has a registered builder.

### Tier 2 — Mac GPU (Metal dress rehearsal)

- Install dev extra: `pip install -e ".[metal]"`.
- Run `EDGE_MODEL_PROFILE=gemma-metal uvicorn src.main:app` on the Mac.
- Pass criteria: `/query`, `/chat`, `/translate` return sensible output; no correctness delta vs `gemma` profile on the same corpus; `/info` reports `backend: llama_metal`.
- Observation (not a gate): record tok/s vs `gemma` for reference.

### Tier 3 — rented CUDA GPU

- Rent a card with ≥ 6 GB VRAM for full Jina reranker parity (3060, 3090, T4-16GB, etc. — ~$0.20–0.50/hr on RunPod or Vast.ai). A 4 GB card works only with the MiniLM reranker.
- On the rental: `git clone`, `docker build -f deploy/docker/Dockerfile.cuda -t voxedge:cuda .`
- `docker run --gpus all -p 8080:8080 -v $PWD/data:/data voxedge:cuda`
- From Mac: `ssh -L 8080:localhost:8080 user@rental`, then run `tests/test_e2e.py` against `http://localhost:8080`.
- Pass criteria: `/info` reports embedder on `cuda`, reranker on `cuda`, LLM with all layers offloaded; `nvidia-smi` shows one Python process resident holding all three model weights; `/query` on the sample corpus returns a correct grounded answer; end-to-end latency materially lower than the `gemma` CPU baseline on the same rental box.

### Tier 4 — real Spark (deferred)

- Same `Dockerfile.cuda`, same `gemma-cuda` profile. Run with `--gpus all`.
- Only new thing validated here: the CUDA 12.6 runtime base works on Spark's Blackwell driver. Everything else was proven in Tier 3.

## Rollout

1. **PR 1 — abstraction only.** Add protocols, refactor existing classes behind them, wire the factory. No new backends, no new profiles, no new deps. Pi tests must pass unchanged. *Only risky PR — everything after is additive.*
2. **PR 2 — Metal backend + `gemma-metal` profile.** Dev extra only, no Docker image.
3. **PR 3 — CUDA backends + `gemma-cuda` profile + `Dockerfile.cuda` + k8s `gpu` overlay.** Tested on rented GPU per Tier 3.

## Open questions

- Bake the same Gemma 4 GGUF into `Dockerfile.cuda`, or mount from a volume for parity with Spark field deployments? Default: bake it — simpler ops, Pi already does this.
- CUDA 12.6 runtime on Spark's driver — should work via forward compat, but untested here. Verify at Tier 4; if it fails, pin to whatever CUDA version the Spark base image supports.
- FastEmbed `providers=` kwarg forwarding — must be confirmed in PR 1 spike before committing to the FastEmbed-CUDA path vs a direct ONNX Runtime or sentence-transformers fallback.
