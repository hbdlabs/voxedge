# Running VoxEdge on RunPod (GPU)

A quick way to run VoxEdge on a **rented NVIDIA GPU** — to validate the CUDA path
(`aya-cuda` / `gemma-cuda` profiles) or to demo the GPU build without owning a
card. You **don't build a Docker image** here: the pod already has CUDA, so you
run VoxEdge directly in it. The slow part (compiling llama-cpp CUDA) is skipped
entirely by installing a **prebuilt CUDA wheel**.

> This is throwaway test tooling, not a production deployment path. For production
> GPU, use `deploy/docker/Dockerfile.cuda` (Gemma) on your own hardware / k8s.

## 1. Create the pod

In the RunPod console → **Deploy → Pods**:

- **GPU:** any modern card works. Cheapest sensible choice is an **RTX A4000/A5000**
  (Ampere, plenty for a 2–3 GB model). A10, A40, 4090, A100, and even **Blackwell
  (RTX PRO)** also work — see the PTX note below.
- **Template:** one with the **CUDA toolkit** — a *PyTorch 2.x / CUDA 12.x `-devel`*
  image, e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
- **Container disk:** ≥ 25 GB.
- **Expose HTTP port 8080.**

Then **Connect → Web Terminal** (or SSH).

## 2. Get a GitHub token

The repo is private, so you need a token with read access. Create a **fine-grained
PAT** scoped to just `hbdlabs/voxedge` (Resource owner → `hbdlabs`, Contents:
Read-only, short expiry) at github.com/settings/personal-access-tokens/new — or a
classic token with `repo` scope. **Delete it after the test.**

## 3. Run

```bash
export GH_TOKEN=ghp_xxxxxxxx
export CUDA_TAG=cu124          # match the pod's CUDA — run `nvidia-smi` (cu121/cu122/cu124)
git clone https://$GH_TOKEN@github.com/hbdlabs/voxedge.git
cd voxedge
bash deploy/runpod/cuda_smoke_test.sh
```

The script clones nothing extra; it installs deps + a prebuilt CUDA
`llama-cpp-python` wheel + `onnxruntime-gpu`, downloads tiny-aya, drops in a small
text corpus, and starts on the `aya-cuda` profile. (If you already cloned, just run
the script — it re-clones into `./voxedge`, so run it from the parent dir, or run
the steps inline.)

## 4. Verify CUDA is actually engaged

In a second terminal:

```bash
curl localhost:8080/info
```

You want:

- `"backend": "llama_cuda"`, `"n_gpu_layers": -1` → LLM on the GPU
- `CUDAExecutionProvider` in both `embedder_active_providers` and
  `reranker_active_providers` → fastembed on the GPU

If those show **`CPUExecutionProvider` only**, cuDNN wasn't found — see the cuDNN
note below.

Then a real query:

```bash
curl -XPOST localhost:8080/query -H 'content-type: application/json' \
  -d '{"question":"How do I prevent malaria?"}'
```

## Gotchas (learned the hard way)

- **cuDNN on `-devel` images.** A CUDA `-devel` template ships the toolkit but
  often **not** system cuDNN, which `onnxruntime-gpu` needs. The script installs
  `onnxruntime-gpu` *with* its deps (pulling `nvidia-cudnn-cu12`) and adds them to
  `LD_LIBRARY_PATH`. Without that, the embedder/reranker silently run on CPU and
  `/info` lies about it.
- **Blackwell (RTX PRO) + older CUDA.** Blackwell is `sm_120`, newer than the
  prebuilt wheel targets and newer than CUDA 12.4's `nvcc`. It still works via
  **PTX JIT** — the first call compiles kernels on the fly (~60 s warmup, then
  cached). A fallback *compile* would fail on CUDA 12.4 (can't target sm_120), so
  rely on the prebuilt wheel + PTX, or use a CUDA 12.8 template.
- **First query is slow, then ~1–7 s.** That's the one-time JIT warmup for each
  model (LLM, embedder, reranker). Subsequent queries are fast.
- **Match `CUDA_TAG` to the pod.** `nvidia-smi` shows the CUDA version; set
  `CUDA_TAG=cu121/cu122/cu124` accordingly. If no matching prebuilt wheel exists,
  the script falls back to compiling (slower).

## Trying other configs on the pod

- **Gemma instead of tiny-aya:** download the Gemma 4 GGUF and restart with
  `EDGE_MODEL_PROFILE=gemma-cuda EDGE_MODEL_PATH=<gemma.gguf> EDGE_N_CTX=8192`.
- **Bigger answers:** `EDGE_MAX_TOKENS=400` (the kiosk default of 100 is short).
- **Cross-lingual corpus:** set `EDGE_TRANSLATE_QUERIES=true` so non-English
  queries are translated into the corpus language before retrieval (otherwise a
  Vietnamese/Spanish query can retrieve the wrong English doc).
- **Concurrency:** the server serializes LLM calls (one model context), so
  parallel `/query` requests queue rather than run in parallel — expected.

## Tear down

GPU pods bill **per second**. Stop/terminate the pod when done, and **delete the
GitHub token** you created.
