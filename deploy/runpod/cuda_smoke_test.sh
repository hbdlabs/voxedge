#!/usr/bin/env bash
# voxedge CUDA smoke test on a RunPod GPU pod.
#
# Verifies the `aya-cuda` profile actually runs the LLM (llama.cpp) AND fastembed
# (embedder + reranker) on the GPU. Works on any Ampere/Ada GPU (A5000/A10/A40/
# 4090/A100) — our CUDA arch pin covers 80;86;89.
#
# Use a RunPod template that has the CUDA *toolkit* + cuDNN, e.g. a PyTorch 2.x /
# CUDA 12.x image. Expose HTTP port 8080. Then paste this in the pod terminal.
set -euo pipefail

# --- 0) Code: clone the private repo (PAT with read access to hbdlabs/voxedge) ---
: "${GH_TOKEN:?export GH_TOKEN=<github PAT with repo read> first}"
git clone "https://${GH_TOKEN}@github.com/hbdlabs/voxedge.git"
cd voxedge

# --- 1) Base python deps (fastembed, qdrant-edge, fastapi, llama-cpp CPU placeholder) ---
pip install -U pip
pip install -e .

# --- 2) GPU llama.cpp: prefer the prebuilt CUDA wheel (no compile). Falls back to
#        a fast native compile if that exact version isn't published for this CUDA. ---
CUDA_TAG="${CUDA_TAG:-cu124}"   # cu121 / cu122 / cu124 — match the pod's CUDA
pip install --force-reinstall --no-deps "llama-cpp-python==0.3.23" \
    --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/${CUDA_TAG}" \
  || CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80;86;89" \
       pip install --force-reinstall --no-deps "llama-cpp-python==0.3.23"

# --- 3) GPU fastembed: swap CPU onnxruntime for the GPU build (needs cuDNN, which
#        the PyTorch/CUDA templates already have) ---
pip uninstall -y onnxruntime || true
pip install --force-reinstall --no-deps onnxruntime-gpu

# --- 4) Model ---
mkdir -p /data/models
curl -L -o /data/models/tiny-aya-global-q4_k_m.gguf \
  "https://huggingface.co/CohereLabs/tiny-aya-global-GGUF/resolve/main/tiny-aya-global-q4_k_m.gguf"

# --- 5) Plain-text corpus so /query works without bun/liteparse (a .txt is read
#        directly; the baked PDF would need bun). Embedder+reranker still GPU. ---
rm -rf data/corpus && mkdir -p data/corpus
cat > data/corpus/health.txt <<'EOF'
Malaria is spread by mosquitoes. Prevention includes insecticide-treated bed nets,
repellent, and antimalarial medication. Symptoms include fever, chills, and headache.
EOF

# --- 6) Run on the GPU (aya-cuda profile) ---
export EDGE_MODEL_PROFILE=aya-cuda
export EDGE_MODEL_PATH=/data/models/tiny-aya-global-q4_k_m.gguf
export EDGE_MODE=full
export EDGE_RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
export EDGE_QDRANT_DIR=/data/qdrant
export EDGE_CACHE_DIR=/data/model_cache
echo ">>> starting voxedge on GPU — once up, in another shell run:"
echo "    curl localhost:8080/info        # confirm CUDAExecutionProvider + n_gpu_layers=-1"
echo "    curl -XPOST localhost:8080/query -H 'content-type: application/json' -d '{\"question\":\"How do I prevent malaria?\"}'"
uvicorn src.main:app --host 0.0.0.0 --port 8080
