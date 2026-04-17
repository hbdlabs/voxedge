# GPU profile — lessons from Tier 3 validation

Notes captured during the first end-to-end run of `gemma-cuda` on a rented
RTX 4090. These should feed back into `Dockerfile.cuda`, the README, and
the spec before the next rental run or a real Spark deployment.

## 1. `onnxruntime` vs. `onnxruntime-gpu` — mutually hostile

Symptom: after `pip install "." onnxruntime-gpu`, FastEmbed reported only
`['AzureExecutionProvider', 'CPUExecutionProvider']`. CUDAExecutionProvider
wasn't even listed.

Cause: installing the VoxEdge package pulled in `onnxruntime` (CPU) as a
transitive dependency. The two packages share some files on disk and
Python imported the CPU one. Having both installed at the same time is
not supported by the ONNX Runtime maintainers.

What worked: `pip uninstall -y onnxruntime` then
`pip install --force-reinstall --no-deps onnxruntime-gpu`. The second
step is required because the first uninstall clobbers files shared with
`onnxruntime-gpu`, leaving that package half-broken.

**Fix for `Dockerfile.cuda`:** install `onnxruntime-gpu` *after* the
VoxEdge package, then explicitly remove `onnxruntime` (CPU) if it got
pulled in transitively. Or pin `fastembed` to a version whose
dependencies don't include plain `onnxruntime`. Or ship a constraints
file that pins `onnxruntime-gpu` and excludes `onnxruntime`.

## 2. `onnxruntime-gpu` needs cuDNN 9 at runtime

Symptom: after fixing #1, providers listed `['CUDAExecutionProvider',
'CPUExecutionProvider']`, and FastEmbed accepted `cuda=True` without
error. But `session.get_providers()` at runtime returned only
`['CPUExecutionProvider']`. The model silently ran on CPU while
`/info` still reported `embedder_device: cuda`.

Cause: `onnxruntime-gpu` 1.24.x requires **cuDNN 9** and CUDA 12.x.
The RunPod base image ships CUDA 12.4 runtime but no cuDNN. The
CUDAExecutionProvider library failed to load
(`libcudnn.so.9: cannot open shared object file`) and FastEmbed fell
back to CPU silently — logged a `RuntimeWarning`, not an error.

**Fix for `Dockerfile.cuda`:** either

- base the image on the `...cudnn-runtime-ubuntu24.04` variant of
  `nvcr.io/nvidia/cuda`, which bundles cuDNN 9, or
- install `libcudnn9-cuda-12` from NVIDIA's apt repo in the runtime
  stage.

The CUDA version of llama-cpp-python does NOT need cuDNN — that's why
the LLM worked on GPU even without it. cuDNN is only needed by
onnxruntime-gpu (for the embedder and reranker).

## 3. `/info` reports profile intent, not runtime reality

Symptom: `/info` showed `embedder_device: cuda, reranker_device: cuda`
even while those components were actually running on CPU.

Cause: `/info` echoes the profile's declared device, not what the
underlying ONNX session is actually using. When CUDA silently falls
back to CPU (see #2), there is nothing in the response to reveal it.

**Fix for `src/main.py` / `src/embedder.py` / `src/reranker.py`:**
expose a property like `embedder.active_providers` that returns
`session.get_providers()`. `/info` should report those actual
providers, not just the profile's declared intent.

Optionally add a startup assertion: if `profile.embedder_device ==
"cuda"` and `"CUDAExecutionProvider"` is not in the actual providers
list, fail loud. Better to crash at boot than silently serve CPU
performance under a profile named `gemma-cuda`.

## 4. Docker isn't available on RunPod pods

Symptom: `docker --version` → `command not found`.

Cause: RunPod pods are themselves Docker containers running on shared
hosts. The host's `docker` binary isn't exposed inside the pod. Docker
in Docker needs privileged mode, which community pods don't grant.

Implication: `deploy/docker/Dockerfile.cuda` can't be built on RunPod
directly. To validate the actual image, either

- use a RunPod template that enables DinD (rare),
- switch to a provider that gives a real VM (Lambda Labs, AWS, GCP), or
- validate by running the image's contents directly (what we did —
  install the same packages and run `uvicorn` straight) and accept
  that the Docker wrapper is thin.

Current Tier 3 validates the Python runtime path, not the Docker layer.

## 5. Pod boot timing and SSH access

Symptom: right after deploy, direct TCP SSH returned
`kex_exchange_identification: read: Connection reset by peer` for the
first ~1–2 minutes.

Cause: pod's sshd takes time to inject keys and start after the
container comes up.

Also: RunPod's *proxy* SSH (`ssh.runpod.io`) forces an interactive
shell and will not run a remote command argument — the argument is
discarded and an interactive session starts instead. Non-interactive
automation requires the **direct TCP** endpoint ("SSH over exposed
TCP" in the Connect tab).

## 6. GGUF + mmap on FUSE mounts

Symptom on Mac dev: trying to point `EDGE_MODEL_PATH` at a file
inside an OrbStack Docker image's exposed filesystem failed with
`Failed to load model from file` — even with `use_mmap=False`.

Cause: the OrbStack FUSE mount doesn't support the syscalls
llama.cpp needs to load the model, mmap or not.

Fix: copy the GGUF to a normal filesystem path (`~/models/`) for dev
testing. For Docker/k8s deployments this isn't an issue because the
model is already on a regular volume.

## 7. GGUF model architecture compatibility

Symptom: `gemma4` GGUF failed to load with
`unknown model architecture: 'gemma4'` on `llama-cpp-python 0.3.19`.

Cause: Gemma 4 architecture support landed in `llama-cpp-python` 0.3.20.
Our pyproject says `>=0.3.20`, but Mac venv had 0.3.19 cached from an
earlier install.

Fix: always `pip install --upgrade "llama-cpp-python>=0.3.20"` when
moving between model generations.

## 8. What Tier 3 actually validated

- ✅ Profile → backend → factory dispatch works end-to-end on CUDA.
- ✅ `cuda=True` is accepted and requested for FastEmbed at the API
  surface.
- ✅ `llama-cpp-python` with `GGML_CUDA=on` compiled and loaded all
  layers on the GPU (verified via `nvidia-smi`: ~2.5 GB resident in
  one process).
- ✅ `/chat`, `/translate`, `/query` (with an ingested corpus) all
  return correct, grounded answers, including cross-lingual
  (Norwegian and French questions → English source → answers in the
  question's language).
- ✅ Latency: `/chat` 60–540 ms, `/translate` 60–150 ms, full RAG
  `/query` 3.4–6.7 s (prefill-bound on 5-chunk context).
- ❌ **NOT validated:** embedder and reranker actually running on
  GPU. They silently fell back to CPU due to missing cuDNN 9.
  Needs a re-run after fixing #2.
- ❌ **NOT validated:** `Dockerfile.cuda` end-to-end (no Docker on the
  rental; see #4). Needs a non-RunPod provider or a local GPU box.
