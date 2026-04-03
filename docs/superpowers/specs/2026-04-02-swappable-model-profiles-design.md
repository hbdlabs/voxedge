# Swappable Model Profiles

## Goal

Make the LLM interchangeable by defining model-specific behavior (prompt templates, generation parameters, stop sequences, patches) in typed profiles. Switching models means changing one Dockerfile line and one env var.

## Current State

`generator.py` has Tiny Aya behavior hardcoded:
- Jinja2 loopcontrols monkey-patch (Aya GGUF embeds a chat template using `{% break %}`)
- Raw chat format
- Specific prompt templates for RAG, chat, and translate
- Specific stop sequences (`\nAnswer:`, `\nQuestion:`, etc.)
- `repeat_penalty=1.3`, `temperature=0.3`

Adding a second model (Gemma 4 E2B) requires these to be different per model.

## Design

### ModelProfile dataclass

New file `src/profiles.py`:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    name: str
    rag_template: str
    chat_template: str
    translate_template: str
    stop_rag: list[str] = field(default_factory=list)
    stop_chat: list[str] = field(default_factory=list)
    stop_translate: list[str] = field(default_factory=list)
    temperature: float = 0.3
    repeat_penalty: float = 1.3
    translate_repeat_penalty: float = 1.5
    chat_format: str = "raw"
    n_ctx_default: int = 4096
    patches: list[str] = field(default_factory=list)
```

### Aya profile

```python
AYA = ModelProfile(
    name="aya",
    rag_template="""You are a helpful assistant at a community knowledge kiosk.

Rules:
- ONLY use information from the Context below
- If the Context does not answer the question, reply ONLY with: "I don't have information about that."
- Do NOT make up information
- Do NOT add information from your own knowledge
- Answer in the same language as the question
- Keep your answer short and direct

Context:
{context}

Question: {question}

Answer:""",
    chat_template="{system}\n\nUser: {message}\n\nAssistant:",
    translate_template="Translate from {source} to {target}.\n\n{source}: {text}\n\n{target}:",
    stop_rag=["\nQuestion:", "\n\n\n", "\nNote:", "(Note:", "\nAnswer:"],
    stop_chat=["\nUser:", "\n\n\n"],
    stop_translate=[],  # dynamic, built from source/target lang names
    temperature=0.3,
    repeat_penalty=1.3,
    translate_repeat_penalty=1.5,
    chat_format="raw",
    n_ctx_default=4096,
    patches=["jinja2_loopcontrols"],
)
```

### Gemma profile

```python
GEMMA = ModelProfile(
    name="gemma",
    rag_template="""<start_of_turn>user
You are a helpful assistant. Use ONLY the provided context to answer. If the context does not contain the answer, say "I don't have information about that." Answer in the same language as the question. Keep your answer short.

Context:
{context}

Question: {question}<end_of_turn>
<start_of_turn>model
""",
    chat_template="<start_of_turn>user\n{system}\n{message}<end_of_turn>\n<start_of_turn>model\n",
    translate_template="<start_of_turn>user\nTranslate from {source} to {target}. Output ONLY the translation.\n\n{text}<end_of_turn>\n<start_of_turn>model\n",
    stop_rag=["<end_of_turn>"],
    stop_chat=["<end_of_turn>"],
    stop_translate=["<end_of_turn>"],
    temperature=0.3,
    repeat_penalty=1.1,
    translate_repeat_penalty=1.1,
    chat_format="raw",
    n_ctx_default=8192,
    patches=[],
)
```

Note: The Gemma prompt format uses its native turn markers. The exact format will be verified against the GGUF during implementation — llama-cpp-python may handle the chat template automatically, in which case we use `chat_format="gemma"` instead of raw templates.

### Profile registry

```python
PROFILES = {
    "aya": AYA,
    "gemma": GEMMA,
}

def get_profile(name: str) -> ModelProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown model profile: {name}. Available: {list(PROFILES.keys())}")
    return PROFILES[name]
```

### Changes to generator.py

- Remove hardcoded prompt templates and move to profile
- Remove hardcoded Jinja2 patch — apply conditionally based on `profile.patches`
- Constructor takes `ModelProfile` parameter
- `generate()`, `chat()`, `translate()` use profile templates and stop sequences
- Build methods use `profile.rag_template.format(context=..., question=...)` etc.

### Changes to config.py

Add one setting:

```python
model_profile: str = "aya"
```

### Changes to main.py

Load profile at startup, pass to Generator:

```python
from src.profiles import get_profile

profile = get_profile(settings.model_profile)
generator = Generator(
    model_path=settings.model_path,
    profile=profile,
    n_ctx=settings.n_ctx or profile.n_ctx_default,
    n_threads=settings.n_threads,
)
```

### Dockerfiles

Move current `Dockerfile` to `deploy/docker/Dockerfile.aya`.

Create `deploy/docker/Dockerfile.gemma` — identical except:
- Downloads Gemma 4 E2B GGUF from `unsloth/gemma-4-E2B-it-GGUF`
- Sets `ENV EDGE_MODEL_PROFILE=gemma`
- Sets `ENV EDGE_MODEL_PATH=/data/models/gemma-4-e2b-q4_k_m.gguf`

Root `Dockerfile` stays as-is (Aya) for backward compatibility.

### /info endpoint

Add `model_profile` to the info response so operators can see which profile is active.

## Files changed

- Create: `src/profiles.py`
- Modify: `src/generator.py` — use profile instead of hardcoded values
- Modify: `src/config.py` — add `model_profile` setting
- Modify: `src/main.py` — load profile, pass to generator, show in /info
- Create: `deploy/docker/Dockerfile.aya`
- Create: `deploy/docker/Dockerfile.gemma`
- Modify: `tests/test_generator.py` — pass profile in tests

## What doesn't change

- `embedder.py`, `store.py`, `reranker.py`, `parser.py`, `chunker.py`, `ingest.py`, `query.py`
- API endpoints and their request/response shapes
- Existing `Dockerfile` in root
