"""End-to-end test with real model. Requires GGUF model at ~/models/tiny-aya-global-q4_k_m.gguf"""

import os
from pathlib import Path

import pytest

from src.embedder import Embedder
from src.generator import Generator
from src.store import VectorStore
from src.ingest import ingest_file
from src.query import query_brain

MODEL_PATH = os.path.expanduser("~/models/tiny-aya-global-q4_k_m.gguf")
pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="GGUF model not found at ~/models/tiny-aya-global-q4_k_m.gguf",
)


def test_full_e2e(tmp_path: Path):
    """Ingest a document, query it, get a real generated answer."""

    # 1. Write a test document
    doc = tmp_path / "health_guide.txt"
    doc.write_text(
        "Malaria is a serious disease spread by mosquitoes. "
        "Prevention includes sleeping under insecticide-treated bed nets, "
        "using mosquito repellent, and taking antimalarial medication. "
        "Symptoms include fever, chills, headache, and body aches. "
        "If you suspect malaria, visit your nearest health clinic immediately. "
        "Early treatment with artemisinin-based combination therapy is effective."
    )

    # 2. Set up real components
    embedder = Embedder()
    store = VectorStore(path=str(tmp_path / "qdrant"), vector_size=384)
    generator = Generator(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)

    # 3. Ingest
    result = ingest_file(doc, embedder, store, chunk_size=300, chunk_overlap=50)
    assert result.chunks > 0
    print(f"\nIngested: {result.chunks} chunks, language={result.language}")

    # 4. Query in English
    answer = query_brain(
        question="How can I prevent malaria?",
        embedder=embedder,
        store=store,
        generator=generator,
        top_k=3,
        score_threshold=0.0,
        max_tokens=200,
    )

    print(f"\nQuestion: How can I prevent malaria?")
    print(f"Answer: {answer.answer}")
    print(f"Sources: {answer.sources}")
    print(f"Language: {answer.language}")

    assert len(answer.answer) > 0
    assert len(answer.sources) > 0

    # 5. Query in Spanish
    answer_es = query_brain(
        question="Como puedo prevenir la malaria?",
        embedder=embedder,
        store=store,
        generator=generator,
        top_k=3,
        score_threshold=0.0,
        max_tokens=200,
    )

    print(f"\nQuestion: Como puedo prevenir la malaria?")
    print(f"Answer: {answer_es.answer}")
    print(f"Language: {answer_es.language}")

    assert len(answer_es.answer) > 0

    store.close()
