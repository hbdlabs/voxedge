"""Tests for the build(profile, ...) factories on generator/embedder/reranker.

These tests verify the factories dispatch correctly on profile runtime fields
without actually instantiating heavy model objects.
"""
from dataclasses import replace
from unittest.mock import patch

import pytest

from src import embedder as embedder_module
from src import generator as generator_module
from src import reranker as reranker_module
from src.profiles import GEMMA


def test_generator_build_accepts_llama_cpu():
    with patch.object(generator_module, "Generator") as mock_gen:
        generator_module.build(profile=GEMMA, model_path="/fake.gguf")
        mock_gen.assert_called_once()


def test_generator_build_rejects_unknown_backend():
    bad = replace(GEMMA, backend="nonsense")
    with pytest.raises(ValueError, match="Unsupported backend"):
        generator_module.build(profile=bad, model_path="/fake.gguf")


def test_generator_build_accepts_llama_metal_and_cuda():
    for backend in ("llama_metal", "llama_cuda"):
        profile = replace(GEMMA, backend=backend)
        with patch.object(generator_module, "Generator") as mock_gen:
            generator_module.build(profile=profile, model_path="/fake.gguf")
            mock_gen.assert_called_once()


def test_embedder_build_cpu_does_not_pass_cuda():
    with patch.object(embedder_module, "Embedder") as mock_emb:
        embedder_module.build(profile=GEMMA, model_name="m")
        mock_emb.assert_called_once_with(model_name="m", cache_dir=None, cuda=False)


def test_embedder_build_cuda_passes_cuda_true():
    profile = replace(GEMMA, embedder_device="cuda")
    with patch.object(embedder_module, "Embedder") as mock_emb:
        embedder_module.build(profile=profile, model_name="m")
        mock_emb.assert_called_once_with(model_name="m", cache_dir=None, cuda=True)


def test_embedder_build_rejects_unknown_device():
    bad = replace(GEMMA, embedder_device="tpu")
    with pytest.raises(ValueError, match="Unsupported embedder_device"):
        embedder_module.build(profile=bad, model_name="m")


def test_reranker_build_cpu_does_not_pass_cuda():
    with patch.object(reranker_module, "Reranker") as mock_rer:
        reranker_module.build(profile=GEMMA, model_name="m")
        mock_rer.assert_called_once_with(model_name="m", cache_dir=None, cuda=False)


def test_reranker_build_cuda_passes_cuda_true():
    profile = replace(GEMMA, reranker_device="cuda")
    with patch.object(reranker_module, "Reranker") as mock_rer:
        reranker_module.build(profile=profile, model_name="m")
        mock_rer.assert_called_once_with(model_name="m", cache_dir=None, cuda=True)


def test_reranker_build_rejects_unknown_device():
    bad = replace(GEMMA, reranker_device="tpu")
    with pytest.raises(ValueError, match="Unsupported reranker_device"):
        reranker_module.build(profile=bad, model_name="m")


def test_default_profiles_have_cpu_runtime_fields():
    """Ensure aya and gemma keep today's Pi-identical CPU defaults."""
    from src.profiles import AYA, GEMMA as _GEMMA
    for p in (AYA, _GEMMA):
        assert p.backend == "llama_cpu"
        assert p.embedder_device == "cpu"
        assert p.reranker_device == "cpu"
        assert p.n_gpu_layers == 0


def test_gemma_metal_profile():
    """gemma-metal offloads all layers, embedder/reranker stay on CPU."""
    from src.profiles import get_profile
    p = get_profile("gemma-metal")
    assert p.backend == "llama_metal"
    assert p.n_gpu_layers == -1
    assert p.embedder_device == "cpu"
    assert p.reranker_device == "cpu"
    # Prompt templates and stops must match gemma — only runtime differs.
    from src.profiles import GEMMA
    assert p.rag_template == GEMMA.rag_template
    assert p.chat_template == GEMMA.chat_template
    assert p.translate_template == GEMMA.translate_template
