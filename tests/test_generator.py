from unittest.mock import MagicMock, patch
from src.profiles import get_profile


def test_build_prompt_aya():
    """Aya profile prompt includes context and question."""
    from src.generator import build_prompt

    profile = get_profile("aya")
    prompt = build_prompt(
        profile=profile,
        chunks=["Chunk 1 text.", "Chunk 2 text."],
        question="What is this about?",
    )
    assert "Chunk 1 text." in prompt
    assert "Chunk 2 text." in prompt
    assert "What is this about?" in prompt
    assert "ONLY use information from the Context below" in prompt


def test_build_prompt_gemma():
    """Gemma profile prompt uses turn markers."""
    from src.generator import build_prompt

    profile = get_profile("gemma")
    prompt = build_prompt(
        profile=profile,
        chunks=["Some context."],
        question="What?",
    )
    assert "<start_of_turn>user" in prompt
    assert "Some context." in prompt
    assert "What?" in prompt
    assert "<start_of_turn>model" in prompt


def test_build_prompt_empty_chunks():
    """Prompt with no chunks still includes the question."""
    from src.generator import build_prompt

    profile = get_profile("aya")
    prompt = build_prompt(profile=profile, chunks=[], question="Hello?")
    assert "Hello?" in prompt


@patch("src.generator.Llama")
def test_generate_calls_model(mock_llama_cls):
    """Generator calls llama-cpp-python with the correct prompt."""
    from src.generator import Generator

    mock_llm = MagicMock()
    mock_llm.create_completion.return_value = {
        "choices": [{"text": "The answer is 42."}]
    }
    mock_llama_cls.return_value = mock_llm

    profile = get_profile("aya")
    gen = Generator(model_path="/fake/model.gguf", profile=profile)
    result = gen.generate(chunks=["context here"], question="What is it?")

    assert result == "The answer is 42."
    mock_llm.create_completion.assert_called_once()
