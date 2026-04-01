from unittest.mock import MagicMock, patch
from src.generator import Generator, build_prompt


def test_build_prompt():
    """Prompt template includes context and question."""
    prompt = build_prompt(
        chunks=["Chunk 1 text.", "Chunk 2 text."],
        question="What is this about?",
    )
    assert "Chunk 1 text." in prompt
    assert "Chunk 2 text." in prompt
    assert "What is this about?" in prompt
    assert "Use ONLY the provided context" in prompt


def test_build_prompt_empty_chunks():
    """Prompt with no chunks still includes the question."""
    prompt = build_prompt(chunks=[], question="Hello?")
    assert "Hello?" in prompt


@patch("src.generator.Llama")
def test_generate_calls_model(mock_llama_cls):
    """Generator calls llama-cpp-python with the correct prompt."""
    mock_llm = MagicMock()
    mock_llm.create_completion.return_value = {
        "choices": [{"text": "The answer is 42."}]
    }
    mock_llama_cls.return_value = mock_llm

    gen = Generator(model_path="/fake/model.gguf")
    result = gen.generate(chunks=["context here"], question="What is it?")

    assert result == "The answer is 42."
    mock_llm.create_completion.assert_called_once()
