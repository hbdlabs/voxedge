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
    """Gemma profile prompt includes context and question."""
    from src.generator import build_prompt

    profile = get_profile("gemma")
    prompt = build_prompt(
        profile=profile,
        chunks=["Some context."],
        question="What?",
    )
    assert "Some context." in prompt
    assert "What?" in prompt
    assert "Answer:" in prompt


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


@patch("src.generator.Llama")
def test_generate_serializes_concurrent_calls(mock_llama_cls):
    """Concurrent generations must not overlap in the LLM.

    llama.cpp is not thread-safe; without the lock, parallel requests race the
    shared context and crash the process (GGML_ASSERT). This proves the lock
    keeps at most one caller inside the model at a time.
    """
    import threading
    import time

    from src.generator import Generator

    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def fake_completion(**_kwargs):
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.02)  # hold the "model" so overlaps would be observable
        with guard:
            state["active"] -= 1
        return {"choices": [{"text": "ok"}]}

    mock_llm = MagicMock()
    mock_llm.create_completion.side_effect = fake_completion
    mock_llama_cls.return_value = mock_llm

    gen = Generator(model_path="/fake/model.gguf", profile=get_profile("aya"))

    threads = [
        threading.Thread(target=gen.generate, args=(["ctx"], "q?"))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mock_llm.create_completion.call_count == 8
    assert state["max"] == 1, f"lock failed: {state['max']} concurrent calls in the LLM"
