import pytest
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Provides a temporary data directory for tests."""
    return tmp_path


@pytest.fixture
def sample_text() -> str:
    return (
        "Vaccines are important for children under 5. "
        "The WHO recommends DPT, polio, and measles vaccines. "
        "These should be administered according to the national schedule. "
        "Consult your local health worker for the exact timing. "
        "Side effects are usually mild and temporary."
    )


@pytest.fixture
def sample_chunks() -> list[str]:
    return [
        "Vaccines are important for children under 5. The WHO recommends DPT, polio, and measles vaccines.",
        "The WHO recommends DPT, polio, and measles vaccines. These should be administered according to the national schedule.",
        "These should be administered according to the national schedule. Consult your local health worker for the exact timing.",
        "Consult your local health worker for the exact timing. Side effects are usually mild and temporary.",
    ]
