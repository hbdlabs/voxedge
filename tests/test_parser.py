import pytest
from pathlib import Path
from src.parser import parse_file


def test_parse_txt_file(tmp_path: Path):
    """Plain text files are returned as-is."""
    f = tmp_path / "test.txt"
    f.write_text("Hello, this is a test document.")
    result = parse_file(f)
    assert result == "Hello, this is a test document."


def test_parse_md_file(tmp_path: Path):
    """Markdown files are returned as-is."""
    f = tmp_path / "test.md"
    f.write_text("# Title\n\nSome content here.")
    result = parse_file(f)
    assert result == "# Title\n\nSome content here."


def test_parse_nonexistent_file(tmp_path: Path):
    """Nonexistent file raises FileNotFoundError."""
    f = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError):
        parse_file(f)
