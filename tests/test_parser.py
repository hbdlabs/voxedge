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


def test_parse_pdf_uses_python_binding_and_flips_y(tmp_path: Path, monkeypatch):
    """A PDF is parsed via the in-process liteparse binding; the binding's
    bottom-origin y is flipped to top-origin so header/footer filtering works."""
    import sys
    import types

    class FakeItem:
        def __init__(self, x, y, font_size, text):
            self.x, self.y, self.font_size, self.text = x, y, font_size, text

    class FakePage:
        height = 100.0
        # bottom-origin y: 95=top (header), 50=middle (body), 8=bottom (footer)
        text_items = [
            FakeItem(10, 95.0, 12.0, "HeaderLine"),
            FakeItem(10, 50.0, 12.0, "BodyLine"),
            FakeItem(10, 8.0, 12.0, "FooterLine"),
        ]

    class FakeResult:
        pages = [FakePage()]

    class FakeLiteParse:
        def __init__(self, **kwargs):
            assert kwargs.get("ocr_enabled") is False  # honors no-OCR policy

        def parse(self, _path):
            return FakeResult()

    fake = types.ModuleType("liteparse")
    fake.LiteParse = FakeLiteParse
    monkeypatch.setitem(sys.modules, "liteparse", fake)

    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4\n")
    out = parse_file(f)

    assert "BodyLine" in out          # middle of page survives
    assert "HeaderLine" not in out    # top margin stripped after y-flip
    assert "FooterLine" not in out    # bottom margin stripped after y-flip
