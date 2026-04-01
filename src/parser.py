import subprocess
from pathlib import Path


# File extensions that are plain text — no parsing needed
_PLAINTEXT_EXTENSIONS = {".txt", ".md", ".text", ".markdown"}


def parse_file(path: Path) -> str:
    """Extract text from a document file.

    Plain text and markdown files are read directly.
    PDF, DOCX, and other formats are parsed via LiteParse (Bun subprocess).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() in _PLAINTEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")

    return _parse_with_liteparse(path)


def _parse_with_liteparse(path: Path) -> str:
    """Invoke LiteParse CLI via Bun to extract text from a document."""
    result = subprocess.run(
        ["bunx", "@llamaindex/liteparse", "parse", str(path), "--format", "text"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"LiteParse failed for {path}: {result.stderr}")
    return result.stdout.strip()
