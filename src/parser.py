import json
import subprocess
from pathlib import Path


_PLAINTEXT_EXTENSIONS = {".txt", ".md", ".text", ".markdown"}

# Lines with font sizes at or below this are likely OCR artifacts from screenshots
_MIN_FONT_SIZE = 8

# Footer/header region (within this many points of page top/bottom)
_MARGIN_Y = 60


def parse_file(path: Path) -> str:
    """Extract text from a document file.

    Plain text and markdown files are read directly.
    PDF, DOCX, and other formats are parsed via LiteParse with spatial layout
    reconstruction to handle garbled tables and screenshots.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() in _PLAINTEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")

    return _parse_with_liteparse(path)


def _parse_with_liteparse(path: Path) -> str:
    """Parse a document using LiteParse JSON mode and reconstruct clean text."""
    result = subprocess.run(
        ["bunx", "@llamaindex/liteparse", "parse", str(path), "--format", "json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"LiteParse failed for {path}: {result.stderr}")

    data = json.loads(result.stdout)
    return _reconstruct_from_spatial(data)


def _reconstruct_from_spatial(data: dict) -> str:
    """Reconstruct clean text from LiteParse spatial JSON output.

    Groups text items into lines by Y position, filters out:
    - Headers/footers (near page edges)
    - Garbled OCR regions (tiny/mixed font sizes)
    Joins remaining lines into clean text.
    """
    all_lines = []

    for page in data.get("pages", []):
        page_height = page.get("height", 842)
        items = page.get("textItems", [])
        if not items:
            continue

        # Sort by Y then X
        items = sorted(items, key=lambda i: (round(i["y"], 0), i["x"]))

        # Group into lines (items within 5px vertically = same line)
        lines = []
        current_line = []
        current_y = -999

        for item in items:
            if abs(item["y"] - current_y) > 5:
                if current_line:
                    lines.append((current_y, current_line))
                current_line = []
                current_y = item["y"]
            current_line.append(item)
        if current_line:
            lines.append((current_y, current_line))

        # Filter and reconstruct each line
        for y, line_items in lines:
            # Skip header/footer regions
            if y < _MARGIN_Y or y > page_height - _MARGIN_Y:
                continue

            # Check font sizes in this line
            font_sizes = [item.get("fontSize", 12) for item in line_items]
            avg_font = sum(font_sizes) / len(font_sizes)
            distinct_sizes = len(set(round(s) for s in font_sizes))

            # Skip garbled OCR lines:
            # - tiny font (< 8px) = OCR'd screenshot text
            # - mixed font sizes with small avg = table/UI artifacts
            if avg_font < _MIN_FONT_SIZE:
                continue
            if distinct_sizes > 2 and avg_font < 10:
                continue

            # Reconstruct line text
            text = " ".join(item["text"].strip() for item in line_items if item["text"].strip())
            if not text:
                continue

            # Skip bare numbers (orphaned table cell values)
            stripped = text.replace(" ", "").replace(",", "").replace(".", "")
            if stripped.isdigit() and len(text) < 20:
                continue

            all_lines.append(text)

    return "\n".join(all_lines)
