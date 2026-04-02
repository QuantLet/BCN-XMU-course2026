from __future__ import annotations

from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"


def test_all_markdown_docs_have_bilingual_pairs() -> None:
    markdown_files = sorted(path.name for path in DOCS_DIR.glob("*.md"))
    missing_pairs: list[str] = []

    for filename in markdown_files:
        path = DOCS_DIR / filename
        stem = path.stem
        if stem.endswith("_zh"):
            counterpart = DOCS_DIR / f"{stem[:-3]}.md"
        else:
            counterpart = DOCS_DIR / f"{stem}_zh.md"
        if not counterpart.exists():
            missing_pairs.append(f"{filename} -> missing {counterpart.name}")

    assert not missing_pairs, "\n".join(missing_pairs)
