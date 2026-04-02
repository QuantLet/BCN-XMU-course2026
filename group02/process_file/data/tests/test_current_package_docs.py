from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_zh_focuses_on_current_package_only() -> None:
    content = _read("docs/README_zh.md")

    assert "processed/final_dataset_v2_plus.csv" in content
    assert "docs/data_dictionary_v2_plus.md" in content
    assert "docs/qa_report_v2_plus.md" in content
    assert "docs/btc_project_report.docx" in content
    assert "V1" not in content
    assert "V2-core" not in content
    assert "升级" not in content
    assert "历史" not in content


def test_data_spec_zh_focuses_on_current_package_only() -> None:
    content = _read("docs/data_spec_zh.md")

    assert "processed/final_dataset_v2_plus.csv" in content
    assert "fear_greed_index" in content
    assert "difficulty" in content
    assert "V1" not in content
    assert "V2-core" not in content
    assert "升级" not in content
    assert "版本线" not in content


def test_source_inventory_zh_describes_current_sources_without_version_history() -> None:
    content = _read("docs/source_inventory_zh.md")

    assert "processed/final_dataset_v2_plus.csv" in content
    assert "Alternative.me" in content
    assert "difficulty" in content
    assert "V1" not in content
    assert "V2-core" not in content
    assert "版本线" not in content
