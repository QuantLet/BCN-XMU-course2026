from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "docs" / "generate_btc_project_report.py"
SPEC = importlib.util.spec_from_file_location("generate_btc_project_report", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
REPORT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT_MODULE)


def test_report_output_path_uses_docs_directory() -> None:
    assert REPORT_MODULE.OUTPUT_PATH == Path("docs/btc_project_report.docx")


def test_report_text_promotes_v2_plus_package() -> None:
    document = REPORT_MODULE.build_report()
    full_text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    assert "final_dataset_v2_plus.csv" in full_text
    assert "fear_greed_index" in full_text
    assert "difficulty" in full_text
    assert "升级" not in full_text
    assert "V1 基线" not in full_text
    assert "V2-core" not in full_text
