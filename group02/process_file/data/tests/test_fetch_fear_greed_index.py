from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fetch_fear_greed_index.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_fear_greed_index", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_rows_from_payload_sorts_and_filters_by_start_date() -> None:
    module = _load_module()
    payload = {
        "data": [
            {"value": "60", "timestamp": "1517529600"},
            {"value": "40", "timestamp": "1517356800"},
            {"value": "50", "timestamp": "1517443200"},
        ]
    }

    rows = module.build_rows_from_payload(payload, start_date="2018-02-01")

    assert rows == [
        {"date": "2018-02-01", "fear_greed_index": 50.0},
        {"date": "2018-02-02", "fear_greed_index": 60.0},
    ]


def test_write_csv_writes_expected_headers(tmp_path: Path) -> None:
    module = _load_module()
    output_path = tmp_path / "raw_fear_greed_index.csv"
    rows = [{"date": "2018-02-01", "fear_greed_index": 50.0}]

    module.write_csv(output_path, rows)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "date,fear_greed_index",
        "2018-02-01,50.0",
    ]
