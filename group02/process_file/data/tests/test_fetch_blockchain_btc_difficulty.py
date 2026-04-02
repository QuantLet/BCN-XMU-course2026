from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fetch_blockchain_btc_difficulty.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_blockchain_btc_difficulty", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_rows_uses_chart_payload_and_sorts_dates() -> None:
    module = _load_module()

    def fake_fetch_chart(chart_name: str, start_date: str):
        assert chart_name == "difficulty"
        assert start_date == "2018-01-01"
        return [
            {"x": 1514851200, "y": 2.0},
            {"x": 1514764800, "y": 1.5},
        ]

    setattr(module, "fetch_chart", fake_fetch_chart)

    rows = module.build_rows("2018-01-01")

    assert rows == [
        {"date": "2018-01-01", "difficulty": 1.5},
        {"date": "2018-01-02", "difficulty": 2.0},
    ]


def test_write_csv_writes_expected_headers(tmp_path: Path) -> None:
    module = _load_module()
    output_path = tmp_path / "raw_blockchain_difficulty.csv"
    rows = [{"date": "2018-01-01", "difficulty": 1.5}]

    module.write_csv(output_path, rows)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "date,difficulty",
        "2018-01-01,1.5",
    ]
