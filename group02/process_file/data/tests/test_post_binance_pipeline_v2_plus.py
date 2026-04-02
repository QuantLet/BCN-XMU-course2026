from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "run_post_binance_pipeline_v2_plus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_post_binance_pipeline_v2_plus", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_post_binance_commands_v2_plus_includes_difficulty_fetch() -> None:
    module = _load_module()
    commands = module.build_post_binance_commands_v2_plus("python")

    assert commands == [
        ["python", "notebooks/fetch_blockchain_btc_metrics.py"],
        ["python", "notebooks/fetch_fred_macro.py"],
        ["python", "notebooks/fetch_fear_greed_index.py"],
        ["python", "notebooks/fetch_blockchain_btc_difficulty.py"],
        ["python", "notebooks/build_dataset_v2_plus.py"],
    ]


def test_build_post_binance_commands_v2_plus_keeps_order() -> None:
    module = _load_module()
    commands = module.build_post_binance_commands_v2_plus("py")

    assert commands[0][-1] == "notebooks/fetch_blockchain_btc_metrics.py"
    assert commands[1][-1] == "notebooks/fetch_fred_macro.py"
    assert commands[2][-1] == "notebooks/fetch_fear_greed_index.py"
    assert commands[3][-1] == "notebooks/fetch_blockchain_btc_difficulty.py"
    assert commands[4][-1] == "notebooks/build_dataset_v2_plus.py"
