from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "build_dataset_v2_plus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_dataset_v2_plus", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_price_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=45, freq="D", tz="UTC")
    rows = []
    for i, dt in enumerate(dates, start=1):
        close_time = dt + pd.Timedelta(hours=23, minutes=59, seconds=59, milliseconds=999)
        if i == len(dates):
            close_time = dt + pd.Timedelta(hours=8)
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "open_time": dt.isoformat(),
                "close_time": close_time.isoformat(),
                "open": float(i),
                "high": float(i) + 0.5,
                "low": float(i) - 0.5,
                "close": float(i) + 0.25,
                "volume": float(1000 + i * 10),
                "quote_volume": float(2000 + i * 10),
                "trade_count": 100 + i,
            }
        )
    return pd.DataFrame(rows)


def _make_onchain_frame(price_dates: pd.Series) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(price_dates), dtype="datetime64[ns]")
    values = pd.Series(range(1, len(dates) + 1), dtype="float64")
    date_strings = dates.dt.strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "date": date_strings,
            "active_addresses": values,
            "tx_count": values + 100.0,
            "hash_rate": values + 200.0,
            "fees": values + 300.0,
        }
    )


def _make_macro_frame(price_dates: pd.Series) -> pd.DataFrame:
    dates = list(pd.to_datetime(price_dates))
    weekdays = [dt for dt in dates if dt.weekday() < 5]
    values = pd.Series(range(1, len(weekdays) + 1), dtype="float64")
    weekday_strings = pd.Series(weekdays, dtype="datetime64[ns]").dt.strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "date": weekday_strings,
            "vix": values,
            "sp500": values + 10.0,
            "dxy_proxy": values + 20.0,
        }
    )


def _make_sentiment_frame(price_dates: pd.Series) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(price_dates), dtype="datetime64[ns]")
    usable = dates.iloc[15:].reset_index(drop=True)
    values = pd.Series(range(10, 10 + len(usable)), dtype="float64")
    return pd.DataFrame(
        {
            "date": usable.dt.strftime("%Y-%m-%d"),
            "fear_greed_index": values,
        }
    )


def _make_difficulty_frame(price_dates: pd.Series) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(price_dates), dtype="datetime64[ns]")
    usable = dates.iloc[5:].reset_index(drop=True)
    values = pd.Series(range(1000, 1000 + len(usable)), dtype="float64")
    return pd.DataFrame(
        {
            "date": usable.dt.strftime("%Y-%m-%d"),
            "difficulty": values,
        }
    )


def test_build_final_dataset_v2_plus_adds_and_shifts_difficulty() -> None:
    module = _load_module()
    price_df = _make_price_frame()
    trimmed = module.remove_latest_binance_day(price_df)
    onchain_df = _make_onchain_frame(trimmed["date"])
    macro_df = _make_macro_frame(trimmed["date"])
    sentiment_df = _make_sentiment_frame(trimmed["date"])
    difficulty_df = _make_difficulty_frame(trimmed["date"])

    final_df = module.build_final_dataset_v2_plus(
        price_df,
        onchain_df,
        macro_df,
        sentiment_df,
        difficulty_df,
    )

    assert "2024-02-14" not in final_df["date"].tolist()
    assert "fear_greed_index" in final_df.columns
    assert "difficulty" in final_df.columns
    assert "difficulty_change" in final_df.columns

    row = final_df.loc[final_df["date"] == "2024-02-10"].iloc[0]
    difficulty_prev = difficulty_df.loc[difficulty_df["date"] == "2024-02-09", "difficulty"].iloc[0]
    difficulty_prev_prev = difficulty_df.loc[difficulty_df["date"] == "2024-02-08", "difficulty"].iloc[0]
    expected_difficulty_change = (difficulty_prev - difficulty_prev_prev) / difficulty_prev_prev

    assert row["difficulty"] == difficulty_prev
    assert row["difficulty_change"] == pytest.approx(expected_difficulty_change)


def test_build_final_dataset_v2_plus_writes_v2_plus_outputs(tmp_path: Path) -> None:
    module = _load_module()
    price_df = _make_price_frame()
    trimmed = module.remove_latest_binance_day(price_df)
    onchain_df = _make_onchain_frame(trimmed["date"])
    macro_df = _make_macro_frame(trimmed["date"])
    sentiment_df = _make_sentiment_frame(trimmed["date"])
    difficulty_df = _make_difficulty_frame(trimmed["date"])

    output_path = tmp_path / "final_dataset_v2_plus.csv"
    dictionary_path = tmp_path / "data_dictionary_v2_plus.md"
    qa_path = tmp_path / "qa_report_v2_plus.md"

    final_df = module.build_final_dataset_v2_plus(
        price_df,
        onchain_df,
        macro_df,
        sentiment_df,
        difficulty_df,
        output_path=output_path,
        data_dictionary_path=dictionary_path,
        qa_report_path=qa_path,
    )

    assert len(final_df) > 0
    assert output_path.exists()
    assert dictionary_path.exists()
    assert qa_path.exists()
