from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PRICE_PATH = Path("raw/raw_btc_ohlcv.csv")
DEFAULT_ONCHAIN_PATH = Path("raw/raw_blockchain_metrics.csv")
DEFAULT_MACRO_PATH = Path("raw/raw_fred_macro.csv")
DEFAULT_SENTIMENT_PATH = Path("raw/raw_fear_greed_index.csv")
DEFAULT_DIFFICULTY_PATH = Path("raw/raw_blockchain_difficulty.csv")
DEFAULT_OUTPUT_PATH = Path("processed/final_dataset_v2_plus.csv")
DEFAULT_DICTIONARY_PATH = Path("docs/data_dictionary_v2_plus.md")
DEFAULT_QA_PATH = Path("docs/qa_report_v2_plus.md")

BASE_PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
]
ONCHAIN_COLUMNS = ["active_addresses", "tx_count", "hash_rate", "fees"]
PRICE_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "volatility_7d",
    "ma_ratio_7_30",
    "high_low_spread",
    "volume_change",
]
MACRO_SOURCE_COLUMNS = ["vix", "sp500", "dxy_proxy"]
FINAL_MACRO_COLUMNS = ["vix", "sp500_return", "dxy_proxy"]
V2_DERIVED_COLUMNS = ["active_addresses_change", "fees_per_tx"]
SENTIMENT_COLUMNS = ["fear_greed_index"]
DIFFICULTY_COLUMNS = ["difficulty", "difficulty_change"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the V2-plus BTC daily dataset from raw source files."
    )
    parser.add_argument("--price-path", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--onchain-path", default=str(DEFAULT_ONCHAIN_PATH))
    parser.add_argument("--macro-path", default=str(DEFAULT_MACRO_PATH))
    parser.add_argument("--sentiment-path", default=str(DEFAULT_SENTIMENT_PATH))
    parser.add_argument("--difficulty-path", default=str(DEFAULT_DIFFICULTY_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--data-dictionary-path", default=str(DEFAULT_DICTIONARY_PATH))
    parser.add_argument("--qa-report-path", default=str(DEFAULT_QA_PATH))
    return parser.parse_args()


def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    normalized = normalized.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return normalized.reset_index(drop=True).copy()


def remove_latest_binance_day(price_df: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_date_column(price_df)
    latest_date = str(normalized["date"].max())
    return normalized.loc[normalized["date"] != latest_date].reset_index(drop=True).copy()


def engineer_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    engineered = price_df.copy()
    close = engineered["close"]
    engineered["ret_1d"] = close.pct_change(1)
    engineered["ret_3d"] = close.pct_change(3)
    engineered["volatility_7d"] = close.pct_change().rolling(7).std()
    engineered["ma_ratio_7_30"] = close.rolling(7).mean() / close.rolling(30).mean()
    engineered["high_low_spread"] = (engineered["high"] - engineered["low"]) / close
    engineered["volume_change"] = engineered["volume"].pct_change(1)
    return engineered


def _prepare_external_frame(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    prepared = _normalize_date_column(df)
    for column in required_columns:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    return prepared.loc[:, ["date", *required_columns]].copy()


def write_data_dictionary(output_path: Path) -> None:
    descriptions = {
        "date": ("Dataset row date", "No"),
        "open": ("Lagged BTC open price from Binance", "Yes"),
        "high": ("Lagged BTC high price from Binance", "Yes"),
        "low": ("Lagged BTC low price from Binance", "Yes"),
        "close": ("Lagged BTC close price from Binance", "Yes"),
        "volume": ("Lagged BTC traded volume from Binance", "Yes"),
        "quote_volume": ("Lagged BTC quote volume from Binance", "Yes"),
        "trade_count": ("Lagged Binance trade count", "Yes"),
        "ret_1d": ("1-day BTC return", "Yes"),
        "ret_3d": ("3-day BTC return", "Yes"),
        "volatility_7d": ("7-day rolling close-return volatility", "Yes"),
        "ma_ratio_7_30": ("7-day / 30-day moving-average ratio", "Yes"),
        "high_low_spread": ("Intraday high-low spread divided by close", "Yes"),
        "volume_change": ("1-day percentage change in volume", "Yes"),
        "active_addresses": ("Blockchain.com unique active addresses", "Yes"),
        "tx_count": ("Blockchain.com confirmed transactions per day", "Yes"),
        "hash_rate": ("Blockchain.com network hash rate", "Yes"),
        "fees": ("Blockchain.com total transaction fees", "Yes"),
        "vix": ("FRED VIXCLS index level", "Yes"),
        "sp500_return": ("S&P 500 daily return derived from FRED SP500", "Yes"),
        "dxy_proxy": ("FRED DTWEXBGS dollar index proxy", "Yes"),
        "active_addresses_change": ("1-day percentage change in active addresses", "Yes"),
        "fees_per_tx": ("Daily fees divided by transaction count", "Yes"),
        "fear_greed_index": ("Alternative.me crypto fear and greed index", "Yes"),
        "difficulty": ("Blockchain.com network difficulty", "Yes"),
        "difficulty_change": ("1-day percentage change in network difficulty", "Yes"),
        "target": ("1 if close_t > close_{t-1}, else 0", "No"),
    }
    lines = [
        "# Data Dictionary (V2 Plus)",
        "",
        "| Column | Description | Shifted for Modeling |",
        "|---|---|---|",
    ]
    for column, (description, shifted) in descriptions.items():
        lines.append(f"| `{column}` | {description} | {shifted} |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qa_report(output_path: Path, final_df: pd.DataFrame, removed_date: str) -> None:
    missing_rates = pd.Series(final_df.isna().mean(), dtype="float64").sort_index()
    lines = [
        "# QA Report (V2 Plus)",
        "",
        f"- Rows: {len(final_df)}",
        f"- Columns: {len(final_df.columns)}",
        f"- Date range: {final_df['date'].min()} -> {final_df['date'].max()}",
        f"- Dropped latest Binance date: {removed_date}",
        f"- Target positive ratio: {final_df['target'].mean():.4f}",
        "",
        "## Missing Rate",
        "",
        "| Column | Missing Rate |",
        "|---|---|",
    ]
    for column, value in missing_rates.items():
        lines.append(f"| `{column}` | {value:.4%} |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_final_dataset_v2_plus(
    price_df: pd.DataFrame,
    onchain_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    output_path: Path | None = None,
    data_dictionary_path: Path | None = None,
    qa_report_path: Path | None = None,
) -> pd.DataFrame:
    normalized_price = _normalize_date_column(price_df)
    normalized_price[BASE_PRICE_COLUMNS] = normalized_price[BASE_PRICE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    normalized_price["target"] = (
        normalized_price["close"] > normalized_price["close"].shift(1)
    ).astype("Int64")

    removed_date = str(normalized_price["date"].max())
    trimmed_price = remove_latest_binance_day(normalized_price)
    trimmed_price = engineer_price_features(trimmed_price)

    onchain = _prepare_external_frame(onchain_df, ONCHAIN_COLUMNS)
    onchain[ONCHAIN_COLUMNS] = onchain[ONCHAIN_COLUMNS].apply(pd.to_numeric, errors="coerce")

    macro = _prepare_external_frame(macro_df, MACRO_SOURCE_COLUMNS)
    macro[MACRO_SOURCE_COLUMNS] = macro[MACRO_SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    sentiment = _prepare_external_frame(sentiment_df, SENTIMENT_COLUMNS)
    sentiment[SENTIMENT_COLUMNS] = sentiment[SENTIMENT_COLUMNS].apply(pd.to_numeric, errors="coerce")

    difficulty = _prepare_external_frame(difficulty_df, ["difficulty"])
    difficulty[["difficulty"]] = difficulty[["difficulty"]].apply(pd.to_numeric, errors="coerce")

    merged = trimmed_price[["date", *BASE_PRICE_COLUMNS, *PRICE_FEATURE_COLUMNS, "target"]].merge(
        onchain, on="date", how="left"
    )
    merged = merged.merge(macro, on="date", how="left")
    merged = merged.merge(sentiment, on="date", how="left")
    merged = merged.merge(difficulty, on="date", how="left")

    merged[ONCHAIN_COLUMNS + MACRO_SOURCE_COLUMNS + SENTIMENT_COLUMNS + ["difficulty"]] = merged[
        ONCHAIN_COLUMNS + MACRO_SOURCE_COLUMNS + SENTIMENT_COLUMNS + ["difficulty"]
    ].ffill()
    merged["sp500_return"] = merged["sp500"].pct_change(1)
    merged["active_addresses_change"] = merged["active_addresses"].pct_change(1)
    merged["fees_per_tx"] = merged["fees"] / merged["tx_count"].replace(0, pd.NA)
    merged["difficulty_change"] = merged["difficulty"].pct_change(1)
    merged = merged.drop(columns=["sp500"])

    feature_columns = [
        *BASE_PRICE_COLUMNS,
        *PRICE_FEATURE_COLUMNS,
        *ONCHAIN_COLUMNS,
        *FINAL_MACRO_COLUMNS,
        *V2_DERIVED_COLUMNS,
        *SENTIMENT_COLUMNS,
        *DIFFICULTY_COLUMNS,
    ]
    merged[feature_columns] = merged[feature_columns].shift(1)
    final_df = merged.dropna(subset=[*feature_columns, "target"]).reset_index(drop=True).copy()
    final_df["target"] = final_df["target"].astype(int)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
    if data_dictionary_path is not None:
        write_data_dictionary(data_dictionary_path)
    if qa_report_path is not None:
        write_qa_report(qa_report_path, final_df, removed_date)
    return final_df


def main() -> None:
    args = parse_args()
    price_df = pd.read_csv(args.price_path)
    onchain_df = pd.read_csv(args.onchain_path)
    macro_df = pd.read_csv(args.macro_path)
    sentiment_df = pd.read_csv(args.sentiment_path)
    difficulty_df = pd.read_csv(args.difficulty_path)
    final_df = build_final_dataset_v2_plus(
        price_df,
        onchain_df,
        macro_df,
        sentiment_df,
        difficulty_df,
        output_path=Path(args.output_path),
        data_dictionary_path=Path(args.data_dictionary_path),
        qa_report_path=Path(args.qa_report_path),
    )
    print(f"Saved {len(final_df)} rows to {args.output_path}")
    print(f"Date range: {final_df['date'].min()} -> {final_df['date'].max()}")


if __name__ == "__main__":
    main()
