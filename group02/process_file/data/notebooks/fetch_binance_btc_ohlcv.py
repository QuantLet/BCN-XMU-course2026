from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = "https://api.binance.com/api/v3/klines"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1d"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_OUTPUT = Path("raw/raw_btc_ohlcv.csv")
ONE_DAY_MS = 24 * 60 * 60 * 1000
HEADERS = [
    "date",
    "symbol",
    "interval",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Binance daily BTC OHLCV data and save it as CSV."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep between paginated requests.",
    )
    return parser.parse_args()


def date_to_millis(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
    )
    request = urllib.request.Request(
        f"{BASE_URL}?{params}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if isinstance(data, dict) and data.get("code") is not None:
        raise RuntimeError(f"Binance API error: {data}")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Binance response type: {type(data).__name__}")
    return data


def build_rows(symbol: str, interval: str, klines: list[list]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in klines:
        open_time_ms = int(item[0])
        close_time_ms = int(item[6])
        open_dt = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
        close_dt = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
        rows.append(
            {
                "date": open_dt.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "interval": interval,
                "open_time": open_dt.isoformat(),
                "close_time": close_dt.isoformat(),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "quote_volume": float(item[7]),
                "trade_count": int(item[8]),
                "taker_buy_base_volume": float(item[9]),
                "taker_buy_quote_volume": float(item[10]),
            }
        )
    return rows


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    start_ms = date_to_millis(args.start_date)
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    all_rows: list[dict[str, object]] = []
    next_start_ms = start_ms

    while next_start_ms <= end_ms:
        klines = fetch_klines(args.symbol, args.interval, next_start_ms, end_ms)
        if not klines:
            break

        all_rows.extend(build_rows(args.symbol, args.interval, klines))

        last_open_time = int(klines[-1][0])
        candidate_next_start = last_open_time + ONE_DAY_MS
        if candidate_next_start <= next_start_ms:
            raise RuntimeError("Pagination did not advance; stopping to avoid infinite loop.")

        next_start_ms = candidate_next_start
        time.sleep(args.sleep_seconds)

    if not all_rows:
        raise RuntimeError("No OHLCV rows were returned from Binance.")

    output_path = Path(args.output)
    write_csv(output_path, all_rows)

    print(f"Saved {len(all_rows)} rows to {output_path}")
    print(f"Date range: {all_rows[0]['date']} -> {all_rows[-1]['date']}")


if __name__ == "__main__":
    main()
