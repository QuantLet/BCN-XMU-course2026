from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = "https://api.blockchain.info/charts"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_OUTPUT = Path("raw/raw_blockchain_difficulty.csv")
CHART_NAME = "difficulty"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily Bitcoin network difficulty from Blockchain.com."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def fetch_chart(chart_name: str, start_date: str) -> list[dict[str, object]]:
    params = urllib.parse.urlencode(
        {
            "timespan": "all",
            "start": start_date,
            "format": "json",
            "sampled": "false",
        }
    )
    request = urllib.request.Request(
        f"{BASE_URL}/{chart_name}?{params}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    values = data.get("values")
    if not isinstance(values, list):
        raise RuntimeError(f"Unexpected Blockchain.com response for {chart_name}: {data}")
    return values


def build_rows(start_date: str) -> list[dict[str, object]]:
    rows = []
    for item in fetch_chart(CHART_NAME, start_date):
        raw_x = item.get("x")
        raw_y = item.get("y")
        if raw_x is None or raw_y is None:
            raise RuntimeError(f"Missing x/y in Blockchain.com row for {CHART_NAME}: {item}")
        timestamp = int(str(raw_x))
        value = float(str(raw_y))
        date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append({"date": date, "difficulty": value})
    rows.sort(key=lambda row: str(row["date"]))
    return rows


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["date", "difficulty"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args.start_date)
    if not rows:
        raise RuntimeError("No Blockchain.com difficulty rows were returned.")
    output_path = Path(args.output)
    write_csv(output_path, rows)
    print(f"Saved {len(rows)} rows to {output_path}")
    print(f"Date range: {rows[0]['date']} -> {rows[-1]['date']}")


if __name__ == "__main__":
    main()
