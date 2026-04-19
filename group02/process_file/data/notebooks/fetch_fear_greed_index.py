from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = "https://api.alternative.me/fng/"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_OUTPUT = Path("raw/raw_fear_greed_index.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily Crypto Fear & Greed index data from Alternative.me."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def fetch_payload() -> dict[str, object]:
    params = urllib.parse.urlencode({"limit": 0, "format": "json"})
    request = urllib.request.Request(
        f"{BASE_URL}?{params}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise RuntimeError(f"Unexpected Alternative.me response: {data}")
    return data


def build_rows_from_payload(payload: dict[str, object], start_date: str) -> list[dict[str, object]]:
    records = payload.get("data")
    if not isinstance(records, list):
        raise RuntimeError(f"Missing data array in payload: {payload}")

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    rows: list[dict[str, object]] = []
    for item in records:
        if not isinstance(item, dict):
            raise RuntimeError(f"Unexpected record type: {item}")
        raw_timestamp = item.get("timestamp")
        raw_value = item.get("value")
        if raw_timestamp is None or raw_value is None:
            raise RuntimeError(f"Missing timestamp/value in payload row: {item}")
        timestamp = int(str(raw_timestamp))
        date = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
        if date < start:
            continue
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "fear_greed_index": float(str(raw_value)),
            }
        )

    rows.sort(key=lambda row: str(row["date"]))
    return rows


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["date", "fear_greed_index"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    payload = fetch_payload()
    rows = build_rows_from_payload(payload, start_date=args.start_date)
    if not rows:
        raise RuntimeError("No Fear & Greed rows were returned.")
    output_path = Path(args.output)
    write_csv(output_path, rows)
    print(f"Saved {len(rows)} rows to {output_path}")
    print(f"Date range: {rows[0]['date']} -> {rows[-1]['date']}")


if __name__ == "__main__":
    main()
