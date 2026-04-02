from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_OUTPUT = Path("raw/raw_fred_macro.csv")
SERIES = {
    "VIXCLS": "vix",
    "SP500": "sp500",
    "DTWEXBGS": "dxy_proxy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily macro variables from FRED CSV downloads."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def fetch_series(series_id: str, start_date: str, end_date: str) -> list[dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "id": series_id,
            "cosd": start_date,
            "coed": end_date,
        }
    )
    url = f"{BASE_URL}?{params}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Connection": "close"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if shutil.which("curl"):
                result = subprocess.run(
                    ["curl", "-L", "--max-time", "180", url],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                payload = result.stdout
            else:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = response.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(payload))
            return list(reader)
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    if last_error is not None:
        raise RuntimeError(f"Failed to fetch FRED series {series_id}: {last_error}")
    raise RuntimeError(f"Failed to fetch FRED series {series_id}")


def build_rows(start_date: str) -> list[dict[str, object]]:
    end_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    merged: dict[str, dict[str, object]] = {}
    for series_id, column_name in SERIES.items():
        for item in fetch_series(series_id, start_date, end_date):
            date = item["observation_date"]
            row = merged.setdefault(date, {"date": date})
            value = item[series_id]
            row[column_name] = None if value in {".", ""} else float(value)
    rows = [merged[date] for date in sorted(merged)]
    return rows


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["date", *SERIES.values()]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args.start_date)
    if not rows:
        raise RuntimeError("No FRED rows were returned.")
    output_path = Path(args.output)
    write_csv(output_path, rows)
    print(f"Saved {len(rows)} rows to {output_path}")
    print(f"Date range: {rows[0]['date']} -> {rows[-1]['date']}")


if __name__ == "__main__":
    main()
