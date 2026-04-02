from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V2-plus dataset pipeline strictly after the Binance raw table is already available."
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for each downstream step.",
    )
    return parser.parse_args()


def build_post_binance_commands_v2_plus(python_executable: str) -> list[list[str]]:
    return [
        [python_executable, "notebooks/fetch_blockchain_btc_metrics.py"],
        [python_executable, "notebooks/fetch_fred_macro.py"],
        [python_executable, "notebooks/fetch_fear_greed_index.py"],
        [python_executable, "notebooks/fetch_blockchain_btc_difficulty.py"],
        [python_executable, "notebooks/build_dataset_v2_plus.py"],
    ]


def main() -> None:
    args = parse_args()
    for command in build_post_binance_commands_v2_plus(args.python_executable):
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
