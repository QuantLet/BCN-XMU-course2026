"""
run_all.py
==========
One-click pipeline runner for:
"The Asymmetric Spillover Effects of Underlying Asset Volatility on NFT Liquidity"

Executes all five scripts in sequential order.
Make sure all dependencies are installed first:
    pip install -r requirements.txt

Note: Script 1b (Kaggle processing) requires data/kaggle_raw_nft.csv (~1.15 GB).
      Place the raw dataset in the data/ directory before running.
"""

import subprocess
import sys
import time

SCRIPTS = [
    ("1a_historical_macro.py",    "Step 1a │ Fetching ETH high-frequency data & computing RV/BPV/Jump"),
    ("1b_kaggle_process.py",      "Step 1b │ Processing Kaggle NFT dataset & classifying Blue-chip/Tail"),
    ("2_panel_construction.py",   "Step 2  │ Merging macro & micro data into econometric panel"),
    ("3_var_model_irf.py",        "Step 3  │ Fitting VAR model & plotting Orthogonalized IRFs"),
    ("4_local_projections_irf.py","Step 4  │ Running Jorda Local Projections (robustness check)"),
]

def run_pipeline():
    print("=" * 65)
    print("  NFT Volatility Spillover — Full Empirical Pipeline")
    print("=" * 65)

    total = len(SCRIPTS)
    for i, (script, description) in enumerate(SCRIPTS, start=1):
        print(f"\n[{i}/{total}] {description}")
        print("-" * 65)

        start = time.time()
        result = subprocess.run(
            [sys.executable, f"scripts/{script}"],
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"✅ Completed in {elapsed:.1f}s")
        else:
            print(f"❌ Script failed with exit code {result.returncode}.")
            print("   Pipeline halted. Please fix the error above and rerun.")
            sys.exit(result.returncode)

    print("\n" + "=" * 65)
    print("  🎉 Full pipeline completed successfully!")
    print("  Output figures saved in: figures/")
    print("=" * 65)

if __name__ == "__main__":
    run_pipeline()
