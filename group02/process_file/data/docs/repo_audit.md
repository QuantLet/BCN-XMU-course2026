# Repository Audit

Chinese version: [repo_audit_zh.md](repo_audit_zh.md)

## 1. Purpose

This document records the current retained repository structure for the formal BTC dataset package.

## 2. Current Scope

The repository currently covers:

- raw source fetching
- final dataset construction
- dataset-level documentation and handoff materials
- regression tests for the formal package
- docx report generation

Downstream model training, backtesting, and SHAP workflows remain outside this repository.

## 3. Current Validation Commands

```bash
python notebooks/fetch_blockchain_btc_metrics.py
python notebooks/fetch_fred_macro.py
python notebooks/fetch_fear_greed_index.py
python notebooks/fetch_blockchain_btc_difficulty.py
python notebooks/build_dataset_v2_plus.py
python docs/generate_btc_project_report.py
python -m pytest -q
```

## 4. Current Retained Structure

```text
docs/
  README*.md
  data_spec*.md
  source_inventory*.md
  data_cleaning_rules*.md
  date_alignment_checklist*.md
  repo_audit*.md
  data_dictionary_v2_plus.md
  qa_report_v2_plus.md
  source_note_v2_plus.md
  generate_btc_project_report.py
  btc_project_report.docx
notebooks/
  fetch_binance_btc_ohlcv.py
  fetch_blockchain_btc_metrics.py
  fetch_blockchain_btc_difficulty.py
  fetch_fred_macro.py
  fetch_fear_greed_index.py
  build_dataset_v2_plus.py
  run_post_binance_pipeline_v2_plus.py
processed/
  final_dataset_v2_plus.csv
raw/
  raw_btc_ohlcv.csv
  raw_blockchain_metrics.csv
  raw_blockchain_difficulty.csv
  raw_fred_macro.csv
  raw_fear_greed_index.csv
tests/
  test_fetch_fear_greed_index.py
  test_fetch_blockchain_btc_difficulty.py
  test_build_dataset_v2_plus.py
  test_post_binance_pipeline_v2_plus.py
  test_generate_btc_project_report.py
```

## 5. Audit Conclusion

The retained repository currently provides a complete upstream BTC dataset delivery package centered on `processed/final_dataset_v2_plus.csv` and its supporting docs and report.
