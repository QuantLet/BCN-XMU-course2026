# BTC Dataset Delivery Guide

Chinese version: [README_zh.md](README_zh.md)

## 1. Purpose

This document describes the current formal BTC dataset package kept in this repository.

## 2. Current Package

- Asset: `BTC`
- Frequency: `Daily`
- Raw start date: `2018-01-01`
- Final dataset: `processed/final_dataset_v2_plus.csv`
- Label: `target_t = 1[close_t > close_{t-1}]`
- Leakage rule: all modeling features are lagged by one period in the final table

## 3. Current Scripts

- `notebooks/fetch_binance_btc_ohlcv.py`
- `notebooks/fetch_blockchain_btc_metrics.py`
- `notebooks/fetch_fred_macro.py`
- `notebooks/fetch_fear_greed_index.py`
- `notebooks/fetch_blockchain_btc_difficulty.py`
- `notebooks/build_dataset_v2_plus.py`
- `notebooks/run_post_binance_pipeline_v2_plus.py`

## 4. Recommended Run Order

### Full refresh

```bash
python notebooks/fetch_binance_btc_ohlcv.py
python notebooks/fetch_blockchain_btc_metrics.py
python notebooks/fetch_fred_macro.py
python notebooks/fetch_fear_greed_index.py
python notebooks/fetch_blockchain_btc_difficulty.py
python notebooks/build_dataset_v2_plus.py
```

### If Binance raw data already exists

```bash
python notebooks/run_post_binance_pipeline_v2_plus.py
```

## 5. Current Deliverables

- `processed/final_dataset_v2_plus.csv`
- `docs/data_dictionary_v2_plus.md`
- `docs/qa_report_v2_plus.md`
- `docs/source_note_v2_plus.md`
- `docs/btc_project_report.docx`
- `docs/final_handoff_checklist.md`

## 6. File Role Index

### Markdown files

- `docs/README.md`: English overview of the current BTC delivery package, run order, outputs, and environment notes.
- `docs/README_zh.md`: Chinese counterpart to `docs/README.md` for the same package overview.
- `docs/data_spec.md`: Main dataset specification covering scope, label definition, leakage control, and delivery constraints.
- `docs/data_spec_zh.md`: Chinese counterpart to `docs/data_spec.md`.
- `docs/source_inventory.md`: Source inventory describing each retained upstream data source and its mapped fields.
- `docs/source_inventory_zh.md`: Chinese counterpart to `docs/source_inventory.md`.
- `docs/data_cleaning_rules.md`: Cleaning rules for ordering, deduplication, missing values, lag handling, and outlier treatment.
- `docs/data_cleaning_rules_zh.md`: Chinese counterpart to `docs/data_cleaning_rules.md`.
- `docs/date_alignment_checklist.md`: Manual checklist for checking date alignment, leakage risk, and column shift errors.
- `docs/date_alignment_checklist_zh.md`: Chinese counterpart to `docs/date_alignment_checklist.md`.
- `docs/final_handoff_checklist.md`: Final delivery checklist covering required files, scripts, tests, and rebuild expectations.
- `docs/final_handoff_checklist_zh.md`: Chinese counterpart to `docs/final_handoff_checklist.md`.
- `docs/repo_audit.md`: Repository audit note capturing the kept structure, scope boundaries, and validation commands.
- `docs/repo_audit_zh.md`: Chinese counterpart to `docs/repo_audit.md`.
- `docs/data_dictionary_v2_plus.md`: Field-level dictionary for the final V2-plus dataset, including column meanings and modeling shift status.
- `docs/qa_report_v2_plus.md`: QA snapshot for the current final dataset, including row count, date range, target ratio, and missing rates.
- `docs/source_note_v2_plus.md`: V2-plus source decision note explaining why `difficulty` and `difficulty_change` were added.

### Python files

- `notebooks/fetch_binance_btc_ohlcv.py`: Fetches Binance BTC daily OHLCV data and writes the main raw price table.
- `notebooks/fetch_blockchain_btc_metrics.py`: Fetches Blockchain.com on-chain metrics such as active addresses, transaction count, hash rate, and fees.
- `notebooks/fetch_blockchain_btc_difficulty.py`: Fetches the Blockchain.com `difficulty` series as a dedicated raw input.
- `notebooks/fetch_fred_macro.py`: Fetches macro series from FRED for `vix`, `sp500`, and the dollar-index proxy.
- `notebooks/fetch_fear_greed_index.py`: Fetches the Alternative.me Fear and Greed Index series.
- `notebooks/build_dataset_v2_plus.py`: Core build script that merges raw sources, engineers features, applies the one-period lag rule, and writes the final dataset plus supporting docs.
- `notebooks/run_post_binance_pipeline_v2_plus.py`: Runner script that executes the post-Binance pipeline in the intended order before the final build.
- `docs/generate_btc_project_report.py`: Generates the Word delivery report from `processed/final_dataset_v2_plus.csv`.
- `tests/test_build_dataset_v2_plus.py`: Regression tests for dataset construction, lag semantics, and V2-plus output artifacts.
- `tests/test_post_binance_pipeline_v2_plus.py`: Tests the pipeline runner to ensure the expected command list and execution order stay fixed.
- `tests/test_fetch_blockchain_btc_difficulty.py`: Unit tests for difficulty fetch parsing, sorting, and CSV output shape.
- `tests/test_fetch_fear_greed_index.py`: Unit tests for Fear and Greed fetch filtering, ordering, and CSV output shape.
- `tests/test_generate_btc_project_report.py`: Tests the report generator for output path and required report content.
- `tests/test_current_package_docs.py`: Tests that the key package docs stay focused on the current V2-plus deliverable.
- `tests/test_docs_bilingual_pairs.py`: Tests that bilingual Markdown document pairs remain complete in `docs/`.

## 7. Dataset Snapshot

Current verified snapshot:

- Rows: `2966`
- Columns: `27`
- Date range: `2018-02-02` to `2026-03-17`
- Dropped latest Binance date: `2026-03-18`

## 8. Feature Groups

- Price fields: `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `trade_count`
- On-chain fields: `active_addresses`, `tx_count`, `hash_rate`, `fees`, `difficulty`
- Macro fields: `vix`, `sp500_return`, `dxy_proxy`
- Sentiment field: `fear_greed_index`
- Derived fields: `ret_1d`, `ret_3d`, `volatility_7d`, `ma_ratio_7_30`, `high_low_spread`, `volume_change`, `active_addresses_change`, `fees_per_tx`, `difficulty_change`
- Label: `target`

## 9. Verification Scope

The current repository verifies that:

- the latest incomplete Binance day is excluded
- target construction remains stable
- external and derived features are lagged by one day
- the V2-plus downstream runner executes the intended pipeline only

## 10. Environment Note

- `notebooks/fetch_fred_macro.py` prefers system `curl` when available
- `docs/generate_btc_project_report.py` requires `python-docx`
