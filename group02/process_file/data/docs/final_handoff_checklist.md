# Final Handoff Checklist

Chinese version: [final_handoff_checklist_zh.md](final_handoff_checklist_zh.md)

## 1. Purpose

This checklist is for the current formal BTC dataset package only.

Use it before handing the repository or the output files to downstream teammates.

## 2. Required Deliverables

Make sure the following files exist and match the current package:

- `processed/final_dataset_v2_plus.csv`
- `docs/data_dictionary_v2_plus.md`
- `docs/qa_report_v2_plus.md`
- `docs/source_note_v2_plus.md`
- `docs/btc_project_report.docx`

## 3. Required Raw Inputs

The current package depends on these raw files:

- `raw/raw_btc_ohlcv.csv`
- `raw/raw_blockchain_metrics.csv`
- `raw/raw_fred_macro.csv`
- `raw/raw_fear_greed_index.csv`
- `raw/raw_blockchain_difficulty.csv`

## 4. Required Scripts

The current formal pipeline uses:

- `notebooks/fetch_binance_btc_ohlcv.py`
- `notebooks/fetch_blockchain_btc_metrics.py`
- `notebooks/fetch_fred_macro.py`
- `notebooks/fetch_fear_greed_index.py`
- `notebooks/fetch_blockchain_btc_difficulty.py`
- `notebooks/build_dataset_v2_plus.py`
- `notebooks/run_post_binance_pipeline_v2_plus.py`
- `docs/generate_btc_project_report.py`

## 5. Required Tests

The current kept tests are:

- `tests/test_fetch_fear_greed_index.py`
- `tests/test_fetch_blockchain_btc_difficulty.py`
- `tests/test_build_dataset_v2_plus.py`
- `tests/test_post_binance_pipeline_v2_plus.py`
- `tests/test_current_package_docs.py`
- `tests/test_generate_btc_project_report.py`

## 6. Rebuild Commands

### Full rebuild

```bash
python notebooks/fetch_binance_btc_ohlcv.py
python notebooks/fetch_blockchain_btc_metrics.py
python notebooks/fetch_fred_macro.py
python notebooks/fetch_fear_greed_index.py
python notebooks/fetch_blockchain_btc_difficulty.py
python notebooks/build_dataset_v2_plus.py
python docs/generate_btc_project_report.py
```

### Downstream rebuild if Binance raw data already exists

```bash
python notebooks/run_post_binance_pipeline_v2_plus.py
python docs/generate_btc_project_report.py
```

### Validation

```bash
python -m pytest -q tests/test_fetch_fear_greed_index.py tests/test_fetch_blockchain_btc_difficulty.py tests/test_build_dataset_v2_plus.py tests/test_post_binance_pipeline_v2_plus.py tests/test_current_package_docs.py tests/test_generate_btc_project_report.py
```

## 7. Final Checks

- [ ] `processed/final_dataset_v2_plus.csv` exists
- [ ] `docs/btc_project_report.docx` exists
- [ ] all retained tests pass
- [ ] `docs/data_dictionary_v2_plus.md` matches the final CSV fields
- [ ] `docs/qa_report_v2_plus.md` matches the current row/column/date summary
- [ ] `docs/source_note_v2_plus.md` matches the current retained sources

## 8. Files Not To Delete

- `数据工程.zip`
- `BCP分工.pdf`

These files were explicitly retained.
