# 最终交付清单

英文版：[final_handoff_checklist.md](final_handoff_checklist.md)

## 1. 文档目的

本清单只针对当前正式 BTC 数据集交付包。

在把仓库或输出文件交接给下游组员之前，建议按本清单逐项核对。

## 2. 必备交付物

请确认以下文件存在，且内容对应当前正式版本：

- `processed/final_dataset_v2_plus.csv`
- `docs/data_dictionary_v2_plus.md`
- `docs/qa_report_v2_plus.md`
- `docs/source_note_v2_plus.md`
- `docs/btc_project_report.docx`

## 3. 必备原始输入

当前正式版本依赖以下原始文件：

- `raw/raw_btc_ohlcv.csv`
- `raw/raw_blockchain_metrics.csv`
- `raw/raw_fred_macro.csv`
- `raw/raw_fear_greed_index.csv`
- `raw/raw_blockchain_difficulty.csv`

## 4. 当前正式脚本

当前正式流水线使用以下脚本：

- `notebooks/fetch_binance_btc_ohlcv.py`
- `notebooks/fetch_blockchain_btc_metrics.py`
- `notebooks/fetch_fred_macro.py`
- `notebooks/fetch_fear_greed_index.py`
- `notebooks/fetch_blockchain_btc_difficulty.py`
- `notebooks/build_dataset_v2_plus.py`
- `notebooks/run_post_binance_pipeline_v2_plus.py`
- `docs/generate_btc_project_report.py`

## 5. 当前保留测试

当前需要保留的测试为：

- `tests/test_fetch_fear_greed_index.py`
- `tests/test_fetch_blockchain_btc_difficulty.py`
- `tests/test_build_dataset_v2_plus.py`
- `tests/test_post_binance_pipeline_v2_plus.py`
- `tests/test_current_package_docs.py`
- `tests/test_generate_btc_project_report.py`

## 6. 重建命令

### 完整重建

```bash
python notebooks/fetch_binance_btc_ohlcv.py
python notebooks/fetch_blockchain_btc_metrics.py
python notebooks/fetch_fred_macro.py
python notebooks/fetch_fear_greed_index.py
python notebooks/fetch_blockchain_btc_difficulty.py
python notebooks/build_dataset_v2_plus.py
python docs/generate_btc_project_report.py
```

### 已有 Binance 原始表时的下游重建

```bash
python notebooks/run_post_binance_pipeline_v2_plus.py
python docs/generate_btc_project_report.py
```

### 验证命令

```bash
python -m pytest -q tests/test_fetch_fear_greed_index.py tests/test_fetch_blockchain_btc_difficulty.py tests/test_build_dataset_v2_plus.py tests/test_post_binance_pipeline_v2_plus.py tests/test_current_package_docs.py tests/test_generate_btc_project_report.py
```

## 7. 最终核对项

- [ ] `processed/final_dataset_v2_plus.csv` 已存在
- [ ] `docs/btc_project_report.docx` 已存在
- [ ] 当前保留测试全部通过
- [ ] `docs/data_dictionary_v2_plus.md` 与最终 CSV 字段一致
- [ ] `docs/qa_report_v2_plus.md` 与当前行数、列数、日期范围一致
- [ ] `docs/source_note_v2_plus.md` 与当前保留数据源一致

## 8. 不要删除的文件

- `数据工程.zip`
- `BCP分工.pdf`

这两份文件已被明确要求保留。
