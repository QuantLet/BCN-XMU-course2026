# 仓库审计记录

英文版：[repo_audit.md](repo_audit.md)

## 1. 文档目的

本文档用于记录当前 BTC 正式数据集交付包所对应的仓库保留结构。

## 2. 当前覆盖范围

当前仓库覆盖以下内容：

- 原始数据抓取
- 最终数据集构建
- 数据集层面的文档与交付材料
- 正式交付包的回归测试
- docx 报告生成

下游模型训练、回测和 SHAP 解释仍不在本仓库范围内。

## 3. 当前推荐验证命令

```bash
python notebooks/fetch_blockchain_btc_metrics.py
python notebooks/fetch_fred_macro.py
python notebooks/fetch_fear_greed_index.py
python notebooks/fetch_blockchain_btc_difficulty.py
python notebooks/build_dataset_v2_plus.py
python docs/generate_btc_project_report.py
python -m pytest -q
```

## 4. 当前保留结构

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

## 5. 当前审计结论

当前保留仓库已经形成一套围绕 `processed/final_dataset_v2_plus.csv` 的完整 BTC 上游数据交付包。
