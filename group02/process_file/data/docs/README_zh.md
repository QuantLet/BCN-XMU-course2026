# BTC 数据集交付说明

英文版：[README.md](README.md)

## 1. 文档目的

本文档用于说明当前正式保留的 BTC 数据集交付包。

## 2. 当前数据包

- 资产：`BTC`
- 频率：日频
- 原始起始日期：`2018-01-01`
- 最终数据集：`processed/final_dataset_v2_plus.csv`
- 标签定义：`target_t = 1[close_t > close_{t-1}]`
- 防泄露规则：最终训练表中的全部建模特征统一滞后 1 期

## 3. 当前脚本

- `notebooks/fetch_binance_btc_ohlcv.py`
- `notebooks/fetch_blockchain_btc_metrics.py`
- `notebooks/fetch_fred_macro.py`
- `notebooks/fetch_fear_greed_index.py`
- `notebooks/fetch_blockchain_btc_difficulty.py`
- `notebooks/build_dataset_v2_plus.py`
- `notebooks/run_post_binance_pipeline_v2_plus.py`

## 4. 推荐执行顺序

### 完整刷新

```bash
python notebooks/fetch_binance_btc_ohlcv.py
python notebooks/fetch_blockchain_btc_metrics.py
python notebooks/fetch_fred_macro.py
python notebooks/fetch_fear_greed_index.py
python notebooks/fetch_blockchain_btc_difficulty.py
python notebooks/build_dataset_v2_plus.py
```

### 已有 Binance 原始表时

```bash
python notebooks/run_post_binance_pipeline_v2_plus.py
```

## 5. 当前交付物

- `processed/final_dataset_v2_plus.csv`
- `docs/data_dictionary_v2_plus.md`
- `docs/qa_report_v2_plus.md`
- `docs/source_note_v2_plus.md`
- `docs/btc_project_report.docx`
- `docs/final_handoff_checklist_zh.md`

## 6. 文件作用索引

### Markdown 文件

- `docs/README.md`：当前 BTC 交付包的英文总说明，概览运行顺序、产物和环境要求。
- `docs/README_zh.md`：`docs/README.md` 的中文对照版，说明同一套当前交付包信息。
- `docs/data_spec.md`：数据集主规范，定义范围、标签口径、防泄露规则和交付约束。
- `docs/data_spec_zh.md`：`docs/data_spec.md` 的中文对照版。
- `docs/source_inventory.md`：数据源清单，说明当前保留的上游来源及其字段映射关系。
- `docs/source_inventory_zh.md`：`docs/source_inventory.md` 的中文对照版。
- `docs/data_cleaning_rules.md`：数据清洗规则，覆盖排序、去重、缺失值、滞后处理和异常值策略。
- `docs/data_cleaning_rules_zh.md`：`docs/data_cleaning_rules.md` 的中文对照版。
- `docs/date_alignment_checklist.md`：日期对齐人工检查清单，用于排查泄露风险和列错位问题。
- `docs/date_alignment_checklist_zh.md`：`docs/date_alignment_checklist.md` 的中文对照版。
- `docs/final_handoff_checklist.md`：最终交付核对清单，覆盖必需文件、脚本、测试和重建要求。
- `docs/final_handoff_checklist_zh.md`：`docs/final_handoff_checklist.md` 的中文对照版。
- `docs/repo_audit.md`：仓库审计说明，记录当前保留结构、范围边界和验证命令。
- `docs/repo_audit_zh.md`：`docs/repo_audit.md` 的中文对照版。
- `docs/data_dictionary_v2_plus.md`：最终 V2-plus 数据集的数据字典，逐列说明字段含义和是否已为建模滞后。
- `docs/qa_report_v2_plus.md`：当前最终数据集的 QA 快照，汇总行列数、日期范围、目标比例和缺失率。
- `docs/source_note_v2_plus.md`：V2-plus 来源决策说明，解释为何新增 `difficulty` 和 `difficulty_change`。

### Python 文件

- `notebooks/fetch_binance_btc_ohlcv.py`：抓取 Binance BTC 日线 OHLCV 数据，并写出主价格原始表。
- `notebooks/fetch_blockchain_btc_metrics.py`：抓取 Blockchain.com 链上指标，如活跃地址数、交易数、算力和手续费。
- `notebooks/fetch_blockchain_btc_difficulty.py`：单独抓取 Blockchain.com 的 `difficulty` 序列，作为新增原始输入。
- `notebooks/fetch_fred_macro.py`：抓取 FRED 宏观序列，用于生成 `vix`、`sp500` 和美元指数代理。
- `notebooks/fetch_fear_greed_index.py`：抓取 Alternative.me 的 Fear and Greed Index 序列。
- `notebooks/build_dataset_v2_plus.py`：核心构建脚本，负责合并原始来源、构造特征、统一滞后 1 期并输出最终数据集和配套文档。
- `notebooks/run_post_binance_pipeline_v2_plus.py`：下游 runner，按既定顺序执行 Binance 之后的抓取和最终构建流程。
- `docs/generate_btc_project_report.py`：根据 `processed/final_dataset_v2_plus.csv` 生成 Word 交付报告。
- `tests/test_build_dataset_v2_plus.py`：验证数据集构建逻辑、滞后语义和 V2-plus 产物写出。
- `tests/test_post_binance_pipeline_v2_plus.py`：验证流水线 runner 的命令列表和执行顺序保持稳定。
- `tests/test_fetch_blockchain_btc_difficulty.py`：验证难度抓取脚本的解析、排序和 CSV 输出格式。
- `tests/test_fetch_fear_greed_index.py`：验证 Fear and Greed 抓取脚本的过滤、排序和 CSV 输出格式。
- `tests/test_generate_btc_project_report.py`：验证报告生成脚本的输出路径和必要内容。
- `tests/test_current_package_docs.py`：验证关键交付文档始终聚焦当前 V2-plus 正式包。
- `tests/test_docs_bilingual_pairs.py`：验证 `docs/` 下双语 Markdown 文档配对保持完整。

## 7. 当前数据集状态

当前核验通过的快照如下：

- 行数：`2966`
- 列数：`27`
- 日期范围：`2018-02-02` 到 `2026-03-17`
- 已剔除最新 Binance 日期：`2026-03-18`

## 8. 当前字段分组

- 价格列：`open`, `high`, `low`, `close`, `volume`, `quote_volume`, `trade_count`
- 链上列：`active_addresses`, `tx_count`, `hash_rate`, `fees`, `difficulty`
- 宏观列：`vix`, `sp500_return`, `dxy_proxy`
- 情绪列：`fear_greed_index`
- 衍生列：`ret_1d`, `ret_3d`, `volatility_7d`, `ma_ratio_7_30`, `high_low_spread`, `volume_change`, `active_addresses_change`, `fees_per_tx`, `difficulty_change`
- 标签列：`target`

## 9. 当前验证范围

当前仓库已经验证以下事项：

- 最新未完成 Binance 日线不会进入最终输出
- `target` 构造逻辑稳定
- 外部特征和衍生特征统一滞后 1 期
- V2-plus 下游 runner 只执行当前正式流水线

## 10. 环境说明

- `notebooks/fetch_fred_macro.py` 在当前环境下优先使用系统 `curl`
- `docs/generate_btc_project_report.py` 依赖 `python-docx`
