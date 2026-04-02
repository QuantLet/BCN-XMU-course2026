# 日期对齐抽样核对清单

英文版：[date_alignment_checklist.md](date_alignment_checklist.md)

## 1. 文档目的

本清单用于人工抽样核对当前最新版 `processed/final_dataset_v2_plus.csv` 的日期对齐语义。

重点排查三类高风险问题：

- 整列整体偏移 1 天
- 特征与标签混用了同一天的信息
- merge 后未来值被错误回填到过去

## 2. 当前核心语义

对于当前 `V2-plus` 版本：

1. `date=t` 这一行的 `target` 含义是 `1[close_t > close_{t-1}]`
2. `date=t` 这一行的全部建模特征只能来自 `t-1` 或更早的信息
3. `fear_greed_index`、`difficulty` 和全部衍生特征也必须遵守同样的统一滞后规则

## 3. 抽样时需要对照的文件

对某个抽样日期 `t`，建议同时查看：

- `raw/raw_btc_ohlcv.csv`
- `raw/raw_blockchain_metrics.csv`
- `raw/raw_fred_macro.csv`
- `raw/raw_fear_greed_index.csv`
- `raw/raw_blockchain_difficulty.csv`
- `processed/final_dataset_v2_plus.csv`

## 4. V2-plus 专项检查

- [ ] `date=t` 的 `fear_greed_index` 对应到 `t-1` 或更早已知情绪值
- [ ] `date=t` 的 `difficulty` 对应到原始难度表中的 `t-1` 值
- [ ] `date=t` 的 `difficulty_change` 只反映 `t-1` 时点已知的难度变化
- [ ] `active_addresses_change` 和 `fees_per_tx` 在派生后仍被统一滞后

## 5. 建议执行时点

建议在以下时点至少执行一次：

- 新增一个数据源后
- 修改统一滞后逻辑后
- 刷新正式交付包后
- 向下游组员正式交接之前
