# 缺失值与清洗规则

英文版：[data_cleaning_rules.md](data_cleaning_rules.md)

## 1. 文档目的

本文档用于记录当前 BTC 数据集正式构建器所使用的清洗规则。

## 2. 当前构建器

- `notebooks/build_dataset_v2_plus.py`

## 3. 当前规则

- `date` 统一标准化为 `%Y-%m-%d`
- 全部记录按 `date` 排序
- 重复日期保留最后一条
- 源字段使用 `pd.to_numeric(..., errors="coerce")` 数值化
- 最终导出前剔除最新 Binance 日期
- 在裁剪后的价格表上构造价格衍生特征
- 在最终滞后前，对保留外部源做前向填充
- 对全部建模特征统一执行 1 期滞后
- 滞后完成后，对必需特征列和 `target` 执行 `dropna`

## 4. 当前覆盖字段

- 来自 Binance 的价格字段
- 来自 Blockchain.com 的链上字段
- 来自 FRED 的宏观字段
- 情绪字段 `fear_greed_index`
- 网络难度字段 `difficulty`
- 包括 `active_addresses_change`、`fees_per_tx`、`difficulty_change` 在内的衍生特征

## 5. 当前输出状态

- 当前保留快照中的导出特征列缺失率为 0
- 导出结果已经是可直接建模的最终表，而不是原始特征拼接结果
- 可用样本起点晚于原始起始日期，原因来自 warm-up 和数据源覆盖约束

## 6. 异常值策略

当前构建器不做截尾、裁剪或缩放。

这些步骤仍然属于下游建模阶段。
