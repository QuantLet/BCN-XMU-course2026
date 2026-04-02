# BTC 数据集规范

英文版：[data_spec.md](data_spec.md)

## 1. 适用范围

- 资产：`BTC`
- 频率：日频
- 原始起始日期：`2018-01-01`
- 最终数据包：`processed/final_dataset_v2_plus.csv`
- 主要用途：为 baseline、RNN 和 SHAP 解释提供干净、可复用、可防泄露的特征数据集

## 2. 标签定义

```text
target_t = 1[close_t > close_{t-1}]
```

## 3. 防泄露规则

所有建模特征只能使用预测时点之前已经可获得的信息。

默认规则：

```text
all features should be shifted by 1 period when needed
```

## 4. 当前交付物

- `processed/final_dataset_v2_plus.csv`
- `docs/data_dictionary_v2_plus.md`
- `docs/qa_report_v2_plus.md`
- `docs/source_note_v2_plus.md`
- `docs/btc_project_report.docx`

## 5. 当前特征范围

- 价格输入列
- 价格衍生列
- 链上字段
- 宏观字段
- `fear_greed_index`
- `active_addresses_change`
- `fees_per_tx`
- `difficulty`
- `difficulty_change`
- `target`

## 6. 建模约束

- 不允许使用未来价格或未来收益
- 滚动统计量只能基于历史窗口计算
- 时间可得性不明确的外部特征必须采取保守滞后
- 最终可用样本起点可能晚于原始起始日期，因为 warm-up 和数据源覆盖约束会裁掉前段样本

## 7. 当前数据源补充说明

- `fear_greed_index` 的原始覆盖从 `2018-02-01` 开始
- `difficulty` 来自 `Blockchain.com Charts`
