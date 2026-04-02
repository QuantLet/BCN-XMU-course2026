# 数据源清单

英文版：[source_inventory.md](source_inventory.md)

## 1. 文档目的

本文档用于记录当前 BTC 数据集正式交付包所使用的数据源。

## 2. 项目约束

- 资产：`BTC`
- 频率：日频
- 原始起始日期：`2018-01-01`
- 最终输出：`processed/final_dataset_v2_plus.csv`
- 防泄露规则：全部建模特征在需要时统一采用保守的 1 期滞后

## 3. 当前数据源表

| 优先级 | 数据类别 | 数据源 | 角色 | 当前使用字段 | 是否需要 Key | 说明 |
|---|---|---|---|---|---|---|
| P0 | BTC OHLCV | Binance Spot API | 主源 | `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `trade_count` | 否 | 当前 BTC 价格主干。 |
| P0 | 链上 | Blockchain.com Charts API | 主源 | `active_addresses`, `tx_count`, `hash_rate`, `fees`, `difficulty` | 否 | 当前链上主源。 |
| P0 | 宏观 | FRED | 主源 | `vix`, `sp500`, `dxy_proxy` | 否 | 当前宏观主源。 |
| P0 | 情绪 | Alternative.me | 主源 | `fear_greed_index` | 否 | 当前情绪特征来源。 |

## 4. 当前字段映射

| 最终字段组 | 来源 |
|---|---|
| 价格列 | Binance |
| 链上基础列 | Blockchain.com |
| 宏观列 | FRED |
| 情绪列 | Alternative.me |
| 价格衍生特征 | 由 Binance 价格数据派生 |
| 链上衍生特征 | 由 Blockchain.com 字段派生 |

## 5. 当前数据源结论

当前正式交付包使用：

- `Binance` 提供 BTC 价格数据
- `Blockchain.com` 提供链上活跃度与网络难度
- `FRED` 提供宏观变量
- `Alternative.me` 提供情绪特征
