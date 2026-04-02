# V2 Plus 数据源说明

英文版：[source_note_v2_plus.md](source_note_v2_plus.md)

## 目标

本文档用于记录 `final_dataset_v2_plus.csv` 的当前数据源决策。

## 当前保留来源

- `Binance` -> BTC 日频 OHLCV
- `Blockchain.com Charts` -> `active_addresses`, `tx_count`, `hash_rate`, `fees`
- `FRED` -> `vix`, `sp500`, `dxy_proxy`
- `Alternative.me Fear & Greed Index` -> `fear_greed_index`

## 当前新增来源

- `Blockchain.com Charts` -> `difficulty`

选择该来源的原因：

- 继续使用当前已经保留的免费链上提供方
- 日频覆盖与现有 BTC 数据窗口兼容
- 相比新的付费 API 或脆弱爬虫方案，更容易解释与复现
- 能补充一个不同于活跃度和手续费的网络层信号

## 当前新增衍生特征

- `difficulty_change` = `difficulty` 的单日百分比变化

选择该特征的原因：

- 直接基于当前保留的 `difficulty` 序列构造
- 符合现有 `shift(1)` 防泄露规则
- 同时提供难度水平和难度变化两类信息

## 当前未纳入的特征

- `new_addresses`
- 依赖受限免费历史或额外脆弱流程的 CRIX 类市场代理特征

未纳入原因：

- 在当前正式交付包中，它们还没有达到与已保留来源相同的稳定性、可解释性和可复现性要求。
