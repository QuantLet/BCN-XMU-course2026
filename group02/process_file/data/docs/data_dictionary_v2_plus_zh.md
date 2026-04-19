# 数据字典（V2 Plus）

英文版：[data_dictionary_v2_plus.md](data_dictionary_v2_plus.md)

| 列名 | 含义 | 建模时是否滞后 |
|---|---|---|
| `date` | 数据集行日期 | 否 |
| `open` | Binance BTC 开盘价（滞后后进入建模） | 是 |
| `high` | Binance BTC 最高价（滞后后进入建模） | 是 |
| `low` | Binance BTC 最低价（滞后后进入建模） | 是 |
| `close` | Binance BTC 收盘价（滞后后进入建模） | 是 |
| `volume` | Binance BTC 基础成交量（滞后后进入建模） | 是 |
| `quote_volume` | Binance BTC 计价成交量（滞后后进入建模） | 是 |
| `trade_count` | Binance BTC 日成交笔数（滞后后进入建模） | 是 |
| `ret_1d` | BTC 1 日收益率 | 是 |
| `ret_3d` | BTC 3 日收益率 | 是 |
| `volatility_7d` | 7 日滚动收益率标准差 | 是 |
| `ma_ratio_7_30` | 7 日均线与 30 日均线之比 | 是 |
| `high_low_spread` | 日内波动比例 `(high-low)/close` | 是 |
| `volume_change` | 成交量单日百分比变化 | 是 |
| `active_addresses` | Blockchain.com 活跃地址数 | 是 |
| `tx_count` | Blockchain.com 交易笔数 | 是 |
| `hash_rate` | Blockchain.com 全网算力 | 是 |
| `fees` | Blockchain.com 手续费总额 | 是 |
| `vix` | FRED `VIXCLS` 指标值 | 是 |
| `sp500_return` | 基于 FRED `SP500` 计算的日收益率 | 是 |
| `dxy_proxy` | FRED `DTWEXBGS` 美元指数代理变量 | 是 |
| `active_addresses_change` | 活跃地址数单日百分比变化 | 是 |
| `fees_per_tx` | 日手续费总额除以交易笔数 | 是 |
| `fear_greed_index` | Alternative.me 加密市场恐惧与贪婪指数 | 是 |
| `difficulty` | Blockchain.com 网络难度 | 是 |
| `difficulty_change` | 网络难度单日百分比变化 | 是 |
| `target` | `1[close_t > close_{t-1}]`，否则为 0 | 否 |
