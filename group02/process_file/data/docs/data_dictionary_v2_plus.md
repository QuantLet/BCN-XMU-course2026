# Data Dictionary (V2 Plus)

| Column | Description | Shifted for Modeling |
|---|---|---|
| `date` | Dataset row date | No |
| `open` | Lagged BTC open price from Binance | Yes |
| `high` | Lagged BTC high price from Binance | Yes |
| `low` | Lagged BTC low price from Binance | Yes |
| `close` | Lagged BTC close price from Binance | Yes |
| `volume` | Lagged BTC traded volume from Binance | Yes |
| `quote_volume` | Lagged BTC quote volume from Binance | Yes |
| `trade_count` | Lagged Binance trade count | Yes |
| `ret_1d` | 1-day BTC return | Yes |
| `ret_3d` | 3-day BTC return | Yes |
| `volatility_7d` | 7-day rolling close-return volatility | Yes |
| `ma_ratio_7_30` | 7-day / 30-day moving-average ratio | Yes |
| `high_low_spread` | Intraday high-low spread divided by close | Yes |
| `volume_change` | 1-day percentage change in volume | Yes |
| `active_addresses` | Blockchain.com unique active addresses | Yes |
| `tx_count` | Blockchain.com confirmed transactions per day | Yes |
| `hash_rate` | Blockchain.com network hash rate | Yes |
| `fees` | Blockchain.com total transaction fees | Yes |
| `vix` | FRED VIXCLS index level | Yes |
| `sp500_return` | S&P 500 daily return derived from FRED SP500 | Yes |
| `dxy_proxy` | FRED DTWEXBGS dollar index proxy | Yes |
| `active_addresses_change` | 1-day percentage change in active addresses | Yes |
| `fees_per_tx` | Daily fees divided by transaction count | Yes |
| `fear_greed_index` | Alternative.me crypto fear and greed index | Yes |
| `difficulty` | Blockchain.com network difficulty | Yes |
| `difficulty_change` | 1-day percentage change in network difficulty | Yes |
| `target` | 1 if close_t > close_{t-1}, else 0 | No |
