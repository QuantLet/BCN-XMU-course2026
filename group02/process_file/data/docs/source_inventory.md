# Source Inventory

Chinese version: [source_inventory_zh.md](source_inventory_zh.md)

## 1. Purpose

This document records the source choices used by the current BTC dataset package.

## 2. Project Constraints

- Asset: `BTC`
- Frequency: `Daily`
- Raw start date: `2018-01-01`
- Final output: `processed/final_dataset_v2_plus.csv`
- Leakage rule: all modeling features use a conservative one-period lag when needed

## 3. Current Source Table

| Priority | Data Type | Source | Role | Fields in Use | Key Required | Notes |
|---|---|---|---|---|---|---|
| P0 | BTC OHLCV | Binance Spot API | Primary | `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `trade_count` | No | Current BTC price backbone. |
| P0 | On-chain | Blockchain.com Charts API | Primary | `active_addresses`, `tx_count`, `hash_rate`, `fees`, `difficulty` | No | Current on-chain source. |
| P0 | Macro | FRED | Primary | `vix`, `sp500`, `dxy_proxy` | No | Current macro source. |
| P0 | Sentiment | Alternative.me | Primary | `fear_greed_index` | No | Current sentiment source. |

## 4. Current Feature Mapping

| Final Field Group | Source |
|---|---|
| Price fields | Binance |
| On-chain base fields | Blockchain.com |
| Macro fields | FRED |
| Sentiment field | Alternative.me |
| Price-derived features | Derived from Binance price data |
| On-chain derived features | Derived from Blockchain.com fields |

## 5. Current Source Decision

The current formal package uses:

- `Binance` for BTC price data
- `Blockchain.com` for on-chain activity and network difficulty
- `FRED` for macro variables
- `Alternative.me` for sentiment
