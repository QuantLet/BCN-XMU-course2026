# V2 Plus Source Note

Chinese version: [source_note_v2_plus_zh.md](source_note_v2_plus_zh.md)

## Goal

This note records the source decisions for `final_dataset_v2_plus.csv`.

The V2-plus path keeps both V1 and V2-core stable and adds one more low-risk on-chain feature source.

## Retained Sources

- `Binance` -> BTC daily OHLCV
- `Blockchain.com Charts` -> `active_addresses`, `tx_count`, `hash_rate`, `fees`
- `FRED` -> `vix`, `sp500`, `dxy_proxy`
- `Alternative.me Fear & Greed Index` -> `fear_greed_index`

## Added V2-plus Source

- `Blockchain.com Charts` -> `difficulty`

Why this source was accepted:

- It stays within the same free on-chain provider already used in the retained pipeline
- Daily coverage is compatible with the existing BTC dataset window
- The source is easier to explain and reproduce than a new paid API or a scraping-based source
- It adds a network-level signal that is distinct from the current activity and fee features

## Added V2-plus Derived Feature

- `difficulty_change` = 1-day percentage change in `difficulty`

Why this feature was accepted:

- Built directly from the retained `difficulty` series
- Safe under the existing `shift(1)` leakage rule
- Gives the modeling team both a level feature and a change feature

## Deferred Features

- `new_addresses`
- CRIX-like market proxy features that rely on restricted free history or extra fragile pipelines

Reason for deferral:

- They still do not meet the same reliability and reproducibility bar as the accepted V2-plus addition.
