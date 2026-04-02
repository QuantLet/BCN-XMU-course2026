# Cleaning and Missing-Value Rules

Chinese version: [data_cleaning_rules_zh.md](data_cleaning_rules_zh.md)

## 1. Purpose

This document records the cleaning behavior used by the current BTC dataset builder.

## 2. Current Builder

- `notebooks/build_dataset_v2_plus.py`

## 3. Current Rules

- normalize `date` to `%Y-%m-%d`
- sort all rows by `date`
- keep the last row for duplicate dates
- coerce source columns with `pd.to_numeric(..., errors="coerce")`
- remove the latest Binance date before final export
- build price-derived features on the trimmed price frame
- forward-fill retained external sources before the final lag step
- apply a one-period lag to all modeling features
- drop rows with missing required features or target after lagging

## 4. Current Covered Sources and Features

- price fields from Binance
- on-chain fields from Blockchain.com
- macro fields from FRED
- sentiment field `fear_greed_index`
- on-chain difficulty field `difficulty`
- derived features including `active_addresses_change`, `fees_per_tx`, and `difficulty_change`

## 5. Current Output Status

- the exported feature columns have zero missing rate in the kept snapshot
- the exported table is already model-ready rather than a raw feature dump
- the usable range begins after warm-up and source-coverage constraints are satisfied

## 6. Outlier Policy

The current builder does not perform clipping, winsorization, or scaling.

Those operations remain downstream modeling concerns.
