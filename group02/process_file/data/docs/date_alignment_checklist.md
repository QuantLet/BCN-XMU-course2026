# Date Alignment Checklist

Chinese version: [date_alignment_checklist_zh.md](date_alignment_checklist_zh.md)

## 1. Purpose

This checklist is used for manual spot checks of the latest promoted package, `processed/final_dataset_v2_plus.csv`.

It focuses on hidden but high-impact alignment risks:

- whole-column one-day shifts
- same-day feature leakage into target rows
- future values being filled backward after merges

## 2. Core Semantics to Check

For the current `V2-plus` lane:

1. `target` where `date=t` means `1[close_t > close_{t-1}]`
2. all modeling inputs where `date=t` must come from `t-1` or earlier
3. `fear_greed_index`, `difficulty`, and all derived features must also respect the same lag rule

## 3. Spot-Check Targets

For a sampled date `t`, compare:

- price fields in `raw/raw_btc_ohlcv.csv`
- on-chain base fields in `raw/raw_blockchain_metrics.csv`
- macro values in `raw/raw_fred_macro.csv`
- sentiment values in `raw/raw_fear_greed_index.csv`
- difficulty values in `raw/raw_blockchain_difficulty.csv`
- final row where `date=t` in `processed/final_dataset_v2_plus.csv`

## 4. V2-plus Specific Checks

- [ ] `fear_greed_index` where `date=t` matches the most recent value known at `t-1`
- [ ] `difficulty` where `date=t` matches raw difficulty at `t-1`
- [ ] `difficulty_change` where `date=t` reflects the change already knowable by `t-1`
- [ ] `active_addresses_change` and `fees_per_tx` remain lagged after derivation

## 5. When to Run

Run this checklist after:

- adding a new source
- changing lag semantics
- refreshing the official handoff package
- before passing the dataset to downstream teammates
