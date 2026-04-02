# BTC Dataset Specification

Chinese version: [data_spec_zh.md](data_spec_zh.md)

## 1. Scope

- Asset: `BTC`
- Frequency: `Daily`
- Raw start date: `2018-01-01`
- Final package: `processed/final_dataset_v2_plus.csv`
- Primary use: provide a clean, reusable, leak-aware feature dataset for baseline models, RNN models, and SHAP interpretation

## 2. Label Definition

```text
target_t = 1[close_t > close_{t-1}]
```

## 3. Leakage Rule

All modeling features must use only information available before the prediction timestamp.

Default rule:

```text
all features should be shifted by 1 period when needed
```

## 4. Current Deliverables

- `processed/final_dataset_v2_plus.csv`
- `docs/data_dictionary_v2_plus.md`
- `docs/qa_report_v2_plus.md`
- `docs/source_note_v2_plus.md`
- `docs/btc_project_report.docx`

## 5. Current Feature Set

- price input fields
- price-derived fields
- on-chain fields
- macro fields
- `fear_greed_index`
- `active_addresses_change`
- `fees_per_tx`
- `difficulty`
- `difficulty_change`
- `target`

## 6. Modeling Constraints

- no future prices or future returns may be used
- rolling statistics must use historical windows only
- features with uncertain release timing must be lagged conservatively
- the final usable sample range may start later than the raw start date because of warm-up and source coverage constraints

## 7. Current Source Notes

- `fear_greed_index` starts from `2018-02-01`
- `difficulty` is sourced from `Blockchain.com Charts`
