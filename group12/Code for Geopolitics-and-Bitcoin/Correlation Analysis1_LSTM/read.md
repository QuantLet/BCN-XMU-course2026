# LSTM Forecasting for Daily WTI and BTC Prices

This directory contains a runnable PyTorch LSTM project for univariate daily price forecasting on:

- WTI crude spot price (daily, EIA `RWTCd.xls`)
- Bitcoin price (daily, from 2013-01-01 onward)

## Structure

- `train_lstm_forecast.py`: end-to-end pipeline (cleaning, training, evaluation, forecasting, plots)
- `RWTCd.xls`: EIA WTI source file (auto-download if missing)
- `BTC_USD_Bitfinex_historical_data.csv`: BTC historical source file
- `data/`: cleaned daily series
- `outputs/`: predictions, metrics, and figures

## Install

```bash
cd /home/lx/LSTM
pip install -r requirements.txt
```

## Run

```bash
cd /home/lx/LSTM
python train_lstm_forecast.py
```

Example configuration:

```bash
python train_lstm_forecast.py \
  --lookback 60 \
  --epochs 80 \
  --target-mode diff \
  --forecast-days 30 \
  --test-ratio 0.2
```

`--target-mode` options:
- `diff`: predict next-day price difference (recommended, works with negative WTI history)
- `log_return`: predict next-day log return (requires strictly positive prices)

## Outputs

- `data/wti_daily_clean.csv`
- `data/btc_daily_clean.csv`
- `outputs/wti_test_predictions.csv`
- `outputs/wti_future_30d.csv`
- `outputs/wti_forecast.png`
- `outputs/btc_test_predictions.csv`
- `outputs/btc_future_30d.csv`
- `outputs/btc_forecast.png`
- `outputs/metrics.json`

`metrics.json` includes `MAE`, `RMSE`, and `MAPE_percent` for each asset.

## Data sources

- WTI: EIA  
  `https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=RWTC&f=D`
- BTC: local CSV in this project (starting from 2013-01-01)
