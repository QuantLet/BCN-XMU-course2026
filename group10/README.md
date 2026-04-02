# The Asymmetric Spillover Effects of Underlying Asset Volatility on NFT Liquidity

## 📖 Project Overview
This repository contains the complete empirical data pipeline and analysis code for investigating the asymmetric spillover effects of Ethereum (ETH) volatility on Non-Fungible Token (NFT) liquidity. Utilizing high-frequency Binance data and large-scale Kaggle NFT transaction records, this project constructs an end-to-end econometric pipeline. It leverages Vector Autoregression (VAR) models and Jordà's Local Projections (LP) to identify the robust causal response of Blue-chip vs. Tail NFT collections to severe macroeconomic shocks.

## 📊 Data Pipeline Architecture

The project is structured into a rigorous five-step empirical pipeline:

### 1. Macro & Micro Data Acquisition
- **`1a_historical_macro.py`**: Fetches 5-minute interval historical ETH/USDT candlestick data from the Binance API during the 2022 crypto crash period. It aggregates this high-frequency data to calculate daily Realized Volatility (RV), Bipower Variation (BPV), and Jumps, then merges it with the CBOE VIX index downloaded via Yahoo Finance.
- **`1b_kaggle_process.py`**: Processes a massive (1.15GB) raw Kaggle dataset of historical NFT transactions (`kaggle_raw_nft.csv`). It converts Wei to ETH, filters outliers, and categorizes NFT collections into "Blue-chip" (Top 5%) and "Tail" (Bottom 50%) markets based on cumulative historical volume.

### 2. Panel Construction
- **`2_panel_construction.py`**: Merges the daily macroeconomic/volatility indicators (`1a_macro_eth_volatility.csv`) with the micro-level NFT liquidity features (`1b_nft_liquidity_REAL.csv`). This script incorporates essential econometric transformations: log transformations for heavily skewed volume data `Ln(1+Volume)` and robust Last-Observation-Carried-Forward (LOCF) strategies for filling missing floor prices.

### 3. Econometric Modeling & Visualization
- **`3_var_model_irf.py`**: Standardizes the time series and estimates a Vector Autoregression (VAR) model. It utilizes Cholesky decomposition to trace Orthogonalized Impulse Response Functions (IRFs), plotting how a 1 standard deviation shock in ETH Realized Volatility affects Blue-chip vs. Tail NFT volumes over a 10-day horizon.
- **`4_local_projections_irf.py`**: Provides rigorous robustness checks by implementing Jordà's (2005) Local Projections. It estimates the impulse responses using direct OLS horizons equipped with Newey-West HAC robust standard errors to account for serial correlation and heteroskedasticity.

## 🚀 How to Run

1. Ensure the required Python packages are installed:
   ```bash
   pip install pandas numpy matplotlib statsmodels yfinance requests
   ```
2. Place the primary 1.15GB raw NFT dataset (`kaggle_raw_nft.csv`) inside the `data/` directory. *(Note: This file is excluded from GitHub due to size).*
3. Execute the scripts sequentially:
   ```bash
   python scripts/1a_historical_macro.py
   python scripts/1b_kaggle_process.py
   python scripts/2_panel_construction.py
   python scripts/3_var_model_irf.py
   python scripts/4_local_projections_irf.py
   ```
4. Find the resulting high-definition econometric charts in the `figures/` directory.

## 📁 Repository Structure
```text
.
├── README.md                           # This documentation
├── scripts/                            # Empirical Python scripts
├── data/                               # Intermediate and final compiled datasets
├── figures/                            # Output IRF plots and robustness charts
└── The_Asymmetric_Spillover_Effects... # LaTeX Presentation and slides
```

## ⚖️ License
This project is intended for academic research and educational purposes.
