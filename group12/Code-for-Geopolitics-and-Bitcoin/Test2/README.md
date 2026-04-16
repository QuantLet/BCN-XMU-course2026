<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: Mechanism_Analysis_BTC_DXY

Published in: Digital Assets and Geopolitical Risk Research Project 2026

Description: This Quantlet analyzes the dynamic relationship between Bitcoin and the U.S. Dollar Index (DXY) during geopolitical shocks. It employs Pelt change point detection to identify mechanism shifts and utilizes an LSTM neural network combined with SHAP values to quantify non-linear feature attribution. The research demonstrates how Bitcoin's pricing logic transitions from a liquidity-driven asset to a desensitized "digital gold" proxy during extreme market stress.

Keywords: Bitcoin, DXY, Geopolitical Risk, LSTM-SHAP, Change Point Detection, Mechanism Analysis, Digital Gold

Author: Cong Peng

Submitted: Wednesday, April 15, 2026

Data Source: Daily historical price data for BTC-USD and DX-Y.NYB retrieved from Yahoo Finance via the yfinance API.

Input: Logarithmic returns of Bitcoin (BTC_Ret) and the U.S. Dollar Index (DXY_Ret) aligned through an inner join to synchronize global trading days.

Output: Statistical correlation coefficients, Pelt detection inflection points, LSTM model performance metrics, and SHAP summary plots quantifying feature importance.

Example: A comparison of SHAP values before and after the March 2026 shock, showing a 17.60% decrease in the DXY's non-linear contribution to Bitcoin's price volatility despite increased market turbulence.

```
<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/BCN-XMU-course2026/main/group12/Code-for-Geopolitics-and-Bitcoin/Test2/output_2_1.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/BCN-XMU-course2026/main/group12/Code-for-Geopolitics-and-Bitcoin/Test2/output_7_1.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/BCN-XMU-course2026/main/group12/Code-for-Geopolitics-and-Bitcoin/Test2/output_8_0.png" alt="Image" />
</div>

