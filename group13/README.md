# DeFi User Credit Scoring & Risk Detection System 

> **A Dual-Layer Risk Assessment Framework Using PCA and Isolation Forest**
>
>  *Blockchain & Cryptocurrency Analysis - Course Project 2026*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![DeFi](https://img.shields.io/badge/DeFi-Aave_V2-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Project Overview

This project develops a robust, data-driven credit evaluation framework for Decentralized Finance (DeFi) by analyzing **2.94M+ transaction logs** from Aave V2 on the Polygon network. 

Initially, we explored price prediction models; however, moving away from volatile price predictions—which often yield suboptimal results due to static data constraints—we shifted our focus to **long-term on-chain behavioral analytics** to provide a more reliable and systemic risk assessment.

DeFi protocols currently operate without identities or credit histories, heavily relying on capital-inefficient over-collateralization. To bridge this gap, our framework extracts multi-dimensional behavioral features and evaluates users through a **Dual-Layer Analytics System**.

## 🧠 Methodology & Architecture

Our framework consists of two core machine learning layers to comprehensively evaluate user risk:

### Layer 1: PCA-Based Credit Scoring (The "Normal" Risk)
We utilized **Principal Component Analysis (PCA)** to synthesize complex on-chain behaviors (e.g., historical repayment ratios, deposit volumes, account activity) into a standardized, interpretable credit score.
* **Score Range:** 300 - 850 (Mimicking traditional FICO scores).
* **Validation:** Historical data proves a strong positive correlation between higher PCA scores and higher actual repayment rates.
<img width="3000" height="1500" alt="deposit_distribution_academic" src="https://github.com/user-attachments/assets/8ceb9f57-8006-4a67-b2c9-da05b8bb2813" />
<img width="3241" height="1741" alt="credit_score_distribution" src="https://github.com/user-attachments/assets/ce3907b3-3d84-4d9f-bd72-3cd0989374e4" />

### Layer 2: Isolation Forest Anomaly Detection (The "Systemic" Risk)
High credit scores do not guarantee safety in DeFi. We integrated the **Isolation Forest** unsupervised algorithm to detect systemic anomalies.
* It captures extreme behavioral volatility (e.g., sudden massive borrowing, high-frequency irregular transactions).
* **Output:** Anomaly scores where negative values indicate extreme, suspicious behaviors.

---

## 📊 The 4-Quadrant Risk Matrix

By combining the two layers, we segment users into a dual-axis Risk Matrix, yielding critical insights for DeFi protocols:

* **Q1: High-Quality Users (Prime Retail)** - High Credit + Normal Behavior. Reliable borrowers suitable for preferential lending rates.
* **Q2: Subprime Users** - Low Credit + Normal Behavior. Predictably poor credit, requiring standard over-collateralization.
* **Q3: Hidden Whales ⚠️ (Danger Zone)** - High Credit + Abnormal Behavior. Users who historically repay well but currently exhibit extreme, volatile behaviors. *Our model successfully flagged 10 out of 1391 active borrowers as potential market manipulators/whales.*
* **Q4: High-Risk Users** - Low Credit + Abnormal Behavior. Toxic assets that should be flagged for immediate review.
<img width="3210" height="2041" alt="risk_matrix" src="https://github.com/user-attachments/assets/ec51a734-4609-49fe-ba69-2faaed7dc8b4" />
<img width="4098" height="1741" alt="model_validation" src="https://github.com/user-attachments/assets/c57abed5-220f-447e-b535-f4ff177164e2" />

## 🛠️ Tech Stack & Feature Engineering

* **Data Source:** Polygon On-Chain JSON data (Aave V2 Protocol).
* **Tools:** `Python`, `Pandas`, `NumPy`, `Scikit-Learn`, `Seaborn`, `Matplotlib`.
* **Feature Engineering Pipeline:**
  1. **Log Transformation:** `log(x+1)` to normalize skewed crypto asset distributions.
  2. **Winsorization:** Capped outliers at 1st and 99th percentiles to handle extreme whales.
  3. **Standard Scaling:** Z-score normalization for PCA stability.

## 📁 Repository Structure

```text
├── data/                   # Raw JSON data and processed CSV files 
├── notebooks/              # Jupyter Notebooks for EDA and Model Training
│   ├── 01_data_cleaning.ipynb
│   ├── 02_pca_credit_scoring.ipynb
│   └── 03_isolation_forest_risk.ipynb
├── src/                    # Python scripts for data processing and model execution
├── images/                 # Generated visualizations (Risk Matrix, Distributions)
├── slides/                 # Course presentation PDF (DeFi User Credit Scoring.pdf)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
