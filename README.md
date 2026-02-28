# Gold-Price-Forecasting using xLSTM (Matrix Memory) 

## Overview
This repository contains the implementation of my Master's Thesis project at the University of Thessaly. The project focuses on predicting the price of Gold using macroeconomic indicators. It leverages state-of-the-art Deep Learning architectures, specifically the **xLSTM (Extended LSTM)** family introduced in 2024, and evaluates their performance against traditional Standard LSTMs and classical statistical methods.

## Key Features & Innovation
- **xLSTM Implementation:** Custom PyTorch implementation of **sLSTM** (Exponential Gating) and **mLSTM** (Matrix Memory with Query-Key-Value mechanism).
- **Data Engineering:** Integration of local macroeconomic data (Bank of Greece: Interest Rates, Inflation) with global market data (Yahoo Finance API).
- **Cloud Computing:** Model training and evaluation were performed on Google Colab utilizing Cloud GPUs (NVIDIA T4) for accelerated deep learning operations.

## Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Data APIs:** yfinance
* **Environment:** Cloud PaaS (Google Colab)

## Data Sources
1. **Bank of Greece (BoG):** Monthly New Deposit Interest Rates and Consumer Price Index (Inflation).
2. **Yahoo Finance:** Gold Futures (GC=F), US Bond Yields, and Dollar Index (DXY).

## Experimental Results
The project evaluated four different approaches: Standard LSTM, sLSTM, mLSTM, and Classical Statistics (Linear/Logistic Regression).

### 1. Matrix Memory Superiority
The **mLSTM (Matrix Memory)** model significantly outperformed all other architectures in price prediction (Regression), achieving the lowest Root Mean Squared Error (RMSE). By storing states in a Matrix rather than a vector, it successfully maintained high-resolution memory of 12-month historical data.

### 2. Deep Learning vs. Classical Statistics
- **Price Prediction (Regression):** The mLSTM dynamically adapted to market volatility, drastically outperforming Linear Regression, which failed to model the market's non-linear nature.
- **Trend Prediction (Classification):** For simple binary trend prediction (Up/Down), Logistic Regression remained highly competitive, proving that simpler models are sufficient for less complex forecasting tasks.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/NickBisbi/Gold-Price-Forecasting-xLSTM.git](https://github.com/NickBisbi/Gold-Price-Forecasting-xLSTM.git)
