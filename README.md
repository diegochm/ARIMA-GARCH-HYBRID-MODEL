# ARIMA-GARCH-HYBRID-MODEL
This project applies a hybrid ARIMA–GARCH framework to model and forecast the price dynamics of Silver Futures (SI=F)
 
It combines the strength of ARIMA for mean forecasting and GARCH for conditional volatility estimation, creating a complete statistical benchmark for financial time series.

---

🧩 Project Overview

The notebook performs the following stages:

1. Data Collection**
   - Silver Futures (`SI=F`) downloaded from Yahoo Finance via `yfinance`.
   - Date range: 2018–2023.

2. Preliminary Analysis**
   - Stationarity test (ADF).
   - First-order differencing to obtain stationary series.
   - ACF and PACF analysis to identify ARIMA orders.

3. ARIMA Modeling**
   - Tests ARIMA(2,1,2) and ARIMA(1,1,1).
   - Diagnostics with residual plots and Ljung–Box test.
   - Interpretation of coefficients and model significance.

4. GARCH Modeling**
   - Fits a GARCH(1,1) model on ARIMA residuals.
   - Interprets volatility persistence and clustering effects.
   - Generates conditional volatility series.

5. Hybrid Forecast**
   - Combines ARIMA mean forecast + GARCH variance forecast.
   - Builds price projection bands (±2σ).
   - Compares predicted and actual prices (RMSE, MAE, coverage).

6. Rolling Forecast Evaluation**
   - Rolling one-step-ahead ARIMA(1,0,1) + GARCH(1,1) over log returns.
   - Evaluates both return-level and price-level predictive accuracy.
   - Visualizes dynamic prediction and volatility uncertainty.

---

🧠 Key Insights

- ARIMA captures short-term serial correlation (directional structure).
- GARCH captures time-varying volatility (risk clustering).
- The hybrid model produces smoother forecasts — useful as a risk and volatility benchmark, though limited for short-term directional trading.
- Extensions could include:
  - Cointegration + ARMA + GARCH** for mean-reverting spreads.
  - Machine Learning (Random Forest, XGBoost)** over ARIMA–GARCH features.
  - Volatility-based position sizing** for risk management.

---

 🧮 Model Summary

| Component | Model | Purpose |
|------------|--------|----------|
| ARIMA | (1,1,1) | Captures conditional mean (trend and short-term memory). |
| GARCH | (1,1) | Captures conditional variance (volatility clustering). |
| Forecast Horizon | 10 days (static), 1-step rolling | Hybrid predictive analysis. |

---

📊 Results (Summary)

- ADF test** → Non-stationary prices, stationary first difference.  
- Best ARIMA model** → (1,1,1) based on AIC/BIC and residual tests.  
- GARCH(1,1)** → α=0.17, β=0.83 → high volatility persistence.  
- RMSE** | 4.15 USD | Average deviation of ~4 USD between predicted and actual prices. |
- MAE** | 3.74 USD | Typical absolute forecast error per observation. |
- Relative Error** | ≈ 18% | Moderate accuracy given silver’s high volatility. |
- Coverage (±2σ bands)** | ≈ 95–98% | Indicates GARCH volatility bands effectively captured market variability. |



---

💻 Installation

```bash
git clone https://github.com/<your-username>/ARIMA-GARCH-Hybrid.git
cd ARIMA-GARCH-Hybrid
pip install -r requirements.txt

📚 Author

Diego Choquesillo Martínez
📍 Melbourne, Australia
🎓 Quantitative Finance & Algorithmic Trading Portfolio
🔗 https://www.linkedin.com/in/diego-choquesillo-martinez/
