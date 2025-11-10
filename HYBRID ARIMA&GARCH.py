#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade  yfinance')


# In[2]:


import numpy as np
import pandas as pd
import yfinance as yf

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("seaborn-darkgrid")

import warnings
warnings.filterwarnings("ignore")
from datetime import datetime


# In[3]:


get_ipython().system('pip install pandas-datareader')
from statsmodels.tsa.arima.model import ARIMA


# In[4]:


data=yf.download("SI=F",start="2018-01-01", end="2023-12-01",auto_adjust=True)
data.index=pd.to_datetime(data.index)
data.head()


# In[5]:


data.columns=data.columns.droplevel(1)


# In[6]:


print(data.head())
print(data.columns)


# In[7]:


rolling_window=int(len(data)*0.70)


# In[8]:


plt.figure(figsize=(10,7))
plt.plot(data["Close"],"blue")
plt.title("Silver Price Futures" , fontsize=14)
plt.xlabel("Years",fontsize=12)
plt.ylabel("Price",fontsize=12)
plt.show()


# In[9]:


from statsmodels.tsa.stattools import adfuller

result=adfuller(data["Close"])

if result[1] < 0.05 :
    print("Stationary, p-value =%.2f <=0.05" %result[1])
else:
    print("No Stationary, p-value [ %.2f > 0.05]" % result[1])


# In[10]:


stationary_series=data["Close"].diff().dropna()

plt.figure(figsize=(10,7))
stationary_series.plot()
plt.title("First order differenced series")
plt.xlabel("Year")
plt.ylabel("Price")
plt.show()


# In[11]:


result=adfuller(stationary_series)

if result[1] < 0.05 :
    print("Stationary, p-value =%.2f <=0.05" %result[1])
else:
    print("No Stationary, p-value [ %.2f > 0.05]" % result[1])


# In[12]:


fig,(ax1,ax2)= plt.subplots(2,1, figsize=(10,7))

plot_pacf(data["Close"][:rolling_window].diff().dropna(),lags=20,ax=ax1)
ax1.set_xlabel("Lags")
ax1.set_ylabel("Partial Autocorrelation")

plot_acf(data["Close"][:rolling_window].diff().dropna(),lags=20,ax=ax2)
ax2.set_xlabel("Lags")
ax2.set_ylabel("Autocorrelation")

plt.tight_layout()
plt.show()


# In[13]:


warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima.model.ARIMA',
                        FutureWarning)


# In[14]:


model=ARIMA(data["Close"][:rolling_window],order=(2,1,2))
model_fit_0=model.fit()
print(model_fit_0.params.round(2))


# In[15]:


model_fit_0.plot_diagnostics(figsize=(10, 8))
plt.show()


# In[16]:


from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(model_fit_0.resid, lags=10))

model_fit_0.summary()


# In[ ]:





# In[17]:


model=ARIMA(data["Close"][:rolling_window],order=(1,1,1))
model_fit_1=model.fit()
print(model_fit_1.params.round(2))


# In[18]:


model_fit_1.plot_diagnostics(figsize=(10, 8))
plt.show()


# In[19]:


from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(model_fit_1.resid, lags=10))

model_fit_1.summary()


# model_fit_2.summary()

# In[20]:


get_ipython().system('pip install arch')
from arch import arch_model


# In[21]:


residuals=model_fit_1.resid


# In[22]:


model_g=arch_model(residuals, mean="Zero",vol="GARCH",p=1,q=1)
garch_fit = model_g.fit(update_freq=10)


# In[23]:


print(garch_fit.summary())


# In[24]:


H=10
forecast_arima=model_fit_1.get_forecast(steps=H)
mean_forecast=forecast_arima.predicted_mean


# In[25]:


volatility_forecast = garch_fit.forecast(horizon=H).variance
volatility = volatility_forecast.iloc[-1]
volatility.index=mean_forecast.index
cum_mean = np.cumsum(mean_forecast)
cum_var  = np.cumsum(volatility)


# In[26]:


last_price = data["Close"].iloc[rolling_window-1]
price_forecast = last_price * np.exp(cum_mean)
upper_band = last_price * np.exp(cum_mean + 2*np.sqrt(cum_var))
lower_band = last_price * np.exp(cum_mean - 2*np.sqrt(cum_var))


# In[27]:


mask = ~(price_forecast.isna() | upper_band.isna() | lower_band.isna())
pf, ub, lb = price_forecast[mask], upper_band[mask], lower_band[mask]


# In[28]:


plt.figure(figsize=(10,5))
plt.plot(price_forecast, label="Precio proyectado (ARIMA)")
plt.fill_between(price_forecast.index, lower_band, upper_band,
                 color='lightblue', alpha=0.5, label='±2σ (GARCH)')
plt.title("ARIMA prices forecast(1,1,1) + GARCH(1,1)")
plt.xlabel("Future days")
plt.ylabel("Silve price")
plt.legend()
plt.show()


# In[31]:


from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

horizon=1
start=rolling_window
window = rolling_window
log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()


arima_pred = []
garch_std = []
real_values = []


for i in tqdm(range(start, len(log_returns) - horizon)):
    serie = log_returns[i - rolling_window:i]
    
    # ARIMA(1,1,1)
    model = ARIMA(serie, order=(1,1,1))
    model_fit = model.fit()
    forecast_arima = model_fit.get_forecast(steps=horizon)
    pred = forecast_arima.predicted_mean.iloc[0]
    
    # GARCH(1,1)
    residuals = model_fit.resid
    model_g = arch_model(residuals, vol="GARCH", p=1, q=1)
    garch_fit = model_g.fit(disp="off")
    forecast_vol = garch_fit.forecast(horizon=1).variance.values[-1, 0]
    sigma = np.sqrt(forecast_vol)
    
    # Save forecast
    arima_pred.append(pred)
    garch_std.append(sigma)
    real_values.append(log_returns.iloc[i + horizon])




index_forecast = log_returns.index[start:len(arima_pred) + start]
last_price = data["Close"].iloc[start - 1]


forecast_prices = [last_price * np.exp(arima_pred[0])]
for i in range(1, len(arima_pred)):
    next_price = forecast_prices[-1] * np.exp(arima_pred[i])
    forecast_prices.append(next_price)

# Confidence Bands
upper_band = [p * np.exp(2 * s) for p, s in zip(forecast_prices, garch_std)]
lower_band = [p * np.exp(-2 * s) for p, s in zip(forecast_prices, garch_std)]


#


print(len(index_forecast), len(forecast_prices), len(upper_band), len(lower_band))





x = index_forecast.to_list()

plt.figure(figsize=(12,6))
plt.plot(x, forecast_prices, label="Forecast (rolling)")
plt.fill_between(x, lower_band, upper_band, alpha=0.3, label="±2σ (GARCH)")
real_prices = data.loc[index_forecast, "Close"].values 
plt.plot(x, real_prices, label="Real Price", linestyle='--', color='black')
plt.title("Rolling Forecast con ARIMA(1,0,0) + GARCH(1,1)")
plt.xlabel("date")
plt.ylabel("Price")
plt.legend()
plt.show()


# In[36]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# convert to arrays
real_prices = np.array(real_prices)
forecast_prices = np.array(forecast_prices)

# RMSE y MAE
rmse = np.sqrt(mean_squared_error(real_prices, forecast_prices))
mae = mean_absolute_error(real_prices, forecast_prices)

print(f"📊 RMSE: {rmse:.4f}")
print(f"📊 MAE: {mae:.4f}")


# In[37]:


def analyse_strategy(returns, rf=0):
  
    returns = returns.dropna()
    cumulative_returns = (1 + returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    # Calculate for a year
    days = (returns.index[-1] - returns.index[0]).days
    years = days / 365
    
    # CAGR
    cagr = (cumulative_returns.iloc[-1])**(1/years) - 1
    
    # Anual volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio 
    sharpe = (returns.mean() - rf/252) / returns.std() * np.sqrt(252)
    
    # Drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / rolling_max) - 1
    max_dd = drawdown.min()
    
    # Results
    print(f"📊 Total Return: {total_return:.2%}")
    print(f"📈 CAGR: {cagr:.2%}")
    print(f"📉 Anual Volatility: {volatility:.2%}")
    print(f"⚖️ Sharpe Ratio (rf={rf:.1%}): {sharpe:.2f}")
    print(f"📉 Máx Drawdown: {max_dd:.2%}")
    
    # Graphs
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(cumulative_returns, label='Equity Curve', color='blue')
    ax[0].set_ylabel('Capital growth')
    ax[0].legend()
    
    ax[1].plot(drawdown, label='Drawdown', color='red')
    ax[1].axhline(max_dd, color='k', linestyle='--', label=f'Máx Drawdown ({max_dd:.2%})')
    ax[1].set_ylabel('Drawdown')
    ax[1].legend()
    
    plt.suptitle("Strategy analysis")
    plt.tight_layout()
    plt.show()


# In[38]:


strategy_returns = (real_prices / np.roll(real_prices, 1) - 1)[1:]


# In[39]:


analyse_strategy(pd.Series(strategy_returns, index=index_forecast[1:]), rf=0.02)


# In[ ]:




