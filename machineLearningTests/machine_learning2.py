import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox  # Using acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product
import numpy as np

# Load and preprocess the dataset
crash_df = pd.read_csv('crashreport.csv', low_memory=False)
crash_df['Crash Date'] = pd.to_datetime(crash_df['Crash Date'])
crash_df.set_index('Crash Date', inplace=True)
crash_df['Accident Count'] = 1
monthly_accidents = crash_df['Accident Count'].resample('M').sum()

# Including data up to November 1, 2023
monthly_accidents = monthly_accidents[:'2023-11-01']

# Grid Search for SARIMA parameters
p = d = q = range(0, 2)  # Example ranges, can be adjusted
pdq = list(product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(product(p, d, q))]

best_aic = float("inf")
best_pdq = None
best_seasonal_pdq = None
best_model = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(monthly_accidents,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_model = results
        except:
            continue

print('Best SARIMA{}x{} - AIC:{}'.format(best_pdq, best_seasonal_pdq, best_aic))

# Forecasting
forecast_steps = 12  # Number of steps to forecast
forecast = best_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start='2023-11-02', periods=forecast_steps, freq='M')

# Residual Analysis
residuals = best_model.resid
plt.figure(figsize=(12,6))
plt.plot(residuals)
plt.title('Residuals of SARIMA Model')
plt.show()

# Residual ACF and Ljung-Box Test
plt.figure(figsize=(12,6))
pd.plotting.autocorrelation_plot(residuals)
plt.show()

ljung_box = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(ljung_box)

# Performance Metrics
mae = mean_absolute_error(monthly_accidents[-len(residuals):], residuals + monthly_accidents[-len(residuals):])
mse = mean_squared_error(monthly_accidents[-len(residuals):], residuals + monthly_accidents[-len(residuals):])
rmse = np.sqrt(mse)

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)

# Plotting Forecast
plt.figure(figsize=(12,6))
plt.plot(monthly_accidents.index, monthly_accidents, label='Historical Data')
plt.plot(forecast_index, forecast, label='Forecast')
plt.title('SARIMA Forecast from Nov 02, 2023 Onwards')
plt.xlabel('Month')
plt.ylabel('Accident Count')
plt.legend()
plt.show()

# Output the forecasted values
forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted Accidents': forecast})
forecast_df.set_index('Date', inplace=True)
print(forecast_df)
