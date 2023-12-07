import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess the dataset
crash_df = pd.read_csv('crashreport.csv', low_memory=False)
crash_df['Crash Date'] = pd.to_datetime(crash_df['Crash Date'])
crash_df.set_index('Crash Date', inplace=True)

crash_df['Accident Count'] = 1
monthly_accidents = crash_df['Accident Count'].resample('M').sum()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(monthly_accidents.values.reshape(-1,1))

# Function to create dataset for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Split the data into training and test sets
time_step = 100
X, y = create_dataset(scaled_data, time_step)
train_size = int(len(X) * 0.67)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=5)  # Increased epochs

# Predicting and inverse transforming the results
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get back to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Evaluate the model
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f'Train MSE: {train_mse}, Test MSE: {test_mse}')

# Plotting the results
total_data_len = len(monthly_accidents)
train_predict_len = len(train_predict)
test_predict_len = len(test_predict)

train_predict_plot = np.empty((total_data_len, 1))
train_predict_plot[:, :] = np.nan
test_predict_plot = np.empty((total_data_len, 1))
test_predict_plot[:, :] = np.nan

train_predict_plot[time_step:time_step + train_predict_len, :] = train_predict
test_predict_plot[time_step + train_predict_len + 1:time_step + train_predict_len + 1 + test_predict_len, :] = test_predict

plt.figure(figsize=(15,6))
plt.plot(monthly_accidents.values, label='Original Data')
plt.plot(train_predict_plot, label='Train Predict')
plt.plot(test_predict_plot, label='Test Predict')
plt.legend()
plt.show()


# Convert 'Crash Date' to datetime and set it as index
crash_df['Crash Date'] = pd.to_datetime(crash_df['Crash Date'])
crash_df.set_index('Crash Date', inplace=True)

# Summarizing the accident count per month
monthly_accidents = crash_df['Accident Count'].resample('M').sum()

# Plotting the time series
plt.figure(figsize=(12,6))
plt.plot(monthly_accidents)
plt.title('Monthly Accident Count')
plt.xlabel('Month')
plt.ylabel('Accident Count')
plt.show()

# Seasonal Decomposition
result = seasonal_decompose(monthly_accidents, model='additive')
result.plot()
plt.show()

# ARIMA Model
# The order (p,d,q) needs to be determined, here it's set to (1,1,1)
model = ARIMA(monthly_accidents, order=(1,1,1))
fitted_model = model.fit()

# Forecasting the next 12 months
forecast = fitted_model.forecast(steps=12)
plt.figure(figsize=(12,6))
plt.plot(monthly_accidents.index[-24:], monthly_accidents[-24:], label='Historical')
plt.plot(pd.date_range(monthly_accidents.index[-1], periods=12, freq='M'), forecast, label='Forecast')
plt.title('ARIMA Forecast')
plt.xlabel('Month')
plt.ylabel('Accident Count')
plt.legend()
plt.show()
