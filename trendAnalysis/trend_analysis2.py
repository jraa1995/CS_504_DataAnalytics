import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def perform_advanced_time_series_analysis(file_path):
    # Load the dataset
    crash_data = pd.read_csv(file_path, low_memory=False)

    # Convert 'Crash Date' to datetime and set it as index
    crash_data['Crash Date'] = pd.to_datetime(crash_data['Crash Date'])
    crash_data.set_index('Crash Date', inplace=True)

    # Resampling the data by month
    monthly_crash_counts = crash_data.resample('M').size()

    monthly_data = crash_data.resample('M').agg({'Crash Count': 'size', 'Speed Limit': 'mean'})

    # Time Series Decomposition
    decomposition = seasonal_decompose(monthly_crash_counts, model='additive')
    decomposition.plot()
    plt.title('Time Series Decomposition')
    plt.show()

    # Moving Average Analysis
    plt.figure(figsize=(12, 6))
    monthly_crash_counts.plot(label='Original')
    monthly_crash_counts.rolling(window=12).mean().plot(label='12-Month Rolling Mean')
    plt.title('Moving Average Analysis')
    plt.legend()
    plt.show()

    # Correlation Analysis
    plt.figure(figsize=(10, 6))
    sns.heatmap(monthly_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Analysis')
    plt.show()

if __name__ == "__main__":
    file_path = 'crashreport.csv'
    perform_advanced_time_series_analysis(file_path)
