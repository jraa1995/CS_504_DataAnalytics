import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path, dtype={'Local Case Number': str})

def prepare_time_data(df):
    df['Crash Date'] = pd.to_datetime(df['Crash Date'])
    df['Year'] = df['Crash Date'].dt.year
    df['Month'] = df['Crash Date'].dt.month_name()
    df['DayOfWeek'] = df['Crash Date'].dt.day_name()
    df['Hour'] = pd.to_datetime(df['Crash Time'], format='%H:%M').dt.hour
    return df

def plot_trends(df):
    sns.set(style="whitegrid")

    # Yearly trend
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Year', data=df, palette='viridis')
    plt.title('Yearly Crash Trend')
    plt.ylabel('Number of Crashes')
    plt.xticks(rotation=45)
    plt.show()

    # Monthly trend
    plt.figure(figsize=(10, 6))
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    sns.countplot(x='Month', data=df, order=month_order, palette='coolwarm')
    plt.title('Monthly Crash Trend')
    plt.ylabel('Number of Crashes')
    plt.xticks(rotation=45)
    plt.show()

    # Day of the week trend
    plt.figure(figsize=(10, 6))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.countplot(x='DayOfWeek', data=df, order=day_order, palette='muted')
    plt.title('Day of the Week Crash Trend')
    plt.ylabel('Number of Crashes')
    plt.xticks(rotation=45)
    plt.show()

    # Time of the day trend
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Hour', data=df, palette='rocket')
    plt.title('Hourly Crash Trend')
    plt.ylabel('Number of Crashes')
    plt.show()

def plot_dual_axis_crashes_time(df):
    sns.set(style="whitegrid")

    # Count of crashes by hour
    hourly_crash_count = df.groupby('Hour').size()

    # Cumulative crashes throughout the day
    cumulative_crashes = hourly_crash_count.cumsum()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting the crash count
    color = 'tab:blue'
    ax1.set_xlabel('Hour of the Day')
    ax1.set_ylabel('Number of Crashes (Hourly)', color=color)
    ax1.bar(hourly_crash_count.index, hourly_crash_count, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Creating a second y-axis for cumulative crashes
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Crashes', color=color)
    ax2.plot(cumulative_crashes.index, cumulative_crashes, color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Hourly Crash Count and Cumulative Crashes Throughout the Day')
    plt.show()

def perform_trend_analysis(file_path):
    df = load_data(file_path)
    df = prepare_time_data(df)
    plot_trends(df)
    plot_dual_axis_crashes_time(df)


if __name__ == "__main__":
    file_path = 'crashreport.csv'
    perform_trend_analysis(file_path)
