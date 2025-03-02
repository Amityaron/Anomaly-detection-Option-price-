import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import mplcursors

# Set the app title 
st.title('Anomaly Detection Stock Market App')
st.write('Welcome to my Anomaly detection app!')

# Create a text input 
widgetuser_input = st.text_input('Enter a ticker based on Yahoo Finance:', 'SPY')

# Create date inputs for start and end dates
start_date = st.date_input('Start Date', value=pd.to_datetime('2024-01-01'))
end_date = st.date_input('End Date', value=pd.Timestamp.now())

# Download SPY data from Yahoo Finance
symbol = widgetuser_input
spy = yf.download(symbol, start=start_date, end=end_date)

# Verify and flatten multi-level indexes
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

# Define Bollinger Band parameters
n = 22  # number of periods for moving average
l = 2  # number of standard deviations for lower bands
u = 2  # number of standard deviations for upper bands

if 'Close' in spy.columns:
    # Calculate rolling mean and standard deviation
    spy['SMA'] = spy['Close'].rolling(n).mean()
    spy['STD'] = spy['Close'].rolling(n).std()
    
    # Calculate upper and lower bands
    spy['Upper'] = spy['SMA'] + u * spy['STD']
    spy['Lower'] = spy['SMA'] - l * spy['STD']
    
    # Align and remove NaN values
    spy['Close'], spy['Lower'] = spy['Close'].align(spy['Lower'], axis=0)
    spy['Close'], spy['Upper'] = spy['Close'].align(spy['Upper'], axis=0)
    spy.dropna(subset=['Close', 'Lower', 'Upper'], inplace=True)

    # Generate buy and sell signals
    spy['Signal'] = 0
    spy.loc[spy['Close'] < spy['Lower'], 'Signal'] = 1
    spy['Signal'] = spy['Signal'].diff().fillna(0)
    spy.loc[spy['Signal'] < 0, 'Signal'] = 0

    spy['Sell_Signal'] = 0
    spy.loc[spy['Close'] > spy['Upper'], 'Sell_Signal'] = 1
    spy['Sell_Signal'] = spy['Sell_Signal'].diff().fillna(0)
    spy.loc[spy['Sell_Signal'] < 0, 'Sell_Signal'] = 0

# Calculate percentage change for buy signals
lower_dates, buy_prices = [], []
for index, row in spy.loc[spy['Signal'] == 1].iterrows():
    buy_prices.append(row['Close'])
    lower_dates.append(index)

pct_change = [((spy.iloc[-1]['Close'] / x) - 1) * 100 for x in buy_prices]

# Plot the data and signals using Matplotlib
title = f"{symbol} ({start_date} to {end_date})"
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(spy['Close'], label='Close')
ax.plot(spy['SMA'], label='SMA')
ax.plot(spy['Upper'], label='Upper Band')
ax.plot(spy['Lower'], label='Lower Band')
ax.plot(spy.loc[spy['Signal'] == 1, 'Close'], 'o', markersize=10, label='Buy Signal')
ax.plot(spy.loc[spy['Sell_Signal'] == 1, 'Close'], 'x', markersize=10, label='Sell Signal')
plt.axhline(y=spy.iloc[-1]['Close'], color='r', linestyle='--')
ax.legend()

# Enable interactivity with mplcursors
mplcursors.cursor(hover=True)

# Add title and labels
plt.title(title)
ax.set_xlabel('Date')
ax.set_ylabel('Price')

# Display the interactive plot
st.pyplot(fig)

# Display Buy Signal Dates, Buy Prices, and Gain Percentages
df = pd.DataFrame({'Buy_Signal_Date': lower_dates, 'Buy Price': buy_prices, 'Gain_Pct': pct_change})
st.write("Current price:", round(spy.iloc[-1]['Close'], 2))
st.table(df.round(2))

# Monthly Percentage Changes
st.subheader('Monthly Percentage Changes')

# Start and end dates for downloading historical data
start_date_sp500 = '1990-01-01'
end_date_sp500 = pd.Timestamp.now()

# Download the data for the specified ticker symbol
symbol = st.text_input("Enter a ticker symbol:", "SPY")
sp500_data = yf.download(symbol, start=start_date_sp500, end=end_date_sp500)

# Ensure the data is sorted by date
sp500_data.sort_index(inplace=True)

# Calculate daily and monthly returns
sp500_data['Daily Return'] = sp500_data['Close'].pct_change()
monthly_returns = sp500_data['Close'].resample('M').ffill().pct_change() * 100

# Verify and convert `monthly_returns` to a Series if needed
if isinstance(monthly_returns, pd.DataFrame):
    monthly_returns = monthly_returns.squeeze()  # Converts single-column DataFrame to Series

# Check if monthly_returns is empty or invalid
if monthly_returns.empty:
    st.write("Error: No monthly returns data available to display.")
else:
    # Creating DataFrame for monthly returns and other related columns
    monthly_returns_df = monthly_returns.to_frame(name='Monthly Return')
    monthly_returns_df['Year'] = monthly_returns_df.index.year
    monthly_returns_df['Month'] = monthly_returns_df.index.month
    monthly_returns_df['Month Name'] = monthly_returns_df.index.strftime('%B')

    # Filter data for the latest year
    last_year = monthly_returns_df[monthly_returns_df['Year'] == monthly_returns_df['Year'].max()]
    last_year_returns = last_year.set_index('Month Name')['Monthly Return']

    # Plotting Monthly Returns using Matplotlib
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x='Month Name', y='Monthly Return', data=monthly_returns_df,
                order=['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December'], ax=ax)
    ax.scatter(last_year_returns.index, last_year_returns.values, color='red', zorder=5, label='Last Year Returns')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f'Monthly Percentage Changes for {symbol}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Percentage Change')
    ax.grid(True)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.legend()

    # Enable interactivity with mplcursors
    mplcursors.cursor(hover=True)

    # Display the plot and additional information
    st.pyplot(fig)
    st.write(f'Monthly Percentage Changes for {symbol}')
    st.write(f'Data from {start_date_sp500} until {end_date_sp500}')
