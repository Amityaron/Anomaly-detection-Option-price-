# Build Streamlit app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import streamlit as st

# Set the app title 
st.title('Anomaly Detection Stock Market App')
st.write('Welcome to my Anomaly detection app!')

# Create a text input 
widgetuser_input = st.text_input('Enter a ticker based on Yahoo Finance:', 'SPY')

# Create date inputs for start and end dates
start_date = st.date_input('Start Date', value=pd.to_datetime('2024-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2024-10-24'))

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

# Plot the data and signals
title = f"{symbol} ({start_date} to {end_date})"
fig, ax = plt.subplots(figsize=(12, 6))
plt.title(title)
ax.plot(spy['Close'], label='Close')
ax.plot(spy['SMA'], label='SMA')
ax.plot(spy['Upper'], label='Upper Band')
ax.plot(spy['Lower'], label='Lower Band')
ax.plot(spy.loc[spy['Signal'] == 1, 'Close'], 'o', markersize=10, label='Buy Signal')
ax.plot(spy.loc[spy['Sell_Signal'] == 1, 'Close'], 'x', markersize=10, label='Sell Signal')
plt.axhline(y=spy.iloc[-1]['Close'], color='r', linestyle='--')
ax.legend()
st.pyplot(fig)

# Display Buy Signal Dates, Buy Prices, and Gain Percentages
df = pd.DataFrame({'Buy_Signal_Date': lower_dates, 'Buy Price': buy_prices, 'Gain_Pct': pct_change})
st.write("Current price:", round(spy.iloc[-1]['Close'], 2))
st.table(df.round(2))

# Monthly Percentage Changes
st.subheader('Monthly Percentage Changes')
sp500_data = yf.download(symbol, start='1990-01-01', end=pd.Timestamp.now())
sp500_data.sort_index(inplace=True)

sp500_data['Daily Return'] = sp500_data['Adj Close'].pct_change()
monthly_returns = round(sp500_data['Adj Close'].resample('M').ffill().pct_change(), 3) * 100
monthly_returns_df = monthly_returns.to_frame(name='Monthly Return')
monthly_returns_df['Year'] = monthly_returns_df.index.year
monthly_returns_df['Month'] = monthly_returns_df.index.month
monthly_returns_df['Month Name'] = monthly_returns_df.index.strftime('%B')

last_year = monthly_returns_df[monthly_returns_df['Year'] == monthly_returns_df['Year'].max()]
last_year_returns = last_year.set_index('Month Name')['Monthly Return']
fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(x='Month Name', y='Monthly Return', data=monthly_returns_df,
            order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], ax=ax)
ax.scatter(last_year_returns.index, last_year_returns.values, color='red', zorder=5, label='Last Year Returns')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.axhline(y=0, color='r', linestyle='--')
st.pyplot(fig)

# Calculate Probabilities of Positive Monthly Returns
monthly_positive_counts = {month: 0 for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']}
for date, return_value in monthly_returns.items():
    if not pd.isna(return_value) and return_value > 0:
        month_name = date.strftime('%B')
        monthly_positive_counts[month_name] += 1

total_months = pd.Timestamp.now().year - sp500_data.index[0].year
probabilities = {month: f"{(positive_count / total_months) * 100:.1f}%" for month, positive_count in monthly_positive_counts.items()}

# Display the probabilities table
probabilities_df = pd.DataFrame(list(probabilities.items()), columns=['Month', 'Probability (%)'])
st.write("Probability of Positive Monthly Returns:")
st.table(probabilities_df)
