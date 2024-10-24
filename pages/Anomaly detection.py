# Bulid Streamlit app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import streamlit as st

# Set the app title 
st.title('Anomaly Detection Stock Market App') 

# Add a welcome message 
st.write('Welcome to my Anomaly Detection app!') 

# Create a text input 
widgetuser_input = st.text_input('Enter a ticker based on Yahoo Finance:', 'SPY') 

# Create date inputs for start and end dates
start_date = st.date_input('Start Date', value=pd.to_datetime('2024-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2024-07-09'))

# Download stock data from Yahoo Finance
symbol = widgetuser_input
spy = yf.download(symbol, start=start_date, end=end_date)

# Define Bollinger Band parameters
n = 22  # number of periods for moving average
l = 2   # number of standard deviations for lower band
u = 2   # number of standard deviations for upper band

# Calculate rolling mean and standard deviation
spy['SMA'] = spy['Close'].rolling(n).mean()
spy['STD'] = spy['Close'].rolling(n).std()

# Calculate upper and lower bands
spy['Upper'] = spy['SMA'] + u * spy['STD']
spy['Lower'] = spy['SMA'] - l * spy['STD']

# Drop rows with NaN values from rolling calculations
spy['Close'], spy['Lower'] = spy['Close'].align(spy['Lower'], axis=0)
spy['Close'], spy['Upper'] = spy['Close'].align(spy['Upper'], axis=0)
spy.dropna(subset=['Close', 'Lower', 'Upper'], inplace=True)

# Generate buy signals when the Close price crosses below the lower band
spy['Signal'] = 0
spy.loc[spy['Close'] < spy['Lower'], 'Signal'] = 1

# Remove consecutive signals to only show the first buy signal
spy['Signal'] = spy['Signal'].diff().fillna(0)
spy.loc[spy['Signal'] < 0, 'Signal'] = 0

# Generate sell signals when the Close price crosses above the upper band
spy['Sell_Signal'] = 0
spy.loc[spy['Close'] > spy['Upper'], 'Sell_Signal'] = 1

# Remove consecutive signals to only show the first sell signal
spy['Sell_Signal'] = spy['Sell_Signal'].diff().fillna(0)
spy.loc[spy['Sell_Signal'] < 0, 'Sell_Signal'] = 0

# Calculate percentage changes for buy signals
lower_dates = spy.index[spy['Signal'] == 1].tolist()
buy_prices = spy.loc[spy['Signal'] == 1, 'Close'].tolist()
pct_change = [((spy.iloc[-1]['Close'] / price) - 1) * 100 for price in buy_prices]

# Visualize the data and signals
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

# Create a DataFrame to display buy signals and percentage changes
df = pd.DataFrame({'Buy_Signal_Date': lower_dates, 'Buy Price': buy_prices, 'Gain_Pct': pct_change})
st.write("Current price:", round(spy.iloc[-1]['Close'], 2))
st.table(df.round(2))

########################################################################

st.subheader('Monthly Percentage Changes') 
start_date_sp500 = '1990-01-01'
end_date_sp500 = pd.Timestamp.now()

sp500_data = yf.download(symbol, start=start_date_sp500, end=end_date_sp500)
sp500_data.sort_index(inplace=True)
sp500_data['Daily Return'] = sp500_data['Adj Close'].pct_change()
monthly_returns = round(sp500_data['Adj Close'].resample('M').ffill().pct_change(), 3) * 100
monthly_returns_df = monthly_returns.to_frame(name='Monthly Return')
monthly_returns_df['Year'] = monthly_returns_df.index.year
monthly_returns_df['Month'] = monthly_returns_df.index.month
monthly_returns_df['Month Name'] = monthly_returns_df.index.strftime('%B')
last_year = monthly_returns_df[monthly_returns_df['Year'] == monthly_returns_df['Year'].max()]
last_year_returns = last_year.set_index('Month Name')['Monthly Return']
start_date_sp500 = sp500_data.index[0]
total_months = int(end_date_sp500.year - start_date_sp500.year)
fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(x='Month Name', y='Monthly Return', data=monthly_returns_df, order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], ax=ax)
ax.scatter(last_year_returns.index, last_year_returns.values, color='red', zorder=5, label='Last Year Returns')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title(f'Monthly Percentage Changes {symbol} from {start_date_sp500} until {end_date_sp500} number of observation: {total_months}')
ax.set_xlabel('Month')
ax.set_ylabel('Percentage Change')
ax.grid(True)
ax.axhline(y=0, color='r', linestyle='--')
st.write(f'Monthly Percentage Changes: {symbol}') 
st.write(f'From {start_date_sp500} until {end_date_sp500}')
st.write(f'Number of observation: {total_months}')
st.pyplot(fig)

################################################

# Probability calculations for positive monthly returns
monthly_positive_counts = {month: 0 for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']}
for date, return_value in monthly_returns.items():
    if not pd.isna(return_value) and return_value > 0:
        month_name = date.strftime('%B')
        monthly_positive_counts[month_name] += 1

start_date = sp500_data.index[0]
end_date_dt = pd.Timestamp.now()

total_months = int(end_date_dt.year - start_date.year)

probabilities = {month: (count / total_months) * 100 for month, count in monthly_positive_counts.items()}
probabilities_df = pd.DataFrame(list(probabilities.items()), columns=['Month', 'Probability (%)'])

st.write("The probability to get positive return for each month")
st.write("You want to buy in the months with the lowest probability of getting a positive return")
st.table(probabilities_df)
