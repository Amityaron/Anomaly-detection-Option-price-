# Bulid Streamlit app
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas_datareader
import yfinance as yf
from pandas_datareader import data as pdr
from numpy.core.fromnumeric import sort
import datetime
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st




    
# Set the app title 
st.title('Anomaly detection stock market app') 
# Add a welcome message 
st.write('Welcome to my Anomaly detection app!') 


# Create a text input 
widgetuser_input = st.text_input('Enter a ticker  based on yahoo finance:', 'SPY') 
# Display the customized message 

# Create date inputs for start and end dates
start_date = st.date_input('Start Date', value=pd.to_datetime('2023-01-01'))
end_date = st.date_input('End Date', value=pd.Timestamp("today"))

# Download SPY data from Yahoo Finance
symbol=widgetuser_input
spy = yf.download(symbol, start=start_date, end=end_date)

# Define Bollinger Band parameters
n = 30 # number of periods for moving average
l = 2 # number of standard deviations for  lower bands
u=2 # number of standard deviations for upper bands


# Calculate rolling mean and standard deviation
spy['SMA'] = spy['Close'].rolling(n).mean()
spy['STD'] = spy['Close'].rolling(n).std()

# Calculate upper and lower bands
spy['Upper'] = spy['SMA'] + u * spy['STD']
spy['Lower'] = spy['SMA'] - l * spy['STD']

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



# Calculate the total return on investment

# Create a list to store dates where the previous month's average was lower than the stock price
lower_dates = []
buy_prices = []

# Loop through each buy signal and store the price
for index, row in spy.loc[spy['Signal'] == 1].iterrows():
    buy_price = row['Close']
    buy_prices.append(buy_price)
    lower_dates.append(index)

pct_change =  [((spy.iloc[-1]['Close'] / x) - 1) * 100 for x in buy_prices] 





# Visualize the data and signals
title = f"{symbol} ({start_date} to {end_date})"
fig, ax = plt.subplots(figsize=(12,6))
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


df = pd.DataFrame({'Buy_Signal_Date': lower_dates, ' Buy price':buy_prices,'Gain_Pct': pct_change})


st.write("Total return in % :",round(sum(pct_change),2)) 
st.write("Current price:",round( spy.iloc[-1]['Close'],2 ))
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

from datetime import datetime
import streamlit as st

# Define a single ticker symbol
ticker = symbol
start_date = '1990-01-01'
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

# Step 1: Fetch historical data
sp500_data = yf.download(ticker, start=start_date, end=end_date)

# Ensure the data is sorted by date
sp500_data.sort_index(inplace=True)

# Step 2: Calculate the daily percentage changes
sp500_data['Daily Return'] = sp500_data['Adj Close'].pct_change()

# Step 3: Calculate the monthly percentage change (from the 1st to the last day of each month)
monthly_returns = sp500_data['Adj Close'].resample('M').ffill().pct_change()

# Step 4: Initialize a dictionary to store counts of positive returns for each month
monthly_positive_counts = {
    'January': 0,
    'February': 0,
    'March': 0,
    'April': 0,
    'May': 0,
    'June': 0,
    'July': 0,
    'August': 0,
    'September': 0,
    'October': 0,
    'November': 0,
    'December': 0
}

# Step 5: Iterate over monthly returns to count positive returns for each month
for date, return_value in monthly_returns.items():
    if not pd.isna(return_value) and return_value > 0:
        month_name = date.strftime('%B')
        monthly_positive_counts[month_name] += 1

start_date = sp500_data.index[0]
end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
#start_date_dt=datetime.strptime( sp500_data.index[0], '%Y-%m-%d')

# Step 6: Calculate total number of months
total_months = int( end_date_dt.year - start_date.year)

# Step 7: Calculate probabilities for each month
probabilities = {}
for month, positive_count in monthly_positive_counts.items():
    probability = (positive_count / total_months) * 100
    probabilities[month] =  f"{probability:.1f}%"

# Convert the probabilities dictionary into a DataFrame
probabilities_df = pd.DataFrame(list(probabilities.items()), columns=['Month', 'Probability (%)'])


# Display the results in Streamlit
st.write(" The probabilty to get postive return for each mouth ")
st.write(" You want to buy in the months with the lowest probability of getting a positive return ")
st.table(probabilities_df)
# Set the app title 
