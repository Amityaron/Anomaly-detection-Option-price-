import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go

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

# Plot the data and signals using Plotly
title = f"{symbol} ({start_date} to {end_date})"
fig = go.Figure()

# Plot Close, SMA, Upper and Lower bands
fig.add_trace(go.Scatter(x=spy.index, y=spy['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=spy.index, y=spy['SMA'], mode='lines', name='SMA'))
fig.add_trace(go.Scatter(x=spy.index, y=spy['Upper'], mode='lines', name='Upper Band'))
fig.add_trace(go.Scatter(x=spy.index, y=spy['Lower'], mode='lines', name='Lower Band'))

# Plot Buy signals
fig.add_trace(go.Scatter(x=spy.loc[spy['Signal'] == 1].index, 
                         y=spy.loc[spy['Signal'] == 1]['Close'], 
                         mode='markers', 
                         marker=dict(symbol='circle', size=10, color='green'),
                         name='Buy Signal'))

# Plot Sell signals
fig.add_trace(go.Scatter(x=spy.loc[spy['Sell_Signal'] == 1].index, 
                         y=spy.loc[spy['Sell_Signal'] == 1]['Close'], 
                         mode='markers', 
                         marker=dict(symbol='x', size=10, color='red'),
                         name='Sell Signal'))

# Add a horizontal line at the current closing price
fig.add_trace(go.Scatter(x=[spy.index[0], spy.index[-1]], 
                         y=[spy.iloc[-1]['Close'], spy.iloc[-1]['Close']], 
                         mode='lines', 
                         line=dict(dash='dash', color='red'),
                         name='Current Price'))

# Update layout
fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', 
                  template='plotly_dark', showlegend=True)

# Display the plot
st.plotly_chart(fig)

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

    # Plotting Monthly Returns using Plotly
    fig = go.Figure()

    # Box plot for Monthly Returns
    fig.add_trace(go.Box(y=monthly_returns_df['Monthly Return'], 
                         x=monthly_returns_df['Month Name'], 
                         boxmean='sd', 
                         name='Monthly Return', 
                         marker_color='skyblue'))

    # Plot last year returns
    fig.add_trace(go.Scatter(x=last_year_returns.index, 
                             y=last_year_returns.values, 
                             mode='markers', 
                             marker=dict(symbol='circle', size=12, color='red'),
                             name='Last Year Returns'))

    # Update layout
    fig.update_layout(title=f'Monthly Percentage Changes for {symbol}', 
                      xaxis_title='Month', 
                      yaxis_title='Percentage Change',
                      xaxis_tickangle=-45, 
                      template='plotly_dark', showlegend=True)

    # Display the plot
    st.plotly_chart(fig)

    st.write(f'Monthly Percentage Changes for {symbol}')
    st.write(f'Data from {start_date_sp500} until {end_date_sp500}')
