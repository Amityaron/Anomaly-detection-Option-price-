import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import yfinance as yf
import streamlit as st

# Set the app title 
st.title('Sell Call Options Strategy') 

# Add a welcome message 
st.write('Sell call options strategy based on the Black-Scholes Model.')

# Create a text input 
widgetuser_input = st.text_input('Enter a ticker based on Yahoo Finance:', 'SPY') 
days_for_volatility = st.number_input('Enter the number of days to estimate volatility:', 60) 
probability_threshold = st.slider('Select the minimum probability of expiring OTM:', 0.0, 1.0, 0.9)

def get_stock_price(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period='1d')
    if not history.empty:
        return float(history['Close'].iloc[0])
    else:
        st.error("No historical data available for the specified ticker.")
        return None

def get_options_table(ticker_symbol, expiration_date):
    ticker = yf.Ticker(ticker_symbol)
    options_chain = ticker.option_chain(expiration_date)
    
    calls = options_chain.calls
    puts = options_chain.puts

    calls['expirationDate'] = expiration_date
    puts['expirationDate'] = expiration_date

    calls['type'] = 'call'
    puts['type'] = 'put'
    options_table = pd.concat([calls, puts])

    return options_table

def calculate_probability_otm(call_options, stock_price, risk_free_rate, days_to_expiration, volatility):
    call_options['d2'] = (np.log(stock_price / call_options['strike']) + 
                          (risk_free_rate - 0.5 * volatility ** 2) * days_to_expiration / 365) / (volatility * np.sqrt(days_to_expiration / 365))
    call_options['probability_otm'] = norm.cdf(-call_options['d2'])
    return call_options

def filter_options_table(options_table, stock_price, risk_free_rate, volatility):
    call_options = options_table[options_table['type'] == 'call']
    call_options['daysToExpiration'] = (pd.to_datetime(call_options['expirationDate']) - pd.to_datetime('today')).dt.days
    call_options = calculate_probability_otm(call_options, stock_price, risk_free_rate, call_options['daysToExpiration'], volatility)
    return call_options[call_options['probability_otm'] >= probability_threshold]

def estimate_volatility(ticker_symbol, days):
    ticker = yf.Ticker(ticker_symbol)
    start_date = datetime.now() - timedelta(days=days)
    hist = ticker.history(start=start_date)
    if not hist.empty:
        hist['Return'] = hist['Close'].pct_change().dropna()
        return hist['Return'].std() * np.sqrt(252)
    else:
        st.error("Not enough historical data to estimate volatility.")
        return 0.0

# Main app logic
ticker_symbol = widgetuser_input
risk_free_rate = 0.04  # 4% annual risk-free rate

volatility = estimate_volatility(ticker_symbol, days_for_volatility)
stock_price = get_stock_price(ticker_symbol)

if stock_price is not None:
    expiration_dates = yf.Ticker(ticker_symbol).options
    all_filtered_options = pd.DataFrame()

    for expiration_date in expiration_dates[:4]:
        options_table = get_options_table(ticker_symbol, expiration_date)
        filtered_call_options = filter_options_table(options_table, stock_price, risk_free_rate, volatility)
        filtered_call_options['expiration_date'] = expiration_date
        all_filtered_options = pd.concat([all_filtered_options, filtered_call_options])

    st.table(all_filtered_options.sort_values(by='expiration_date', ascending=True))
