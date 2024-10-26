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
st.title('Sell call options strategy') 
# Add a welcome message 
st.write('Sell call options strategy') 
st.markdown(
    """
           - Sell out of the money call options of your securities with the probability you like!
           - Based on Black-Scholes Model.
           
           \
           \
           
             $C = S N(d_1) - K e^{-rt} N(d_2)$
             
             \
             \

             where
             
             \
             \
             
             $d_1 = \\frac{\\ln\\left(\\frac{S}{K}\\right) + \\left(r + \\frac{\\sigma^2}{2}\\right)t}{\\sigma \\sqrt{t}}$
             
             \
             \
             
             $d_2 = d_1 - \sigma \sqrt{t}$
             
             \
             \
             
             - $C$: Call option price
             - $S$: Current stock price
             - $K$: Strike price of the option
             - $r$: Risk-free interest rate
             - $t$: Time to expiration (in years)
             - $\sigma$: Volatility of the stock
             - $N(d)$: Cumulative distribution function of the standard normal distribution
    """
)

#st.image("G:/My Drive/Colab Notebooks/Streamlit_app/pages/blackscholes.png")
# Create a text input 
widgetuser_input = st.text_input('Enter a ticker  based on yahoo finance:', 'SOXX') 
days_for_volatility = st.number_input('Enter a Numbers of days to estimate volatility  ', 60) 
# 
#Add slider bar for probebilty ! 
#
#
# Add slider bar for probability
probability_threshold = st.slider('Select the minimum probability of expiring OTM', 0.0, 1.0, 0.9)



import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

def get_options_table(ticker_symbol, expiration_date):
    # Get the ticker object
    ticker = yf.Ticker(ticker_symbol)

    # Get the current stock price
    st.write(ticker.history(period='1d')['Close'])
    stock_price =float( ticker.history(period='1d')['Close'][0] )
    
    # Get the options chain for the chosen expiration date
    options_chain = ticker.option_chain(expiration_date)

    # Get the calls and puts data
    calls = options_chain.calls
    puts = options_chain.puts

    # Add expiration date to each DataFrame
    calls['expirationDate'] = expiration_date
    puts['expirationDate'] = expiration_date

    # Combine calls and puts into a single DataFrame
    calls['type'] = 'call'
    puts['type'] = 'put'
    options_table = pd.concat([calls, puts])

    return options_table, stock_price

def calculate_probability_otm(call_options, stock_price, risk_free_rate, days_to_expiration, volatility):
    # Calculate d2 from the Black-Scholes formula
    call_options['d2'] = (np.log(stock_price / call_options['strike']) + (risk_free_rate - 0.5 * volatility ** 2) * days_to_expiration / 365) / (volatility * np.sqrt(days_to_expiration / 365))

    # Calculate the probability of expiring OTM for call options
    call_options['probability_otm'] = norm.cdf(-call_options['d2'])

    return call_options

def filter_options_table(options_table, stock_price, risk_free_rate, volatility):
    # Filter for call options
    call_options = options_table[options_table['type'] == 'call']

    # Calculate the days to expiration
    call_options['daysToExpiration'] = (pd.to_datetime(call_options['expirationDate']) - pd.to_datetime('today')).dt.days

    # Calculate the probability of expiring OTM
    call_options = calculate_probability_otm(call_options, stock_price, risk_free_rate, call_options['daysToExpiration'], volatility)

    # Filter for call options with probability of expiring OTM of 90% or higher
    filtered_call_options = call_options[call_options['probability_otm'] >= probability_threshold]

    # Select relevant columns for display
    filtered_call_options = filtered_call_options[['contractSymbol', 'strike', 'lastPrice', 'probability_otm']]

    return filtered_call_options

def estimate_volatility(ticker_symbol, days):
    # Get the ticker object
    ticker = yf.Ticker(ticker_symbol)

    # Calculate the start and end dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Fetch historical data
    hist = ticker.history(start=start_date, end=end_date)

    # Ensure we have enough data
    #if len(hist) < days:
    #    raise ValueError("Not enough historical data to estimate volatility")

    # Calculate daily returns
    hist['Return'] = hist['Close'].pct_change()
    
    # Drop NaN values
    hist = hist.dropna()

    # Calculate the standard deviation of daily returns
    volatility = hist['Return'].std() * np.sqrt(252)  # Annualize the volatility

    return volatility

# Specify the ticker symbol
ticker_symbol = widgetuser_input  # Example: Apple Inc.

# Specify the risk-free rate
risk_free_rate = 0.04  # 4% annual risk-free rate

# Estimate volatility using historical data
volatility = estimate_volatility(ticker_symbol, days_for_volatility)

# Get the ticker object
ticker = yf.Ticker(ticker_symbol)

# Get the available options expiration dates
expiration_dates = ticker.options

# Initialize an empty DataFrame to store results
all_filtered_options = pd.DataFrame()

# Loop through the expiration dates
for expiration_date in expiration_dates[:4]:  # Adjust the range as needed
    # Get the options table for the given ticker and expiration date
    options_table, stock_price = get_options_table(ticker_symbol, expiration_date)

    # Filter the options table
    filtered_call_options = filter_options_table(options_table, stock_price, risk_free_rate, volatility)

    # Add expiration date to the DataFrame
    filtered_call_options['expiration_date'] = expiration_date

    # Append to the final DataFrame
    all_filtered_options = pd.concat([all_filtered_options, filtered_call_options])

# Display the filtered call options
#print(all_filtered_options)

# Optionally, save the filtered call options to a CSV file
#all_filtered_options.to_csv(f'{ticker_symbol}_filtered_call_options.csv', index=False)




st.table(all_filtered_options.sort_values(by='expiration_date', ascending=True))
