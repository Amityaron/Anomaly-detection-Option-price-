# Import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

# Black-Scholes Option Pricing Model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# App title and description
st.title("Options Pricing Model App")
st.write("This app displays call and put options with their Option Pricing Model (OPM) values. Filter by OPM to see options within a specific range.")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")
opm_filter = st.sidebar.slider("Filter by OPM", min_value=0.0, max_value=100.0, value=(0.0, 100.0))

# Function to fetch options data with caching
@st.cache_data
def fetch_options_data(ticker):
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options
    return stock, expiration_dates

try:
    stock, expiration_dates = fetch_options_data(ticker)
    if not expiration_dates:
        st.error("No options data available for this ticker.")
    else:
        # Select expiration date
        exp_date = st.sidebar.selectbox("Select expiration date", expiration_dates)

        # Fetch options chain for the selected expiration date
        options_chain = stock.option_chain(exp_date)
        calls = options_chain.calls
        puts = options_chain.puts

        # Get stock price and set risk-free rate
        stock_price = stock.history(period="1d")["Close"].iloc[-1]
        risk_free_rate = 0.01  # Example risk-free rate

        # Time to expiration
        T = (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.now()).days / 365

        # Calculate OPM for each option and add to dataframes
        def calculate_opm(df, option_type):
            df = df.dropna(subset=['impliedVolatility'])
            df["OPM"] = df.apply(
                lambda x: black_scholes(
                    S=stock_price,
                    K=x["strike"],
                    T=T,
                    r=risk_free_rate,
                    sigma=x["impliedVolatility"],
                    option_type=option_type
                ),
                axis=1
            )
            return df

        # Calculate OPM for calls and puts
        calls = calculate_opm(calls, "call")
        puts = calculate_opm(puts, "put")

        # Concatenate calls and puts into a single DataFrame
        options_df = pd.concat([calls.assign(Type="Call"), puts.assign(Type="Put")])

        # Filter based on OPM range
        filtered_options = options_df[(options_df["OPM"] >= opm_filter[0]) & (options_df["OPM"] <= opm_filter[1])]

        # Display the filtered options
        st.write("### Filtered Options Table")
        st.write(filtered_options[["Type", "strike", "lastPrice", "impliedVolatility", "OPM"]])

except Exception as e:
    st.error("Could not retrieve data for the provided ticker symbol. Please check the ticker and try again.")
    st.error(f"Error details: {e}")
