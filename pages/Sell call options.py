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

# Function to calculate probability of reaching OPM
def probability_opm(S, K, T, sigma, option_type="call"):
    d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)  # Probability of reaching/exceeding the OPM price for calls
    elif option_type == "put":
        return 1 - norm.cdf(d1)  # Probability of reaching/exceeding the OPM price for puts

# App title and description
st.title("Options Pricing Model App with Probability of Reaching OPM")
st.write("This app displays call and put options with their Option Pricing Model (OPM) values and the probability of reaching OPM by expiration. Filter by the probability of OPM.")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")
opm_prob_filter = st.sidebar.slider("Filter by Probability of Reaching OPM", min_value=0.0, max_value=1.0, value=(0.0, 1.0))

# Function to fetch options data (expiration dates only, to be cacheable)
@st.cache_data
def fetch_expiration_dates(ticker):
    stock = yf.Ticker(ticker)
    return stock.options

# Get stock and expiration dates
try:
    expiration_dates = fetch_expiration_dates(ticker)
    if not expiration_dates:
        st.error("No options data available for this ticker.")
    else:
        # Select expiration date
        exp_date = st.sidebar.selectbox("Select expiration date", expiration_dates)

        # Fetch options chain for the selected expiration date
        stock = yf.Ticker(ticker)  # Get the non-cached stock object to fetch options chain
        options_chain = stock.option_chain(exp_date)
        calls = options_chain.calls
        puts = options_chain.puts

        # Get stock price and set risk-free rate
        stock_price = stock.history(period="1d")["Close"].iloc[-1]
        risk_free_rate = 0.01  # Example risk-free rate

        # Time to expiration
        T = (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.now()).days / 365

        # Calculate OPM and probability of reaching OPM for each option
        def calculate_opm_and_opm_prob(df, option_type):
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
            df["P(OPM)"] = df.apply(
                lambda x: probability_opm(
                    S=stock_price,
                    K=x["strike"],
                    T=T,
                    sigma=x["impliedVolatility"],
                    option_type=option_type
                ),
                axis=1
            )
            return df

        # Calculate OPM and P(OPM) for calls and puts
        calls = calculate_opm_and_opm_prob(calls, "call")
        puts = calculate_opm_and_opm_prob(puts, "put")

        # Concatenate calls and puts into a single DataFrame
        options_df = pd.concat([calls.assign(Type="Call"), puts.assign(Type="Put")])

        # Filter based on Probability of Reaching OPM range
        filtered_options = options_df[(options_df["P(OPM)"] >= opm_prob_filter[0]) & (options_df["P(OPM)"] <= opm_prob_filter[1])]

        # Display the filtered options
        st.write("### Filtered Options Table")
        st.write(filtered_options[["Type", "strike", "lastPrice", "impliedVolatility", "OPM", "P(OPM)"]])

except Exception as e:
    st.error("Could not retrieve data for the provided ticker symbol. Please check the ticker and try again.")
    st.error(f"Error details: {e}")
