# Import necessary libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import skew, kurtosis
import streamlit as st

# Set the app title 
st.title('Welcome to my Anomaly Detection App')
st.title('Based on Z-Score') 


# Skewness Explanation
st.markdown("""
    ### Skewness Definitions:
    - Highly positively skewed: Skewness > 1
    - Moderately positively skewed: 0.5 < Skewness ≤ 1
    - Approximately symmetric: -0.5 ≤ Skewness ≤ 0.5
    - Moderately negatively skewed: -1 ≤ Skewness < -0.5
    - Highly negatively skewed: Skewness < -1
""")

# Default ETF tickers
default_etfs = ["QQQ","SPY","XLK","SOXX","XLF"]


# Sidebar for adding new tickers
st.sidebar.header("Add or Remove Tickers")

# Add new ticker input
new_ticker = st.sidebar.text_input("Enter a new ETF ticker (e.g., AAPL)")

# Add ticker button
if st.sidebar.button("Add Ticker"):
    if new_ticker.upper() not in default_etfs:
        default_etfs.append(new_ticker.upper())  # Add ticker to the list
        st.sidebar.success(f"Added {new_ticker.upper()} to the list.")
    else:
        st.sidebar.warning(f"{new_ticker.upper()} is already in the list.")

# Allow the user to remove tickers from the list
tickers_to_remove = st.sidebar.multiselect("Select tickers to remove", default_etfs)

# Remove selected tickers
if st.sidebar.button("Remove Selected Tickers"):
    for ticker in tickers_to_remove:
        default_etfs.remove(ticker)
    st.sidebar.success(f"Removed selected tickers: {', '.join(tickers_to_remove)}")

# Re-fetch data and re-run calculations for the updated ticker list
etfs = default_etfs  # Updated list of tickers
z_score_list = []
current_price_list = []
skewness_list = []
kurtosis_list = []

# Loop through each ETF ticker and calculate Z-score, skewness, kurtosis
for etf in etfs:
    # Get historical data for the last month
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(days=22)  # Last month's data
    data = yf.download(etf, start=start_date, end=end_date)["Adj Close"]

    # Calculate mean and standard deviation for the last month
    mean_last_month = data.mean()
    std_last_month = data.std()
    # Calculate skewness and kurtosis for the last month
    skewness_last_month =  skew(data) 
    kurtosis_last_month = kurtosis(data) 
    skewness_list.append(skewness_last_month)
    kurtosis_list.append(kurtosis_last_month)

    # Get the most recent data point (current price)
    current_price = data.iloc[-1]
    current_price_list.append(round(current_price, 2))

    # Calculate Z score for the current price
    z_score_current_price = round((current_price - mean_last_month) / std_last_month, 2)
    z_score_list.append(z_score_current_price)

# Create DataFrame to display the results
df = pd.DataFrame({
    "ETF Symbol": etfs,
    "Current Price": current_price_list,
    "Z Score": z_score_list,
    "Skewness": skewness_list,
    "Kurtosis": kurtosis_list
})

# Display the table sorted by Z-Score
st.table(df.sort_values(by='Z Score', ascending=True))
