# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis
import streamlit as st

# Set the app title 
st.title('Welcome to my Anomaly Detection App')
st.subheader('Based on Z-Score')

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
default_etfs = ["QQQ", "SPY", "XLK", "SOXX", "XLF","BTC-USD","URTH","IXN"]

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
    data = yf.download(etf, start=start_date, end=end_date)["Adj Close"].dropna()

    # Ensure we have sufficient data before proceeding
    if data.empty:
        st.warning(f"No data for {etf}. Skipping.")
        continue

    # Calculate mean and standard deviation for the last month
    # Ensure 'data' is a Series by selecting only the 'Adj Close' column if 'data' is a DataFrame
    
    st.write(data.mean())
    st.remove(data.mean())
    mean_last_month = float (data.mean())
    std_last_month = float(data.std())

    # Calculate skewness and kurtosis, default to 0 if invalid
    skewness_last_month = round(float(skew(data)), 2) if len(data) > 1 else 0
    kurtosis_last_month = round(float(kurtosis(data)), 2) if len(data) > 1 else 0
    skewness_list.append(skewness_last_month)
    kurtosis_list.append(kurtosis_last_month)

    # Get the most recent data point (current price)
    current_price = float(data.iloc[-1])
    current_price_list.append(round(current_price, 2))

    # Calculate Z score for the current price
    z_score_current_price = round((current_price - mean_last_month) / std_last_month, 2) if std_last_month != 0 else 0
    z_score_list.append(z_score_current_price)

# Create DataFrame to display the results
df = pd.DataFrame({
    "ETF Symbol": etfs,
    "Current Price": current_price_list,
    "Z Score": z_score_list,
    "Skewness": skewness_list,
    "Kurtosis": kurtosis_list
})

# Fill NaN values and sort by Z-Score
#df = df.fillna(0)  # Fill NaN values with 0

# Display the table
df_sorted = df.sort_values(by='Z Score', ascending=True)
st.table(df_sorted)
