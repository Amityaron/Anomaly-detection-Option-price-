# Import required libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from scipy.stats import skew, kurtosis

# Set the app title and other headers
st.title('Welcome to my anomaly detection app')
st.subheader('Based on Z Score')
st.write('Anomaly detection app based on Z Score')
st.image("pages/skewness.png")

# Explanation of skewness interpretation
st.markdown(
    """
    - Highly positively skewed: Skewness > 1
    - Moderately positively skewed: 0.5 < Skewness ≤ 1
    - Approximately symmetric: -0.5 ≤ Skewness ≤ 0.5
    - Moderately negatively skewed: -1 ≤ Skewness < -0.5
    - Highly negatively skewed: Skewness < -1        
    - Long signal when Z score ≤ -2 and Skewness is Approximately symmetric.
    - Long signal when Z score ≤ -2.5 and Skewness is Moderately/Highly positively skewed.
    """
)

# Define default ETF tickers
default_etfs = ["CNDX.L", "CSPX.L", "IUIT.L", "IUFS.L", "IWRD.L", "ISEU.L", "IBIT", "BTC-USD", "XLK", "SOXX", "IVW", "IETC", "IXN", "URTH", "ACWI"]

# Sidebar for adding/removing tickers
st.sidebar.header("Add or Remove Tickers")
new_ticker = st.sidebar.text_input("Enter a new ETF ticker (e.g., AAPL)")
if st.sidebar.button("Add Ticker"):
    if new_ticker.upper() not in default_etfs:
        default_etfs.append(new_ticker.upper())
        st.sidebar.success(f"Added {new_ticker.upper()} to the list.")
    else:
        st.sidebar.warning(f"{new_ticker.upper()} is already in the list.")
tickers_to_remove = st.sidebar.multiselect("Select tickers to remove", default_etfs)
if st.sidebar.button("Remove Selected Tickers"):
    for ticker in tickers_to_remove:
        default_etfs.remove(ticker)
    st.sidebar.success(f"Removed selected tickers: {', '.join(tickers_to_remove)}")

# Calculate Z-Score, Skewness, and Kurtosis for the updated ticker list
etfs = default_etfs
z_score_list, current_price_list, skewness_list, kurtosis_list = [], [], [], []
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(days=22)  # Last 22 days

for etf in etfs:
    data = yf.download(etf, start=start_date, end=end_date)["Adj Close"].dropna()

    # Skip if data is empty
    if data.empty:
        st.warning(f"No data found for {etf}. Skipping...")
        continue

    # Calculate mean and standard deviation
    mean_last_month = data.mean()
    std_last_month = data.std()

    # Calculate skewness and kurtosis, handling any potential issues
    skewness_last_month = skew(data) if len(data) > 1 else 0  # Avoid skew calculation on single-value series
    kurtosis_last_month = kurtosis(data) if len(data) > 1 else 0  # Avoid kurtosis on single-value series
    skewness_list.append(round(skewness_last_month, 2))
    kurtosis_list.append(round(kurtosis_last_month, 2))

    # Get the most recent data point and calculate Z-score
    current_price = data.iloc[-1]
    current_price_list.append(round(current_price, 2))
    z_score_current_price = round((current_price - mean_last_month) / std_last_month, 2)
    z_score_list.append(z_score_current_price)

# Create a DataFrame to display results
df = pd.DataFrame({
    "ETF Symbol": etfs,
    "Current Price": current_price_list,
    "Z Score": z_score_list,
    "Skewness": skewness_list,
    "Kurtosis": kurtosis_list
})

# Display the sorted table
st.table(df.sort_values(by='Z Score', ascending=True))
