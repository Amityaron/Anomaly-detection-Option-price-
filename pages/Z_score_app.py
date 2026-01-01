# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis
import streamlit as st

# Set the app title 
st.title('Welcome to my Anomaly Detection App')
st.subheader('Based on Z-Score')
st.image("pages/skewness.png")
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
default_etfs = ["IVV","IVW","XLK", "SOXX", "XLF", "BTC-USD", "URTH", "IXN","IAU","SMH","IYW","SVIX","CNDX.L"]

# Initialize session state for tickers if it doesn't exist
if 'etfs' not in st.session_state:
    st.session_state.etfs = default_etfs.copy()

# Sidebar for adding new tickers
st.sidebar.header("Add or Remove Tickers")

# Add new ticker input
new_ticker = st.sidebar.text_input("Enter a new ETF ticker (e.g., AAPL)")

# Add ticker button
if st.sidebar.button("Add Ticker"):
    if new_ticker.upper() not in st.session_state.etfs:
        st.session_state.etfs.append(new_ticker.upper())
        st.sidebar.success(f"Added {new_ticker.upper()} to the list.")
    else:
        st.sidebar.warning(f"{new_ticker.upper()} is already in the list.")

# Allow the user to remove tickers from the list
tickers_to_remove = st.sidebar.multiselect("Select tickers to remove", st.session_state.etfs)

# Remove selected tickers
if st.sidebar.button("Remove Selected Tickers"):
    for ticker in tickers_to_remove:
        st.session_state.etfs.remove(ticker)
    st.sidebar.success(f"Removed selected tickers: {', '.join(tickers_to_remove)}")

# Use the session state tickers for analysis
etfs = st.session_state.etfs
results = []

# Loop through each ETF ticker and calculate Z-score, skewness, kurtosis
for etf in etfs:
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(days=22)
    data = yf.download(etf, start=start_date, end=end_date)["Close"].dropna()

    if data.empty:
        st.warning(f"No data for {etf}. Skipping.")
        continue

    mean_last_month = float(data.mean())
    std_last_month = float(data.std())
    values = data.dropna().to_numpy(dtype=float)
    skewness_last_month = round(float(skew(values)), 2) if len(values) > 1 else 0
    kurtosis_last_month = round(float(kurtosis(values)), 2) if len(values) > 1 else 0
    current_price = float(data.iloc[-1])

    z_score_current_price = round((current_price - mean_last_month) / std_last_month, 2) if std_last_month != 0 else 0

    results.append({
        "ETF Symbol": etf,
        "Current Price": round(current_price, 2),
        "Z Score": z_score_current_price,
        "Skewness": skewness_last_month,
        "Kurtosis": kurtosis_last_month
    })

# Create DataFrame to display the results
df = pd.DataFrame(results)

# Fill NaN values and sort by Z-Score
df_sorted = df.sort_values(by='Z Score', ascending=True)
st.table(df_sorted)









