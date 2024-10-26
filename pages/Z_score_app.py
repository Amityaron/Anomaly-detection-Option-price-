# Import required libraries
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from scipy.stats import skew, kurtosis

# Set the app title and other headers
st.title('Anomaly Detection App')
st.subheader('Z-Score-Based Anomaly Detection')
st.write('This app detects anomalies based on Z Score calculations for ETFs.')
st.image("pages/skewness.png")

# Skewness interpretation
st.markdown("""
    - **Highly positively skewed**: Skewness > 1
    - **Moderately positively skewed**: 0.5 < Skewness ≤ 1
    - **Approximately symmetric**: -0.5 ≤ Skewness ≤ 0.5
    - **Moderately negatively skewed**: -1 ≤ Skewness < -0.5
    - **Highly negatively skewed**: Skewness < -1        
""")

# Default ETF tickers
etfs = ["QQQ", "XLK", "XLF", "SOXX", "URTH"]

# Sidebar ticker management
st.sidebar.header("Manage Tickers")
new_ticker = st.sidebar.text_input("Add a new ETF ticker")
if st.sidebar.button("Add") and new_ticker:
    etfs.append(new_ticker.upper())
    st.sidebar.success(f"{new_ticker.upper()} added.")
    
tickers_to_remove = st.sidebar.multiselect("Remove tickers", etfs)
if st.sidebar.button("Remove"):
    etfs = [etf for etf in etfs if etf not in tickers_to_remove]
    st.sidebar.success(f"Removed: {', '.join(tickers_to_remove)}")

# Data collection and calculations
data_summary = []
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(days=22)

for etf in etfs:
    data = yf.download(etf, start=start_date, end=end_date)["Adj Close"].dropna()

    if data.empty:
        st.warning(f"No data for {etf}. Skipping.")
        continue

    # Calculate Z-score, skewness, and kurtosis
    mean = data.mean()
    std = data.std()
    st.write(std)
    # Calculate Z-score only if std is not NaN or 0, else set Z-score to 0
    last_price = data.iloc[-1]
        z_score = (last_price - mean) / std
    skewness = skew(data) if len(data) > 1 else 0
    kurtosis_val = kurtosis(data) if len(data) > 1 else 0

    # Append results
    data_summary.append({
        "ETF": etf,
        "Current Price": round(last_price, 2),
        "Z Score": round(z_score, 2),
        "Skewness": round(skewness, 2),
        "Kurtosis": round(kurtosis_val, 2)
    })

# Display results
df = pd.DataFrame(data_summary)
st.table(df.sort_values("Z Score"))
