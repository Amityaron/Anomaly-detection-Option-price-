import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis
import streamlit as st

st.title('Welcome to my Anomaly Detection App')
st.subheader('Based on Z-Score')
st.image("pages/skewness.png")

st.markdown("""
### Skewness Definitions:
- Highly positively skewed: Skewness > 1
- Moderately positively skewed: 0.5 < Skewness ≤ 1
- Approximately symmetric: -0.5 ≤ Skewness ≤ 0.5
- Moderately negatively skewed: -1 ≤ Skewness < -0.5
- Highly negatively skewed: Skewness < -1
""")

default_etfs = ["IVV", "IVW", "XLK", "SOXX", "XLF", "BTC-USD", "URTH", "IXN", "IAU", "SMH", "IYW", "SVIX", "CNDX.L"]

if 'etfs' not in st.session_state:
    st.session_state.etfs = default_etfs.copy()

st.sidebar.header("Add or Remove Tickers")

new_ticker = st.sidebar.text_input("Enter a new ETF ticker (e.g., AAPL)")

if st.sidebar.button("Add Ticker"):
    ticker = new_ticker.strip().upper()
    if ticker:
        if ticker not in st.session_state.etfs:
            st.session_state.etfs.append(ticker)
            st.sidebar.success(f"Added {ticker} to the list.")
        else:
            st.sidebar.warning(f"{ticker} is already in the list.")
    else:
        st.sidebar.warning("Please enter a ticker.")

tickers_to_remove = st.sidebar.multiselect("Select tickers to remove", st.session_state.etfs)

if st.sidebar.button("Remove Selected Tickers"):
    for ticker in tickers_to_remove:
        if ticker in st.session_state.etfs:
            st.session_state.etfs.remove(ticker)
    st.sidebar.success(f"Removed selected tickers: {', '.join(tickers_to_remove)}")

etfs = st.session_state.etfs
results = []

for etf in etfs:
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(days=22)

    try:
        raw_data = yf.download(etf, start=start_date, end=end_date, progress=False)

        if raw_data.empty:
            st.warning(f"No data for {etf}. Skipping.")
            continue

        if "Close" not in raw_data.columns:
            st.warning(f"'Close' column not found for {etf}. Skipping.")
            continue

        data = raw_data["Close"]

        # Ensure data is a Series, not a DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.squeeze()

        data = pd.to_numeric(data, errors="coerce").dropna()

        if data.empty:
            st.warning(f"No valid closing price data for {etf}. Skipping.")
            continue

        mean_last_month = data.mean()
        std_last_month = data.std()
        values = data.to_numpy(dtype=float)

        skewness_last_month = skew(values) if len(values) > 1 else 0
        kurtosis_last_month = kurtosis(values) if len(values) > 1 else 0
        current_price = data.iloc[-1]

        z_score_current_price = round((current_price - mean_last_month) / std_last_month, 2) if std_last_month != 0 else 0

        results.append({
            "ETF Symbol": etf,
            "Current Price": round(float(current_price), 2),
            "Z Score": z_score_current_price,
            "Skewness": round(float(skewness_last_month), 4),
            "Kurtosis": round(float(kurtosis_last_month), 4)
        })

    except Exception as e:
        st.warning(f"Error processing {etf}: {e}")

df = pd.DataFrame(results)

if not df.empty:
    df_sorted = df.sort_values(by="Z Score", ascending=True)
    st.table(df_sorted)
else:
    st.info("No results to display.")
