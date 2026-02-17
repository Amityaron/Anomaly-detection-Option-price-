# app.py
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Drawdown Hit Probability", layout="wide")

st.title("Probability of hitting a drawdown using future LOWs")
st.write(
    "For each start day **t**, compute a base price and check whether "
    "**min(Low[t+1..t+h])** falls below **base[t] · (1 − X%)** within the next **h** trading days."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Inputs")

ticker = st.sidebar.text_input("Ticker", value="SPY").strip().upper()
start_date = st.sidebar.text_input("Start date (YYYY-MM-DD)", value="2008-01-01").strip()

default_horizons = [1, 3, 5, 10, 15, 20, 25, 30]
horizons = st.sidebar.multiselect(
    "Horizons (trading days)",
    options=list(range(1, 61)),
    default=default_horizons
)
horizons = sorted(set(horizons))

default_thresholds = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
thresholds_pct = st.sidebar.multiselect(
    "Thresholds (%)",
    options=[1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40],
    default=[2, 3, 5, 7, 10, 15, 20, 25, 30]
)
thresholds = sorted([x / 100.0 for x in set(thresholds_pct)])

base_mode = st.sidebar.selectbox(
    "Base definition",
    options=["(High + Low) / 2", "Close", "Low + alpha*(High-Low)"],
    index=0
)

alpha = 0.5
if base_mode == "Low + alpha*(High-Low)":
    alpha = st.sidebar.slider("alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

run = st.sidebar.button("Run")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def download_ohlc(ticker_: str, start_: str) -> pd.DataFrame:
    df = yf.download(ticker_, start=start_, auto_adjust=True, progress=False)
    return df

def get_col(df: pd.DataFrame, col: str, ticker_: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        if (col, ticker_) in df.columns:
            return df[(col, ticker_)]
        if (ticker_, col) in df.columns:
            return df[(ticker_, col)]
        matches = [c for c in df.columns if col in c]
        if not matches:
            raise RuntimeError(f"Could not find {col} in columns: {df.columns}")
        return df[matches[0]]
    if col not in df.columns:
        raise RuntimeError(f"Could not find {col} in columns: {df.columns}")
    return df[col]

def compute_table(df: pd.DataFrame, ticker_: str, horizons_: list[int], thresholds_: list[float], base_mode_: str, alpha_: float):
    high = get_col(df, "High", ticker_).dropna().astype(float)
    low  = get_col(df, "Low",  ticker_).dropna().astype(float)
    close = get_col(df, "Close", ticker_).dropna().astype(float)

    idx = high.index.intersection(low.index).intersection(close.index)
    high = high.loc[idx]
    low = low.loc[idx]
    close = close.loc[idx]

    if base_mode_ == "(High + Low) / 2":
        base = (high + low) * 0.5
    elif base_mode_ == "Close":
        base = close
    else:
        # Low + alpha*(High-Low)
        base = low + alpha_ * (high - low)

    horizon_labels = [f"{h} days" for h in horizons_]
    threshold_labels = [f"{int(round(x * 100))}%" for x in thresholds_]

    result = np.zeros((len(horizons_), len(thresholds_)), dtype=float)
    window_counts = []

    for i, h in enumerate(horizons_):
        future_lows = pd.concat([low.shift(-k) for k in range(1, h + 1)], axis=1)
        min_future_low = future_lows.min(axis=1)

        mask = min_future_low.notna() & base.notna()
        min_future_low = min_future_low[mask]
        base_valid = base[mask]

        window_counts.append(int(mask.sum()))

        drawdown = (min_future_low / base_valid) - 1.0

        for j, thr in enumerate(thresholds_):
            result[i, j] = (drawdown <= -thr).mean() * 100.0

    table = pd.DataFrame(result, index=horizon_labels, columns=threshold_labels)
    return table, window_counts, (idx.min().date(), idx.max().date())

# -----------------------------
# Run
# -----------------------------
if run:
    if not horizons:
        st.error("Please select at least one horizon.")
        st.stop()
    if not thresholds:
        st.error("Please select at least one threshold.")
        st.stop()

    with st.spinner("Downloading data..."):
        df = download_ohlc(ticker, start_date)

    if df.empty:
        st.error("No data returned from yfinance. Check ticker/start date.")
        st.stop()

    with st.spinner("Computing table..."):
        table, counts, (dmin, dmax) = compute_table(df, ticker, horizons, thresholds, base_mode, alpha)

    st.subheader(f"Ticker: {ticker}")
    st.caption(f"Data range used: {dmin} → {dmax}")

    st.markdown(
        "**Event definition:** for each start day *t* and horizon *h*, "
        "compute `min(Low[t+1..t+h])` and check whether it is <= `base[t] · (1 − X%)`."
    )

    # Show as numbers (percent)
    st.dataframe(table.style.format("{:.2f}%"), use_container_width=True)

    # Window counts
    counts_df = pd.DataFrame({"Horizon": [f"{h} days" for h in horizons], "Windows used": counts})
    st.markdown("### Rolling window counts")
    st.dataframe(counts_df, use_container_width=True)

    # Optional download
    csv = table.to_csv(index=True).encode("utf-8")
    st.download_button("Download table as CSV", data=csv, file_name=f"{ticker}_drawdown_table.csv", mime="text/csv")

else:
    st.info("Set parameters on the left and click **Run**.")
