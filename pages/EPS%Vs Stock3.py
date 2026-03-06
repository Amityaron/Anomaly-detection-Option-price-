# app.py
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="EPS % vs Stock % (Earnings-to-Earnings)", layout="wide")

st.title("EPS % Change vs Stock % Change (per earnings / quarter)")

# Put this in st.secrets in production:
# FMP_API_KEY = st.secrets["FMP_API_KEY"]
FMP_API_KEY = "PUT_YOUR_NEW_KEY_HERE"
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

with st.sidebar:
    ticker = st.text_input("Stock ticker", value="NVDA").strip().upper()
    years_back = st.slider("Years back", min_value=1, max_value=20, value=10)
    run = st.button("Run")

@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_fmp_quarterly_eps(ticker: str, years_back: int) -> pd.DataFrame:
    """
    Pull earnings reports from FMP and build:
    - EPS_Date
    - Reported EPS
    - EPS_Change_% (QoQ)
    - EPS_TTM
    """
    url = f"{FMP_BASE_URL}/earnings"
    params = {
        "symbol": ticker,
        "apikey": FMP_API_KEY,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Keep this flexible because field names can vary a bit
    date_candidates = ["date", "fillingDate", "acceptedDate"]
    eps_candidates = ["eps", "epsActual", "actualEps", "reportedEPS", "reportedEps"]

    date_col = next((c for c in date_candidates if c in df.columns), None)
    eps_col = next((c for c in eps_candidates if c in df.columns), None)

    if date_col is None or eps_col is None:
        return pd.DataFrame()

    df = df[[date_col, eps_col]].copy()
    df = df.rename(columns={date_col: "EPS_Date", eps_col: "Reported EPS"})

    df["EPS_Date"] = pd.to_datetime(df["EPS_Date"], errors="coerce")
    df["Reported EPS"] = pd.to_numeric(df["Reported EPS"], errors="coerce")

    df = (
        df.dropna(subset=["EPS_Date", "Reported EPS"])
          .sort_values("EPS_Date")
          .drop_duplicates(subset=["EPS_Date"], keep="last")
          .reset_index(drop=True)
    )

    start_date = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_back)
    df = df[df["EPS_Date"] >= start_date].copy()

    if len(df) < 2:
        return pd.DataFrame()

    df["EPS_Change_%"] = df["Reported EPS"].pct_change() * 100
    df["EPS_TTM"] = df["Reported EPS"].rolling(window=4, min_periods=4).sum()

    return df.reset_index(drop=True)

@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_price_history(ticker: str, years_back: int) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    price_df = stock.history(period=f"{years_back + 1}y", interval="1d")[["Close"]].reset_index()

    if price_df.empty:
        return pd.DataFrame()

    if "Date" in price_df.columns:
        price_df = price_df.rename(columns={"Date": "Price_Date"})
    else:
        price_df = price_df.rename(columns={price_df.columns[0]: "Price_Date"})

    price_df["Price_Date"] = pd.to_datetime(price_df["Price_Date"], errors="coerce")
    if getattr(price_df["Price_Date"].dt, "tz", None) is not None:
        price_df["Price_Date"] = price_df["Price_Date"].dt.tz_convert(None)

    price_df = (
        price_df.dropna(subset=["Price_Date", "Close"])
                .sort_values("Price_Date")
                .reset_index(drop=True)
    )
    return price_df

@st.cache_data(ttl=60 * 60, show_spinner=False)
def build_eps_vs_stock_df(ticker: str, years_back: int) -> pd.DataFrame:
    eps_df = get_fmp_quarterly_eps(ticker, years_back)
    if eps_df.empty:
        return pd.DataFrame()

    price_df = get_price_history(ticker, years_back)
    if price_df.empty:
        return pd.DataFrame()

    aligned = pd.merge_asof(
        eps_df.sort_values("EPS_Date"),
        price_df.sort_values("Price_Date"),
        left_on="EPS_Date",
        right_on="Price_Date",
        direction="backward"
    )

    aligned["Stock_Change_%"] = aligned["Close"].pct_change() * 100

    final_df = aligned[
        ["EPS_Date", "Reported EPS", "EPS_TTM", "EPS_Change_%", "Price_Date", "Close", "Stock_Change_%"]
    ].copy()

    final_df = final_df.dropna(subset=["EPS_Change_%", "Stock_Change_%"]).reset_index(drop=True)
    return final_df

def plot_df(final_df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(final_df["EPS_Date"], final_df["EPS_Change_%"], label="EPS % Change (QoQ)", marker="o")
    ax.plot(final_df["EPS_Date"], final_df["Stock_Change_%"], label="Stock % Change (Earnings-to-Earnings)", marker="x")
    ax.axhline(0)
    ax.set_title(f"{ticker} – Quarterly EPS % Change vs Quarterly Stock % Change")
    ax.set_xlabel("Earnings Date")
    ax.set_ylabel("Percent Change (%)")
    ax.grid(True)
    ax.legend()
    return fig

if run:
    if not ticker:
        st.error("Please enter a ticker (e.g., NVDA, AAPL).")
        st.stop()

    with st.spinner("Downloading data and building table..."):
        try:
            final_df = build_eps_vs_stock_df(ticker, years_back)
        except requests.HTTPError as e:
            st.error(f"FMP HTTP error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    if final_df.empty:
        st.warning(
            "No usable quarterly EPS data or price history was found for this ticker. "
            "Try another stock ticker."
        )
        st.stop()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Chart")
        fig = plot_df(final_df, ticker)
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.subheader("final_df")
        st.dataframe(final_df, use_container_width=True)

    st.download_button(
        "Download final_df as CSV",
        data=final_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_eps_vs_stock_{years_back}y.csv",
        mime="text/csv",
    )
else:
    st.info("Enter a ticker and click **Run**.")
