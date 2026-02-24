# app.py
import re
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="EPS % vs Stock % (Earnings-to-Earnings)", layout="wide")

st.title("EPS % Change vs Stock % Change (per earnings / quarter)")

with st.sidebar:
    ticker = st.text_input("Stock ticker", value="NVDA").strip().upper()
    years_back = st.slider("Years back", min_value=1, max_value=20, value=10)
    run = st.button("Run")


def _safe_to_datetime_series(s: pd.Series) -> pd.Series:
    """Convert series to timezone-naive datetime; robust to strings like 'Aug 16, 2022, 2 AM EDT'."""
    s = s.astype(str)

    # remove common trailing timezone tokens (EDT, EST, GMT, UTC, etc.)
    s = s.str.replace(r"\b[A-Z]{2,5}\b$", "", regex=True).str.strip()

    dt = pd.to_datetime(s, errors="coerce")

    # if timezone-aware, convert to naive
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(None)
    except Exception:
        pass

    return dt


@st.cache_data(ttl=60 * 60)
def build_eps_vs_stock_df(ticker: str, years_back: int) -> pd.DataFrame:
    stock = yf.Ticker(ticker)

    # ---------- EPS (use earnings_history; earnings_dates can KeyError in yfinance) ----------
    eps_raw = pd.DataFrame()
    try:
        eps_raw = stock.get_earnings_history()
    except Exception:
        # fallback for older yfinance versions
        try:
            eps_raw = stock.earnings_history
        except Exception:
            eps_raw = pd.DataFrame()

    if eps_raw is None or eps_raw.empty:
        return pd.DataFrame()

    # Find columns in a version-tolerant way
    date_col = None
    rep_eps_col = None

    for c in eps_raw.columns:
        lc = str(c).lower()
        if date_col is None and ("earnings" in lc and "date" in lc):
            date_col = c
        if rep_eps_col is None and ("reported" in lc and "eps" in lc):
            rep_eps_col = c

    if date_col is None or rep_eps_col is None:
        return pd.DataFrame()

    eps_df = eps_raw[[date_col, rep_eps_col]].rename(
        columns={date_col: "EPS_Date", rep_eps_col: "Reported EPS"}
    )

    eps_df["EPS_Date"] = _safe_to_datetime_series(eps_df["EPS_Date"])
    eps_df["Reported EPS"] = pd.to_numeric(eps_df["Reported EPS"], errors="coerce")

    start_date = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_back)
    eps_df = eps_df[(eps_df["EPS_Date"] >= start_date) & (eps_df["EPS_Date"] <= pd.Timestamp.today())]
    eps_df = eps_df.dropna(subset=["EPS_Date", "Reported EPS"]).sort_values("EPS_Date").reset_index(drop=True)

    if len(eps_df) < 2:
        return pd.DataFrame()

    eps_df["EPS_Change_%"] = eps_df["Reported EPS"].pct_change() * 100

    # ---------- PRICE (daily) ----------
    price_df = stock.history(period=f"{years_back + 1}y", interval="1d")
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return pd.DataFrame()

    price_df = price_df[["Close"]].reset_index()

    # yfinance sometimes returns index name as Date or Datetime
    if "Date" in price_df.columns:
        price_df = price_df.rename(columns={"Date": "Price_Date"})
    elif "Datetime" in price_df.columns:
        price_df = price_df.rename(columns={"Datetime": "Price_Date"})
    else:
        # last resort: take first column as date
        price_df = price_df.rename(columns={price_df.columns[0]: "Price_Date"})

    price_df["Price_Date"] = pd.to_datetime(price_df["Price_Date"], errors="coerce")
    if getattr(price_df["Price_Date"].dt, "tz", None) is not None:
        price_df["Price_Date"] = price_df["Price_Date"].dt.tz_convert(None)

    price_df = price_df.dropna(subset=["Price_Date"]).sort_values("Price_Date")

    # ---------- ALIGN close on/before earnings date ----------
    aligned = pd.merge_asof(
        eps_df.sort_values("EPS_Date"),
        price_df.sort_values("Price_Date"),
        left_on="EPS_Date",
        right_on="Price_Date",
        direction="backward",
    )

    # If Close missing for some aligned rows, drop them before pct_change
    aligned = aligned.dropna(subset=["Close"]).reset_index(drop=True)
    if len(aligned) < 2:
        return pd.DataFrame()

    aligned["Stock_Change_%"] = aligned["Close"].pct_change() * 100

    final_df = aligned[["EPS_Date", "Reported EPS", "EPS_Change_%", "Price_Date", "Close", "Stock_Change_%"]].copy()
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
        final_df = build_eps_vs_stock_df(ticker, years_back)

    if final_df.empty:
        st.warning(
            "No usable earnings (EPS) data found for this ticker (or not enough rows). "
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
