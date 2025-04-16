import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Large-Cap Stock Fundamentals", layout="wide")

st.title("🏦 Large-Cap Stock Fundamentals Viewer")

st.markdown("""
This app displays key financial data for selected **large-cap stocks** using Yahoo Finance.  
You can also **add more custom stocks** using the form below.
""")

large_cap_stocks = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Alphabet (GOOGL)": "GOOGL",
    "Meta (META)": "META",
    "NVIDIA (NVDA)": "NVDA",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "Visa (V)": "V",
    "JPMorgan Chase (JPM)": "JPM",
    "Johnson & Johnson (JNJ)": "JNJ",
    "UnitedHealth (UNH)": "UNH",
    "Procter & Gamble (PG)": "PG",
    "Mastercard (MA)": "MA",
    "Home Depot (HD)": "HD",
    "Exxon Mobil (XOM)": "XOM",
    "Costco (COST)": "COST",
    "PepsiCo (PEP)": "PEP",
    "AbbVie (ABBV)": "ABBV",
    "Coca-Cola (KO)": "KO",
    "Pfizer (PFE)": "PFE",
    "Chevron (CVX)": "CVX",
    "Intel (INTC)": "INTC",
    "Cisco (CSCO)": "CSCO",
    "Walmart (WMT)": "WMT",
    "Adobe (ADBE)": "ADBE"
}

def format_market_cap(value):
    if value is None or value == 0:
        return "N/A"
    trillions = 1_000_000_000_000
    billions = 1_000_000_000
    if value >= trillions:
        return f"${value / trillions:.2f}T"
    elif value >= billions:
        return f"${value / billions:.2f}B"
    else:
        return f"${value:,}"

@st.cache_data(show_spinner=True)
def fetch_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get("trailingPE", None)
        pb = info.get("priceToBook", None)
        debt_equity = info.get("debtToEquity", None)
        earnings_growth = info.get("earningsQuarterlyGrowth", None)
        dividend = "Yes" if info.get("dividendYield", 0) else "No"
        sector = info.get("sector", "Unknown")
        market_cap = info.get("marketCap", None)
        roe = info.get("returnOnEquity", None)
        op_margin = info.get("operatingMargins", None)

        ten_year_positive = "Unknown"
        try:
            net_income = stock.income_stmt.loc["Net Income"]
            if (net_income < 0).any():
                ten_year_positive = "No"
            else:
                ten_year_positive = "Yes"
        except:
            pass

        return {
            "Ticker": ticker.upper(),
            "Sector": sector,
            "Market Cap": market_cap,
            "P/E": pe if pe is not None else np.nan,
            "P/B": pb if pb is not None else np.nan,
            "Debt/Equity": debt_equity if debt_equity is not None else np.nan,
            "Earnings Growth (%)": earnings_growth * 100 if earnings_growth is not None else np.nan,
            "(P/E)*(P/B)": (pe * pb) if pe and pb else np.nan,
            "ROE (%)": roe * 100 if roe is not None else np.nan,
            "Operating Margin (%)": op_margin * 100 if op_margin is not None else np.nan,
            "Dividend Payment": dividend,
            "10Y Positive Earnings": ten_year_positive
        }

    except Exception as e:
        st.warning(f"⚠️ Could not fetch data for {ticker}: {e}")
        return None

large_cap_df = pd.DataFrame([
    fetch_fundamentals(t)
    for t in large_cap_stocks.values()
    if fetch_fundamentals(t) is not None
])

if "custom_stocks_df" not in st.session_state:
    st.session_state.custom_stocks_df = pd.DataFrame(columns=large_cap_df.columns)

st.subheader("➕ Add Another Stock")
with st.form("add_stock_form"):
    new_ticker = st.text_input("Enter Stock Ticker (e.g. TSLA, NFLX)").upper().strip()
    add_button = st.form_submit_button("Fetch & Add")

    if add_button and new_ticker:
        full_current_df = pd.concat([large_cap_df, st.session_state.custom_stocks_df])
        if new_ticker in full_current_df["Ticker"].values:
            st.warning(f"{new_ticker} is already in the table.")
        else:
            result = fetch_fundamentals(new_ticker)
            if result:
                st.session_state.custom_stocks_df = pd.concat(
                    [st.session_state.custom_stocks_df, pd.DataFrame([result])],
                    ignore_index=True
                )
                st.success(f"{new_ticker} added to the table.")

full_df = pd.concat([large_cap_df, st.session_state.custom_stocks_df], ignore_index=True)

st.subheader("📈 Stock Fundamentals Table")
if full_df.empty:
    st.error("No data available.")
else:
    num_cols = [
        "P/E", "P/B", "Debt/Equity", "Earnings Growth (%)", "(P/E)*(P/B)",
        "ROE (%)", "Operating Margin (%)"
    ]
    for col in num_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').round(2)

    full_df["Market Cap"] = pd.to_numeric(full_df["Market Cap"], errors="coerce")
    full_df["Market Cap ($B)"] = (full_df["Market Cap"] / 1e9).round(2)
    full_df.drop(columns="Market Cap", inplace=True)

    st.dataframe(full_df, use_container_width=True)

    # 📊 Bar Plot by Sector - show all stocks in one plot grouped by sector
    st.subheader("📉 Market Cap by Sector")
    fig, ax = plt.subplots(figsize=(16, 8))
    sorted_df = full_df.sort_values("Sector")
    sns.barplot(data=sorted_df, x="Ticker", y="Market Cap ($B)", hue="Sector", dodge=False, ax=ax)

    # Red average line per sector
    for sector in sorted_df["Sector"].unique():
        sector_avg = sorted_df[sorted_df["Sector"] == sector]["Market Cap ($B)"].mean()
        sector_tickers = sorted_df[sorted_df["Sector"] == sector]["Ticker"].values
        sector_xs = [sorted_df.index.get_loc(i) for i in sorted_df[sorted_df["Sector"] == sector].index]
        ax.hlines(sector_avg, xmin=min(sector_xs)-0.4, xmax=max(sector_xs)+0.4, color="red", linestyle="--")

    ax.set_title("Market Cap by Ticker (Grouped by Sector)", fontsize=14)
    ax.set_ylabel("Market Cap (Billions)")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=90)
    ax.legend(loc='upper right')
    st.pyplot(fig)
