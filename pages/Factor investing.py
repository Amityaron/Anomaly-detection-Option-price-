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
        
        # If info is None or missing critical data, return None
        if not info or 'marketCap' not in info:
            st.warning(f"⚠️ No data found for {ticker}")
            return None
        
        # Fetch data, with default values if not available
        pe = info.get("trailingPE", np.nan)
        pb = info.get("priceToBook", np.nan)
        debt_equity = info.get("debtToEquity", np.nan)
        earnings_growth = info.get("earningsQuarterlyGrowth", np.nan)
        dividend = "Yes" if info.get("dividendYield", 0) else "No"
        sector = info.get("sector", "Unknown")
        market_cap = info.get("marketCap", np.nan)
        roe = info.get("returnOnEquity", np.nan)
        op_margin = info.get("operatingMargins", np.nan)

        # 10-year positive earnings check
        ten_year_positive = "Unknown"
        try:
            net_income = stock.financials.loc["Net Income"]
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
            "P/E": pe,
            "P/B": pb,
            "Debt/Equity": debt_equity,
            "Earnings Growth (%)": earnings_growth * 100 if earnings_growth is not np.nan else np.nan,
            "(P/E)*(P/B)": (pe * pb) if pe and pb else np.nan,
            "ROE (%)": roe * 100 if roe is not np.nan else np.nan,
            "Operating Margin (%)": op_margin * 100 if op_margin is not np.nan else np.nan,
            "Dividend Payment": dividend,
            "10Y Positive Earnings": ten_year_positive
        }

    except Exception as e:
        st.warning(f"⚠️ Could not fetch data for {ticker}: {e}")
        return None

large_cap_df = pd.DataFrame([fetch_fundamentals(t) for t in large_cap_stocks.values() if fetch_fundamentals(t) is not None])

# Handle custom stocks
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

# Combine data
full_df = pd.concat([large_cap_df, st.session_state.custom_stocks_df], ignore_index=True)

# Display fundamentals table
st.subheader("📈 Stock Fundamentals Table")
if full_df.empty:
    st.error("No data available.")
else:
    num_cols = ["P/E", "P/B", "Debt/Equity", "Earnings Growth (%)", "(P/E)*(P/B)", "ROE (%)", "Operating Margin (%)"]
    for col in num_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').round(2)

    full_df["Market Cap"] = pd.to_numeric(full_df["Market Cap"], errors="coerce")
    full_df["Market Cap ($B)"] = (full_df["Market Cap"] / 1e9).round(2)
    full_df.drop(columns="Market Cap", inplace=True)

    st.dataframe(full_df, use_container_width=True)

    
