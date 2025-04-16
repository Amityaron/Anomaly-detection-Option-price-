import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Large-Cap Stock Fundamentals", layout="wide")

st.title("🏦 Large-Cap Stock Fundamentals Viewer")

st.markdown("""
This app displays key financial data for selected **large-cap stocks** using Yahoo Finance.  
You can also **add more custom stocks** using the form below.
""")

# Expanded large-cap stock list (15 stocks)
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
    "Exxon Mobil (XOM)": "XOM"
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
            "Market Cap": format_market_cap(market_cap),
            "P/E": pe if pe is not None else 0,
            "P/B": pb if pb is not None else 0,
            "Debt/Equity": debt_equity if debt_equity is not None else 0,
            "Earnings Growth (%)": earnings_growth * 100 if earnings_growth is not None else 0,
            "(P/E)*(P/B)": (pe * pb) if pe and pb else 0,
            "Dividend Payment": dividend,
            "10Y Positive Earnings": ten_year_positive
        }

    except Exception as e:
        st.warning(f"⚠️ Could not fetch data for {ticker}: {e}")
        return None

# Load initial large-cap stocks
large_cap_df = pd.DataFrame([
    fetch_fundamentals(t)
    for t in large_cap_stocks.values()
    if fetch_fundamentals(t) is not None
])

# Storage for added custom stocks
if "custom_stocks_df" not in st.session_state:
    st.session_state.custom_stocks_df = pd.DataFrame(columns=large_cap_df.columns)

# --- Add More Stocks (above the table)
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

# Combine full dataset
full_df = pd.concat([large_cap_df, st.session_state.custom_stocks_df], ignore_index=True)

# --- Show Table
st.subheader("📈 Stock Fundamentals Table")
if full_df.empty:
    st.error("No data available.")
else:
    st.dataframe(full_df.style.format({
        "P/E": "{:.2f}",
        "P/B": "{:.2f}",
        "Debt/Equity": "{:.2f}",
        "Earnings Growth (%)": "{:.2f}%",
        "(P/E)*(P/B)": "{:.2f}"
    }), use_container_width=True)
