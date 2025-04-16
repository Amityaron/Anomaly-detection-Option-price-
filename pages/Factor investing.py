import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Large-Cap Stock Fundamentals", layout="wide")

st.title("🏦 Large-Cap Stock Fundamentals Viewer")

st.markdown("""
Select a large-cap stock from the dropdown or enter a custom ticker.
The app fetches financial data from Yahoo Finance, calculates **(P/E)*(P/B)**, checks for dividends, and reviews long-term earnings trends.
""")

# Predefined list of large-cap tickers
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
    "Johnson & Johnson (JNJ)": "JNJ"
}

# Session-state dataframe
if "stocks_df" not in st.session_state:
    st.session_state.stocks_df = pd.DataFrame(columns=[
        "Ticker", "P/E", "P/B", "Debt/Equity", "Earnings Growth (%)",
        "(P/E)*(P/B)", "Dividend Payment", "10Y Positive Earnings"
    ])

def fetch_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get("trailingPE", None)
        pb = info.get("priceToBook", None)
        debt_equity = info.get("debtToEquity", None)
        earnings_growth = info.get("earningsQuarterlyGrowth", None)

        dividend = "Yes" if info.get("dividendYield", 0) else "No"

        # Attempt to determine 10Y positive earnings
        financials = stock.financials
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
            "P/E": pe if pe is not None else 0,
            "P/B": pb if pb is not None else 0,
            "Debt/Equity": debt_equity if debt_equity is not None else 0,
            "Earnings Growth (%)": earnings_growth * 100 if earnings_growth is not None else 0,
            "(P/E)*(P/B)": (pe * pb) if pe and pb else 0,
            "Dividend Payment": dividend,
            "10Y Positive Earnings": ten_year_positive
        }

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# --- Input section
with st.expander("➕ Add Stock"):
    with st.form("stock_form"):
        col1, col2 = st.columns([2, 1])
        selected_label = col1.selectbox("Select Large-Cap Stock", list(large_cap_stocks.keys()))
        manual_input = col2.text_input("Or enter a custom ticker", "")
        submit = st.form_submit_button("Fetch and Add")

        if submit:
            ticker = manual_input.strip().upper() if manual_input else large_cap_stocks[selected_label]
            data = fetch_fundamentals(ticker)
            if data:
                st.session_state.stocks_df = pd.concat(
                    [st.session_state.stocks_df, pd.DataFrame([data])],
                    ignore_index=True
                )
                st.success(f"✅ Added {ticker} to the table.")

# --- Display section
st.subheader("📈 Current Stock Data")
if st.session_state.stocks_df.empty:
    st.info("No stocks added yet.")
else:
    st.dataframe(st.session_state.stocks_df.style.format({
        "P/E": "{:.2f}",
        "P/B": "{:.2f}",
        "Debt/Equity": "{:.2f}",
        "Earnings Growth (%)": "{:.2f}%",
        "(P/E)*(P/B)": "{:.2f}"
    }), use_container_width=True)
