import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Large-Cap Stock Fundamentals", layout="wide")

st.title("🏦 Large-Cap Stock Fundamentals Viewer")

st.markdown("""
This dashboard automatically loads and displays key financial metrics for selected **large-cap stocks** using Yahoo Finance data.
It includes calculations such as **(P/E)*(P/B)**, dividend checks, and a rough check for positive earnings over 10 years.
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

@st.cache_data(show_spinner=True)
def fetch_all_fundamentals(ticker_dict):
    all_data = []

    for name, ticker in ticker_dict.items():
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            pe = info.get("trailingPE", None)
            pb = info.get("priceToBook", None)
            debt_equity = info.get("debtToEquity", None)
            earnings_growth = info.get("earningsQuarterlyGrowth", None)
            dividend = "Yes" if info.get("dividendYield", 0) else "No"

            # Try to infer 10Y positive earnings from net income
            ten_year_positive = "Unknown"
            try:
                net_income = stock.income_stmt.loc["Net Income"]
                if (net_income < 0).any():
                    ten_year_positive = "No"
                else:
                    ten_year_positive = "Yes"
            except:
                pass

            all_data.append({
                "Ticker": ticker,
                "P/E": pe if pe is not None else 0,
                "P/B": pb if pb is not None else 0,
                "Debt/Equity": debt_equity if debt_equity is not None else 0,
                "Earnings Growth (%)": earnings_growth * 100 if earnings_growth is not None else 0,
                "(P/E)*(P/B)": (pe * pb) if pe and pb else 0,
                "Dividend Payment": dividend,
                "10Y Positive Earnings": ten_year_positive
            })
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch data for {ticker}: {e}")

    return pd.DataFrame(all_data)

# Load all data once
stocks_df = fetch_all_fundamentals(large_cap_stocks)

# --- Display section
st.subheader("📈 Large-Cap Stock Fundamentals Table")
if stocks_df.empty:
    st.error("No data could be loaded. Please check your internet connection or Yahoo Finance availability.")
else:
    st.dataframe(stocks_df.style.format({
        "P/E": "{:.2f}",
        "P/B": "{:.2f}",
        "Debt/Equity": "{:.2f}",
        "Earnings Growth (%)": "{:.2f}%",
        "(P/E)*(P/B)": "{:.2f}"
    }), use_container_width=True)
