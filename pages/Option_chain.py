# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import date


st.set_page_config(page_title="Options Chain CAGR", layout="wide")

st.title("📊 Options Chain - Calls | Strike | Puts")
st.write("CAGR is calculated using LAST option price only.")


# =====================================================
# Helper functions
# =====================================================
def get_stock_price(ticker_obj):
    try:
        price = ticker_obj.fast_info.get("last_price")
        if price is not None and price > 0:
            return float(price)
    except Exception:
        pass

    hist = ticker_obj.history(period="5d")
    if hist.empty:
        raise ValueError("Could not get stock price.")

    return float(hist["Close"].dropna().iloc[-1])


def bs_delta(option_type, S, K, T, r, iv):
    if pd.isna(iv) or iv <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))

    if option_type == "CALL":
        return norm.cdf(d1)

    return norm.cdf(d1) - 1


def calc_cagr_by_last_price(last_price, capital_base, dte):
    """
    CAGR = (1 + last_price / capital_base) ^ (365 / DTE) - 1
    """
    if pd.isna(last_price) or pd.isna(capital_base):
        return np.nan

    if last_price <= 0 or capital_base <= 0 or dte <= 0:
        return np.nan

    return ((1 + last_price / capital_base) ** (365 / dte) - 1) * 100


def prepare_calls(calls, stock_price, dte, r):
    calls = calls.copy()
    T = dte / 365

    calls["Call IV %"] = calls["impliedVolatility"] * 100

    calls["Call Delta"] = calls.apply(
        lambda row: bs_delta(
            option_type="CALL",
            S=stock_price,
            K=row["strike"],
            T=T,
            r=r,
            iv=row["impliedVolatility"]
        ),
        axis=1
    )

    calls["Call ITM"] = stock_price > calls["strike"]

    # Call CAGR by LAST CALL PRICE
    # Capital base = stock price, like covered call logic
    calls["Call CAGR %"] = calls.apply(
        lambda row: calc_cagr_by_last_price(
            last_price=row["lastPrice"],
            capital_base=stock_price,
            dte=dte
        ),
        axis=1
    )

    calls = calls.rename(columns={
        "lastPrice": "Call Last",
        "bid": "Call Bid",
        "ask": "Call Ask",
        "volume": "Call Volume",
        "openInterest": "Call OI"
    })

    keep_cols = [
        "strike",
        "Call Last",
        "Call Bid",
        "Call Ask",
        "Call Volume",
        "Call OI",
        "Call IV %",
        "Call Delta",
        "Call CAGR %",
        "Call ITM"
    ]

    return calls[keep_cols]


def prepare_puts(puts, stock_price, dte, r):
    puts = puts.copy()
    T = dte / 365

    puts["Put IV %"] = puts["impliedVolatility"] * 100

    puts["Put Delta"] = puts.apply(
        lambda row: bs_delta(
            option_type="PUT",
            S=stock_price,
            K=row["strike"],
            T=T,
            r=r,
            iv=row["impliedVolatility"]
        ),
        axis=1
    )

    puts["Put ITM"] = stock_price < puts["strike"]

    # Put CAGR by LAST PUT PRICE
    # Capital base = strike, like cash-secured put logic
    puts["Put CAGR %"] = puts.apply(
        lambda row: calc_cagr_by_last_price(
            last_price=row["lastPrice"],
            capital_base=row["strike"],
            dte=dte
        ),
        axis=1
    )

    puts = puts.rename(columns={
        "lastPrice": "Put Last",
        "bid": "Put Bid",
        "ask": "Put Ask",
        "volume": "Put Volume",
        "openInterest": "Put OI"
    })

    keep_cols = [
        "strike",
        "Put Last",
        "Put Bid",
        "Put Ask",
        "Put Volume",
        "Put OI",
        "Put IV %",
        "Put Delta",
        "Put CAGR %",
        "Put ITM"
    ]

    return puts[keep_cols]


def style_itm(row):
    styles = [""] * len(row)

    for i, col in enumerate(row.index):
        if col.startswith("Call") and row.get("Call ITM") is True:
            styles[i] = "background-color: #264b70; color: white;"

        if col.startswith("Put") and row.get("Put ITM") is True:
            styles[i] = "background-color: #264b70; color: white;"

        if col == "Strike":
            styles[i] = "background-color: #333333; color: white; font-weight: bold;"

    return styles


# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("Inputs")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

risk_free_rate_pct = st.sidebar.number_input(
    "Risk-free rate for delta %",
    value=4.5,
    step=0.1
)

risk_free_rate = risk_free_rate_pct / 100

filter_near_money = st.sidebar.checkbox("Show only strikes near stock price", value=True)

moneyness_range = st.sidebar.slider(
    "Strike range +/- %",
    min_value=5,
    max_value=100,
    value=30,
    step=5
)

show_raw_data = st.sidebar.checkbox("Show raw calls and puts", value=False)


# =====================================================
# Main app
# =====================================================
if ticker:
    try:
        tk = yf.Ticker(ticker)
        expirations = list(tk.options)

        if len(expirations) == 0:
            st.error("No options found for this ticker.")
            st.stop()

        expiration = st.sidebar.selectbox("Expiration", expirations)

        if st.sidebar.button("Load option chain"):
            stock_price = get_stock_price(tk)

            exp_date = pd.to_datetime(expiration).date()
            dte = max((exp_date - date.today()).days, 1)

            chain = tk.option_chain(expiration)

            calls = prepare_calls(
                calls=chain.calls,
                stock_price=stock_price,
                dte=dte,
                r=risk_free_rate
            )

            puts = prepare_puts(
                puts=chain.puts,
                stock_price=stock_price,
                dte=dte,
                r=risk_free_rate
            )

            table = pd.merge(
                calls,
                puts,
                on="strike",
                how="outer"
            ).sort_values("strike")

            table = table.rename(columns={"strike": "Strike"})

            if filter_near_money:
                lower = stock_price * (1 - moneyness_range / 100)
                upper = stock_price * (1 + moneyness_range / 100)
                table = table[
                    (table["Strike"] >= lower) &
                    (table["Strike"] <= upper)
                ]

            # Put Strike in the middle, like Yahoo straddle view
            cols = [
                "Call Last",
                "Call Bid",
                "Call Ask",
                "Call Volume",
                "Call OI",
                "Call IV %",
                "Call Delta",
                "Call CAGR %",
                "Call ITM",

                "Strike",

                "Put Last",
                "Put Bid",
                "Put Ask",
                "Put Volume",
                "Put OI",
                "Put IV %",
                "Put Delta",
                "Put CAGR %",
                "Put ITM"
            ]

            table = table[cols]

            # Round numeric columns
            for col in table.columns:
                if pd.api.types.is_numeric_dtype(table[col]):
                    if "Volume" in col or "OI" in col:
                        table[col] = table[col].fillna(0).astype(int)
                    else:
                        table[col] = table[col].round(2)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ticker", ticker)
            c2.metric("Stock Price", f"{stock_price:,.2f}")
            c3.metric("Expiration", expiration)
            c4.metric("DTE", dte)

            st.warning(
                "CAGR here uses lastPrice only. LastPrice can be stale, "
                "and for ITM options it includes intrinsic value."
            )

            st.subheader("Yahoo Style Option Chain")

            styled_table = table.style.apply(style_itm, axis=1)

            st.dataframe(
                styled_table,
                use_container_width=True,
                height=800
            )

            csv = table.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"{ticker}_{expiration}_options_chain.csv",
                mime="text/csv"
            )

            if show_raw_data:
                st.subheader("Raw Calls")
                st.dataframe(chain.calls, use_container_width=True)

                st.subheader("Raw Puts")
                st.dataframe(chain.puts, use_container_width=True)

            st.markdown(
                """
                ### CAGR formulas used

                **Put CAGR by last put price**

                `Put CAGR = (1 + Put Last Price / Strike) ** (365 / DTE) - 1`

                **Call CAGR by last call price**

                `Call CAGR = (1 + Call Last Price / Stock Price) ** (365 / DTE) - 1`
                """
            )

    except Exception as e:
        st.error(f"Error: {e}")


# =====================================================
# Manual CAGR Calculator
# =====================================================
st.divider()
st.subheader("🧮 Manual CAGR Calculator")

st.write(
    "Enter the numbers manually and click Calculate. "
    "Formula: CAGR = (1 + premium / strike) ** (365 / DTE) - 1"
)

with st.form("manual_cagr_form"):
    calc_col1, calc_col2, calc_col3 = st.columns(3)

    manual_strike = calc_col1.number_input(
        "Strike / Capital Base",
        min_value=0.0,
        value=None,
        step=0.01,
        placeholder="Example: 50"
    )

    manual_premium = calc_col2.number_input(
        "Premium",
        min_value=0.0,
        value=None,
        step=0.01,
        placeholder="Example: 2"
    )

    manual_dte = calc_col3.number_input(
        "DTE",
        min_value=1,
        value=None,
        step=1,
        placeholder="Example: 84"
    )

    calculate_button = st.form_submit_button("Calculate CAGR")

if calculate_button:
    if manual_strike is None or manual_premium is None or manual_dte is None:
        st.error("Please enter Strike, Premium and DTE.")
    elif manual_strike <= 0:
        st.error("Strike / Capital Base must be greater than 0.")
    elif manual_dte <= 0:
        st.error("DTE must be greater than 0.")
    else:
        period_return = manual_premium / manual_strike

        simple_annualized = period_return * (365 / manual_dte) * 100

        manual_cagr = ((1 + period_return) ** (365 / manual_dte) - 1) * 100

        m1, m2, m3 = st.columns(3)

        m1.metric("Period Return", f"{period_return * 100:.2f}%")
        m2.metric("Simple Annualized", f"{simple_annualized:.2f}%")
        m3.metric("CAGR", f"{manual_cagr:.2f}%")
