# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import date


st.set_page_config(page_title="Options Chain CAGR", layout="wide")

st.title("📊 Options Chain: Calls / Puts / IV / Delta / CAGR")
st.write("Enter a ticker and get the full option chain with IV, price, delta and annualized premium CAGR.")


# -----------------------------
# Helpers
# -----------------------------
def get_underlying_price(ticker_obj: yf.Ticker) -> float:
    """
    Try to get the latest underlying price.
    """
    try:
        price = ticker_obj.fast_info.get("last_price")
        if price is not None and price > 0:
            return float(price)
    except Exception:
        pass

    hist = ticker_obj.history(period="5d")
    if hist.empty:
        raise ValueError("Could not get underlying price.")
    return float(hist["Close"].dropna().iloc[-1])


def calc_bs_delta(option_type: str, S: float, K: float, T: float, r: float, iv: float) -> float:
    """
    Black-Scholes delta approximation.

    option_type: "CALL" or "PUT"
    S: underlying price
    K: strike
    T: time to expiration in years
    r: annual risk-free rate, e.g. 0.045
    iv: implied volatility as decimal, e.g. 0.55
    """
    if pd.isna(iv) or iv <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))

    if option_type == "CALL":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def calc_cagr(premium: float, collateral: float, dte: int) -> float:
    """
    CAGR from option premium.

    Example:
    premium = 2
    collateral = 50
    dte = 84

    CAGR = (1 + 2/50) ** (365/84) - 1
    """
    if pd.isna(premium) or premium <= 0 or collateral <= 0 or dte <= 0:
        return np.nan

    return (1 + premium / collateral) ** (365 / dte) - 1


def prepare_options_df(
    df: pd.DataFrame,
    option_type: str,
    expiration: str,
    underlying_price: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """
    Add option type, expiration, DTE, mid price, delta and CAGR.
    """
    if df.empty:
        return df

    df = df.copy()

    exp_date = pd.to_datetime(expiration).date()
    today = date.today()
    dte = max((exp_date - today).days, 0)
    T = dte / 365

    df["optionType"] = option_type
    df["expiration"] = expiration
    df["DTE"] = dte

    # Mid price is usually better than lastPrice for yield calculations
    df["midPrice"] = np.where(
        (df["bid"] > 0) & (df["ask"] > 0),
        (df["bid"] + df["ask"]) / 2,
        df["lastPrice"],
    )

    df["IV_%"] = df["impliedVolatility"] * 100

    df["delta"] = df.apply(
        lambda row: calc_bs_delta(
            option_type=option_type,
            S=underlying_price,
            K=row["strike"],
            T=T,
            r=risk_free_rate,
            iv=row["impliedVolatility"],
        ),
        axis=1,
    )

    # Capital logic:
    # PUT: cash-secured put collateral = strike
    # CALL: covered-call collateral = underlying stock price
    if option_type == "PUT":
        df["collateral"] = df["strike"]
    else:
        df["collateral"] = underlying_price

    df["CAGR"] = df.apply(
        lambda row: calc_cagr(
            premium=row["midPrice"],
            collateral=row["collateral"],
            dte=row["DTE"],
        ),
        axis=1,
    )

    df["CAGR_%"] = df["CAGR"] * 100

    df["moneyness_%"] = (df["strike"] / underlying_price - 1) * 100

    wanted_cols = [
        "optionType",
        "expiration",
        "DTE",
        "contractSymbol",
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "midPrice",
        "impliedVolatility",
        "IV_%",
        "delta",
        "CAGR_%",
        "moneyness_%",
        "volume",
        "openInterest",
        "inTheMoney",
        "lastTradeDate",
    ]

    existing_cols = [c for c in wanted_cols if c in df.columns]
    return df[existing_cols]


@st.cache_data(ttl=300)
def load_option_chain(ticker: str, load_all_expirations: bool, selected_expiration: str | None):
    """
    Load options from yfinance.
    Cache for 5 minutes.
    """
    ticker = ticker.upper().strip()
    tk = yf.Ticker(ticker)

    underlying_price = get_underlying_price(tk)

    expirations = list(tk.options)

    if not expirations:
        raise ValueError("No option expirations found for this ticker.")

    if load_all_expirations:
        exps_to_load = expirations
    else:
        exps_to_load = [selected_expiration]

    all_dfs = []

    for exp in exps_to_load:
        chain = tk.option_chain(exp)

        calls = prepare_options_df(
            df=chain.calls,
            option_type="CALL",
            expiration=exp,
            underlying_price=underlying_price,
            risk_free_rate=st.session_state["risk_free_rate"],
        )

        puts = prepare_options_df(
            df=chain.puts,
            option_type="PUT",
            expiration=exp,
            underlying_price=underlying_price,
            risk_free_rate=st.session_state["risk_free_rate"],
        )

        all_dfs.append(calls)
        all_dfs.append(puts)

    result = pd.concat(all_dfs, ignore_index=True)

    return underlying_price, expirations, result


# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

risk_free_rate_pct = st.sidebar.number_input(
    "Risk-free rate % for delta",
    min_value=0.0,
    max_value=20.0,
    value=4.5,
    step=0.1,
)

st.session_state["risk_free_rate"] = risk_free_rate_pct / 100

load_all_expirations = st.sidebar.checkbox("Load all expirations", value=False)

option_side_filter = st.sidebar.selectbox(
    "Option side",
    ["ALL", "CALL", "PUT"],
    index=0,
)

min_dte = st.sidebar.number_input("Min DTE", min_value=0, value=0, step=1)
max_dte = st.sidebar.number_input("Max DTE", min_value=0, value=2000, step=1)

min_open_interest = st.sidebar.number_input("Min open interest", min_value=0, value=0, step=10)
min_volume = st.sidebar.number_input("Min volume", min_value=0, value=0, step=10)

only_with_bid_ask = st.sidebar.checkbox("Only options with bid and ask", value=True)

max_rows = st.sidebar.number_input("Max rows to show", min_value=50, max_value=10000, value=1000, step=50)


# -----------------------------
# App Logic
# -----------------------------
if ticker:
    try:
        tk_temp = yf.Ticker(ticker)
        expirations_temp = list(tk_temp.options)

        if not expirations_temp:
            st.error("No option expirations found. Try another ticker.")
            st.stop()

        selected_expiration = st.sidebar.selectbox(
            "Expiration",
            expirations_temp,
            index=0,
            disabled=load_all_expirations,
        )

        if st.sidebar.button("Load option chain"):
            with st.spinner("Loading option chain..."):
                underlying_price, expirations, df = load_option_chain(
                    ticker=ticker,
                    load_all_expirations=load_all_expirations,
                    selected_expiration=selected_expiration,
                )

            st.success(f"Loaded options for {ticker}")
            st.metric("Underlying price", f"{underlying_price:,.2f}")

            # Filters
            filtered = df.copy()

            filtered = filtered[
                (filtered["DTE"] >= min_dte) &
                (filtered["DTE"] <= max_dte)
            ]

            if option_side_filter != "ALL":
                filtered = filtered[filtered["optionType"] == option_side_filter]

            if "openInterest" in filtered.columns:
                filtered = filtered[filtered["openInterest"].fillna(0) >= min_open_interest]

            if "volume" in filtered.columns:
                filtered = filtered[filtered["volume"].fillna(0) >= min_volume]

            if only_with_bid_ask:
                filtered = filtered[
                    (filtered["bid"].fillna(0) > 0) &
                    (filtered["ask"].fillna(0) > 0)
                ]

            # Sorting
            sort_col = st.selectbox(
                "Sort by",
                ["CAGR_%", "DTE", "strike", "delta", "IV_%", "openInterest", "volume"],
                index=0,
            )

            ascending = st.checkbox("Sort ascending", value=False)

            filtered = filtered.sort_values(sort_col, ascending=ascending)

            # Format
            display_df = filtered.head(int(max_rows)).copy()

            numeric_cols = [
                "strike",
                "lastPrice",
                "bid",
                "ask",
                "midPrice",
                "IV_%",
                "delta",
                "CAGR_%",
                "moneyness_%",
            ]

            for col in numeric_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].astype(float).round(4)

            st.subheader("Option Chain")
            st.dataframe(display_df, use_container_width=True)

            csv = display_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{ticker}_options_chain.csv",
                mime="text/csv",
            )

            st.subheader("Explanation")

            st.markdown(
                """
                **Price used:**  
                `midPrice = (bid + ask) / 2` when bid and ask exist.  
                Otherwise the app uses `lastPrice`.

                **Delta:**  
                Approximate Black-Scholes delta calculated from:
                - underlying price
                - strike
                - DTE
                - implied volatility
                - risk-free rate

                **PUT CAGR:**  
                Assumes cash-secured put.

                `CAGR = (1 + premium / strike) ** (365 / DTE) - 1`

                **CALL CAGR:**  
                Assumes covered call.

                `CAGR = (1 + premium / underlying_price) ** (365 / DTE) - 1`
                """
            )

    except Exception as e:
        st.error(f"Error: {e}")
