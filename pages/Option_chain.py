# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import date

st.set_page_config(page_title="Options Chain", layout="wide")

st.title("Options Chain - Yahoo Style")


def get_price(ticker_obj):
    try:
        return float(ticker_obj.fast_info["last_price"])
    except Exception:
        hist = ticker_obj.history(period="5d")
        return float(hist["Close"].iloc[-1])


def bs_delta(option_type, S, K, T, r, iv):
    if iv <= 0 or T <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))

    if option_type == "CALL":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def cagr(premium, capital, dte):
    if premium <= 0 or capital <= 0 or dte <= 0:
        return np.nan

    return ((1 + premium / capital) ** (365 / dte) - 1) * 100


def add_calculations(df, side, stock_price, dte, r):
    df = df.copy()

    T = dte / 365

    df["Mid"] = np.where(
        (df["bid"] > 0) & (df["ask"] > 0),
        (df["bid"] + df["ask"]) / 2,
        df["lastPrice"]
    )

    df["IV %"] = df["impliedVolatility"] * 100

    df["Delta"] = df.apply(
        lambda x: bs_delta(
            option_type=side,
            S=stock_price,
            K=x["strike"],
            T=T,
            r=r,
            iv=x["impliedVolatility"]
        ),
        axis=1
    )

    if side == "CALL":
        df["ITM"] = stock_price > df["strike"]
        capital = stock_price
    else:
        df["ITM"] = stock_price < df["strike"]
        capital = df["strike"]

    df["CAGR %"] = df.apply(
        lambda x: cagr(
            premium=x["Mid"],
            capital=capital if side == "CALL" else x["strike"],
            dte=dte
        ),
        axis=1
    )

    return df


ticker = st.text_input("Enter ticker", value="AAPL").upper()

risk_free_rate = st.number_input(
    "Risk free rate %",
    value=4.5,
    step=0.1
) / 100

if ticker:
    tk = yf.Ticker(ticker)
    expirations = list(tk.options)

    if len(expirations) == 0:
        st.error("No options found for this ticker.")
        st.stop()

    expiration = st.selectbox("Expiration date", expirations)

    if st.button("Load options"):
        stock_price = get_price(tk)

        exp_date = pd.to_datetime(expiration).date()
        dte = max((exp_date - date.today()).days, 1)

        chain = tk.option_chain(expiration)

        calls = add_calculations(
            chain.calls,
            side="CALL",
            stock_price=stock_price,
            dte=dte,
            r=risk_free_rate
        )

        puts = add_calculations(
            chain.puts,
            side="PUT",
            stock_price=stock_price,
            dte=dte,
            r=risk_free_rate
        )

        st.subheader(f"{ticker} price: {stock_price:.2f}")
        st.write(f"Expiration: {expiration} | DTE: {dte}")

        calls_small = calls[
            [
                "strike",
                "lastPrice",
                "bid",
                "ask",
                "volume",
                "openInterest",
                "IV %",
                "Delta",
                "CAGR %",
                "ITM"
            ]
        ].copy()

        puts_small = puts[
            [
                "strike",
                "lastPrice",
                "bid",
                "ask",
                "volume",
                "openInterest",
                "IV %",
                "Delta",
                "CAGR %",
                "ITM"
            ]
        ].copy()

        calls_small = calls_small.rename(columns={
            "lastPrice": "Call Last",
            "bid": "Call Bid",
            "ask": "Call Ask",
            "volume": "Call Volume",
            "openInterest": "Call OI",
            "IV %": "Call IV %",
            "Delta": "Call Delta",
            "CAGR %": "Call CAGR %",
            "ITM": "Call ITM"
        })

        puts_small = puts_small.rename(columns={
            "lastPrice": "Put Last",
            "bid": "Put Bid",
            "ask": "Put Ask",
            "volume": "Put Volume",
            "openInterest": "Put OI",
            "IV %": "Put IV %",
            "Delta": "Put Delta",
            "CAGR %": "Put CAGR %",
            "ITM": "Put ITM"
        })

        table = pd.merge(
            calls_small,
            puts_small,
            on="strike",
            how="outer"
        ).sort_values("strike")

        table = table.rename(columns={"strike": "Strike"})

        # Round numbers
        for col in table.columns:
            if table[col].dtype in ["float64", "float32"]:
                table[col] = table[col].round(2)

        # Put strike in the middle like Yahoo
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

        def highlight_itm(row):
            styles = [""] * len(row)

            if row["Call ITM"] is True:
                for i, col in enumerate(row.index):
                    if col.startswith("Call"):
                        styles[i] = "background-color: #264b70"

            if row["Put ITM"] is True:
                for i, col in enumerate(row.index):
                    if col.startswith("Put"):
                        styles[i] = "background-color: #264b70"

            return styles

        styled_table = table.style.apply(highlight_itm, axis=1)

        st.dataframe(
            styled_table,
            use_container_width=True,
            height=800
        )

        csv = table.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"{ticker}_{expiration}_options.csv",
            mime="text/csv"
        )
