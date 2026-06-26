# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import date
from html import escape


# =========================================================
# Streamlit config
# =========================================================
st.set_page_config(
    page_title="Yahoo Style Option Chain",
    layout="wide"
)

st.title("📊 Yahoo Style Options Chain")
st.write("Calls | Strike | Puts with IV, Delta, CAGR and ITM/OTM marking")


# =========================================================
# CSS - Yahoo-like dark table
# =========================================================
st.markdown(
    """
    <style>
    .option-wrapper {
        width: 100%;
        overflow-x: auto;
        border: 1px solid #30363d;
        border-radius: 8px;
    }

    table.option-table {
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        font-size: 14px;
        background-color: #111418;
        color: #d8d8d8;
    }

    .option-table th {
        background-color: #111418;
        color: #e5e5e5;
        padding: 9px 8px;
        border-bottom: 1px solid #30363d;
        text-align: right;
        white-space: nowrap;
        font-weight: 700;
    }

    .option-table td {
        padding: 8px 8px;
        border-bottom: 1px solid #30363d;
        text-align: right;
        white-space: nowrap;
    }

    .option-table .group-header {
        font-size: 26px;
        text-align: center;
        background-color: #111418;
        color: #f1f1f1;
        border-bottom: 1px solid #30363d;
        padding: 12px;
    }

    .option-table .strike-header {
        text-align: center;
        background-color: #111418;
        color: #f1f1f1;
        border-left: 1px solid #30363d;
        border-right: 1px solid #30363d;
    }

    .option-table .strike-cell {
        text-align: center;
        font-weight: 800;
        font-size: 15px;
        background-color: #15191f;
        color: #ffffff;
        border-left: 1px solid #30363d;
        border-right: 1px solid #30363d;
    }

    .call-itm {
        background-color: #29496b;
    }

    .put-itm {
        background-color: #29496b;
    }

    .atm-row .strike-cell {
        background-color: #655300;
        color: #ffffff;
    }

    .badge-itm {
        background-color: #1f6feb;
        color: white;
        padding: 2px 6px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: bold;
    }

    .badge-otm {
        background-color: #30363d;
        color: #d8d8d8;
        padding: 2px 6px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: bold;
    }

    .positive {
        color: #3fb950;
        font-weight: 700;
    }

    .negative {
        color: #f85149;
        font-weight: 700;
    }

    .small-note {
        color: #9ca3af;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Helper functions
# =========================================================
def get_underlying_price(ticker_obj: yf.Ticker) -> float:
    """
    Get latest underlying price.
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


def bs_delta(option_type: str, S: float, K: float, T: float, r: float, iv: float) -> float:
    """
    Black-Scholes delta approximation.

    CALL delta: N(d1)
    PUT delta: N(d1) - 1
    """
    if pd.isna(iv) or iv <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))

    if option_type == "CALL":
        return norm.cdf(d1)

    return norm.cdf(d1) - 1


def calc_cagr(premium: float, capital: float, dte: int) -> float:
    """
    CAGR from selling premium.

    CAGR = (1 + premium / capital) ** (365 / DTE) - 1
    """
    if pd.isna(premium) or premium <= 0 or capital <= 0 or dte <= 0:
        return np.nan

    return (1 + premium / capital) ** (365 / dte) - 1


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fmt_num(x, digits=2):
    x = safe_float(x)
    if pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"


def fmt_int(x):
    x = safe_float(x)
    if pd.isna(x):
        return "-"
    return f"{int(x):,}"


def fmt_pct(x, digits=2):
    x = safe_float(x)
    if pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}%"


def fmt_delta(x):
    x = safe_float(x)
    if pd.isna(x):
        return "-"
    return f"{x:.3f}"


def fmt_badge(is_itm: bool):
    if is_itm:
        return '<span class="badge-itm">ITM</span>'
    return '<span class="badge-otm">OTM</span>'


def choose_premium(row, premium_mode: str):
    """
    For selling options:
    - Bid is conservative
    - Mid is theoretical midpoint
    - Last can be stale
    """
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    last = safe_float(row.get("lastPrice"))

    if premium_mode == "Bid":
        return bid

    if premium_mode == "Mid":
        if not pd.isna(bid) and not pd.isna(ask) and bid > 0 and ask > 0:
            return (bid + ask) / 2
        return last

    return last


def enrich_option_df(
    df: pd.DataFrame,
    option_type: str,
    expiration: str,
    underlying_price: float,
    risk_free_rate: float,
    premium_mode: str
) -> pd.DataFrame:
    """
    Add IV, delta, CAGR, ITM/OTM.
    """
    df = df.copy()

    exp_date = pd.to_datetime(expiration).date()
    today = date.today()
    dte = max((exp_date - today).days, 0)
    T = dte / 365

    df["expiration"] = expiration
    df["DTE"] = dte
    df["optionType"] = option_type

    df["premiumUsed"] = df.apply(lambda row: choose_premium(row, premium_mode), axis=1)

    df["IV_%"] = df["impliedVolatility"] * 100

    df["delta"] = df.apply(
        lambda row: bs_delta(
            option_type=option_type,
            S=underlying_price,
            K=row["strike"],
            T=T,
            r=risk_free_rate,
            iv=row["impliedVolatility"]
        ),
        axis=1
    )

    if option_type == "CALL":
        # Covered call capital base
        df["capitalBase"] = underlying_price
        df["ITM"] = underlying_price > df["strike"]
    else:
        # Cash-secured put capital base
        df["capitalBase"] = df["strike"]
        df["ITM"] = underlying_price < df["strike"]

    df["CAGR_%"] = df.apply(
        lambda row: calc_cagr(
            premium=row["premiumUsed"],
            capital=row["capitalBase"],
            dte=row["DTE"]
        ) * 100,
        axis=1
    )

    return df


@st.cache_data(ttl=300)
def load_chain(
    ticker: str,
    expiration: str,
    risk_free_rate: float,
    premium_mode: str
):
    ticker = ticker.upper().strip()
    tk = yf.Ticker(ticker)

    underlying_price = get_underlying_price(tk)
    chain = tk.option_chain(expiration)

    calls = enrich_option_df(
        df=chain.calls,
        option_type="CALL",
        expiration=expiration,
        underlying_price=underlying_price,
        risk_free_rate=risk_free_rate,
        premium_mode=premium_mode
    )

    puts = enrich_option_df(
        df=chain.puts,
        option_type="PUT",
        expiration=expiration,
        underlying_price=underlying_price,
        risk_free_rate=risk_free_rate,
        premium_mode=premium_mode
    )

    return underlying_price, calls, puts


def build_straddle_html(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    underlying_price: float,
    max_abs_moneyness: float | None = None
) -> str:
    """
    Build Yahoo-style straddle table:
    Calls left | Strike center | Puts right
    """

    calls = calls.copy()
    puts = puts.copy()

    calls = calls.sort_values("strike")
    puts = puts.sort_values("strike")

    # Optional moneyness filter
    if max_abs_moneyness is not None:
        calls = calls[
            abs((calls["strike"] / underlying_price - 1) * 100) <= max_abs_moneyness
        ]
        puts = puts[
            abs((puts["strike"] / underlying_price - 1) * 100) <= max_abs_moneyness
        ]

    call_cols = [
        "lastPrice", "change", "percentChange", "bid", "ask",
        "volume", "openInterest", "IV_%", "delta", "CAGR_%", "ITM"
    ]

    put_cols = [
        "lastPrice", "change", "percentChange", "bid", "ask",
        "volume", "openInterest", "IV_%", "delta", "CAGR_%", "ITM"
    ]

    calls_small = calls[["strike"] + [c for c in call_cols if c in calls.columns]]
    puts_small = puts[["strike"] + [c for c in put_cols if c in puts.columns]]

    merged = pd.merge(
        calls_small,
        puts_small,
        on="strike",
        how="outer",
        suffixes=("_call", "_put")
    ).sort_values("strike")

    # Find closest ATM strike
    merged["atm_distance"] = abs(merged["strike"] - underlying_price)
    atm_strike = merged.loc[merged["atm_distance"].idxmin(), "strike"]
    merged = merged.drop(columns=["atm_distance"])

    html = """
    <div class="option-wrapper">
    <table class="option-table">
        <thead>
            <tr>
                <th class="group-header" colspan="11">Calls</th>
                <th class="group-header strike-header">Strike</th>
                <th class="group-header" colspan="11">Puts</th>
            </tr>
            <tr>
                <th>Last</th>
                <th>Change</th>
                <th>% Chg</th>
                <th>Bid</th>
                <th>Ask</th>
                <th>Vol</th>
                <th>OI</th>
                <th>IV</th>
                <th>Delta</th>
                <th>CAGR</th>
                <th>ITM</th>

                <th class="strike-header">Strike</th>

                <th>Last</th>
                <th>Change</th>
                <th>% Chg</th>
                <th>Bid</th>
                <th>Ask</th>
                <th>Vol</th>
                <th>OI</th>
                <th>IV</th>
                <th>Delta</th>
                <th>CAGR</th>
                <th>ITM</th>
            </tr>
        </thead>
        <tbody>
    """

    for _, row in merged.iterrows():
        strike = safe_float(row["strike"])

        call_itm = bool(row.get("ITM_call")) if not pd.isna(row.get("ITM_call")) else False
        put_itm = bool(row.get("ITM_put")) if not pd.isna(row.get("ITM_put")) else False

        call_class = "call-itm" if call_itm else ""
        put_class = "put-itm" if put_itm else ""
        atm_class = "atm-row" if strike == atm_strike else ""

        call_change = safe_float(row.get("change_call"))
        put_change = safe_float(row.get("change_put"))

        call_change_class = "positive" if call_change > 0 else "negative" if call_change < 0 else ""
        put_change_class = "positive" if put_change > 0 else "negative" if put_change < 0 else ""

        html += f"""
            <tr class="{atm_class}">
                <td class="{call_class}">{fmt_num(row.get("lastPrice_call"))}</td>
                <td class="{call_class} {call_change_class}">{fmt_num(row.get("change_call"))}</td>
                <td class="{call_class}">{fmt_pct(row.get("percentChange_call"))}</td>
                <td class="{call_class}">{fmt_num(row.get("bid_call"))}</td>
                <td class="{call_class}">{fmt_num(row.get("ask_call"))}</td>
                <td class="{call_class}">{fmt_int(row.get("volume_call"))}</td>
                <td class="{call_class}">{fmt_int(row.get("openInterest_call"))}</td>
                <td class="{call_class}">{fmt_pct(row.get("IV_%_call"))}</td>
                <td class="{call_class}">{fmt_delta(row.get("delta_call"))}</td>
                <td class="{call_class}">{fmt_pct(row.get("CAGR_%_call"))}</td>
                <td class="{call_class}">{fmt_badge(call_itm) if not pd.isna(row.get("lastPrice_call")) else "-"}</td>

                <td class="strike-cell">{fmt_num(strike)}</td>

                <td class="{put_class}">{fmt_num(row.get("lastPrice_put"))}</td>
                <td class="{put_class} {put_change_class}">{fmt_num(row.get("change_put"))}</td>
                <td class="{put_class}">{fmt_pct(row.get("percentChange_put"))}</td>
                <td class="{put_class}">{fmt_num(row.get("bid_put"))}</td>
                <td class="{put_class}">{fmt_num(row.get("ask_put"))}</td>
                <td class="{put_class}">{fmt_int(row.get("volume_put"))}</td>
                <td class="{put_class}">{fmt_int(row.get("openInterest_put"))}</td>
                <td class="{put_class}">{fmt_pct(row.get("IV_%_put"))}</td>
                <td class="{put_class}">{fmt_delta(row.get("delta_put"))}</td>
                <td class="{put_class}">{fmt_pct(row.get("CAGR_%_put"))}</td>
                <td class="{put_class}">{fmt_badge(put_itm) if not pd.isna(row.get("lastPrice_put")) else "-"}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    </div>
    """

    return html


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Inputs")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

risk_free_rate_pct = st.sidebar.number_input(
    "Risk-free rate for delta (%)",
    min_value=0.0,
    max_value=20.0,
    value=4.5,
    step=0.1
)

premium_mode = st.sidebar.selectbox(
    "Premium used for CAGR",
    ["Bid", "Mid", "Last"],
    index=0,
    help="For selling options, Bid is more conservative. Mid is theoretical."
)

filter_moneyness = st.sidebar.checkbox("Filter by moneyness", value=False)

max_abs_moneyness = None
if filter_moneyness:
    max_abs_moneyness = st.sidebar.slider(
        "Show strikes within +/- % from stock price",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )

show_raw = st.sidebar.checkbox("Show raw dataframe", value=False)


# =========================================================
# Main app
# =========================================================
if ticker:
    try:
        tk = yf.Ticker(ticker)
        expirations = list(tk.options)

        if not expirations:
            st.error("No options found for this ticker.")
            st.stop()

        expiration = st.sidebar.selectbox("Expiration", expirations)

        load_button = st.sidebar.button("Load option chain")

        if load_button:
            with st.spinner("Loading option chain from yfinance..."):
                underlying_price, calls, puts = load_chain(
                    ticker=ticker,
                    expiration=expiration,
                    risk_free_rate=risk_free_rate_pct / 100,
                    premium_mode=premium_mode
                )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ticker", ticker)
            c2.metric("Underlying Price", f"{underlying_price:,.2f}")
            c3.metric("Expiration", expiration)
            c4.metric("DTE", int(calls["DTE"].iloc[0]))

            st.markdown(
                f"""
                <div class="small-note">
                Blue cells = ITM.  
                Yellow strike = closest ATM strike.  
                CAGR uses premium = <b>{escape(premium_mode)}</b>.
                </div>
                """,
                unsafe_allow_html=True
            )

            html_table = build_straddle_html(
                calls=calls,
                puts=puts,
                underlying_price=underlying_price,
                max_abs_moneyness=max_abs_moneyness
            )

            st.markdown(html_table, unsafe_allow_html=True)

            if show_raw:
                st.subheader("Raw Calls")
                st.dataframe(calls, use_container_width=True)

                st.subheader("Raw Puts")
                st.dataframe(puts, use_container_width=True)

            # CSV export
            calls_export = calls.copy()
            puts_export = puts.copy()

            calls_export["side"] = "CALL"
            puts_export["side"] = "PUT"

            export_df = pd.concat([calls_export, puts_export], ignore_index=True)

            csv = export_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download full options CSV",
                data=csv,
                file_name=f"{ticker}_{expiration}_options.csv",
                mime="text/csv"
            )

            st.markdown(
                """
                ### CAGR logic

                **Put CAGR** assumes cash-secured put:

                `CAGR = (1 + premium / strike) ** (365 / DTE) - 1`

                **Call CAGR** assumes covered call:

                `CAGR = (1 + premium / stock_price) ** (365 / DTE) - 1`

                For naked options, CAGR should be calculated on margin requirement, not strike or stock price.
                """
            )

    except Exception as e:
        st.error(f"Error: {e}")
