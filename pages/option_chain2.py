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
    Returned as percent.
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

    # Call period return by LAST CALL PRICE
    # Capital base = stock price
    calls["Call Period Return %"] = calls.apply(
        lambda row: (row["lastPrice"] / stock_price) * 100
        if not pd.isna(row["lastPrice"]) and row["lastPrice"] >= 0 and stock_price > 0
        else np.nan,
        axis=1
    )

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
        "Call Period Return %",
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

    # Put period return by LAST PUT PRICE
    # Period Return = Put Last Price / Strike
    puts["Put Period Return %"] = puts.apply(
        lambda row: (row["lastPrice"] / row["strike"]) * 100
        if not pd.isna(row["lastPrice"]) and row["lastPrice"] >= 0 and row["strike"] > 0
        else np.nan,
        axis=1
    )

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
        "Put Period Return %",
        "Put CAGR %",
        "Put ITM"
    ]

    return puts[keep_cols]


def style_itm(row):
    styles = [""] * len(row)

    call_itm = bool(row.get("Call ITM", False)) if not pd.isna(row.get("Call ITM", np.nan)) else False
    put_itm = bool(row.get("Put ITM", False)) if not pd.isna(row.get("Put ITM", np.nan)) else False

    for i, col in enumerate(row.index):
        if col.startswith("Call") and call_itm:
            styles[i] = "background-color: #264b70; color: white;"

        if col.startswith("Put") and put_itm:
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

filter_near_money = st.sidebar.checkbox(
    "Show only strikes near stock price",
    value=True
)

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
                "Call Period Return %",
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
                "Put Period Return %",
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
                "CAGR here uses lastPrice only. LastPrice can be stale. "
                "For ITM options, lastPrice includes intrinsic value, so CAGR can be misleading."
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
                ### Table formulas used

                **Put Period Return %**

                `Put Period Return % = Put Last Price / Strike * 100`

                **Put CAGR by last put price**

                `Put CAGR % = ((1 + Put Last Price / Strike) ** (365 / DTE) - 1) * 100`

                **Call Period Return %**

                `Call Period Return % = Call Last Price / Stock Price * 100`

                **Call CAGR by last call price**

                `Call CAGR % = ((1 + Call Last Price / Stock Price) ** (365 / DTE) - 1) * 100`
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

# =====================================================
# Manual Put Ladder Calculator
# =====================================================
st.divider()
st.subheader("🪜 Manual Put Ladder Calculator")

st.write(
    "Enter total capital, DTE, strikes and premiums. "
    "Choose allocation structure and the app will automatically show "
    "3 legs or 4 legs."
)

# IMPORTANT:
# This selectbox must be OUTSIDE the form,
# so Streamlit redraws 3 or 4 legs immediately when you change it.
ladder_structure = st.selectbox(
    "Allocation Structure",
    options=["10/20/30/40", "15/30/55"],
    key="ladder_structure_select"
)

if ladder_structure == "10/20/30/40":
    default_allocations = [10.0, 20.0, 30.0, 40.0]
    default_strikes = [90.0, 70.0, 60.0, 50.0]
    default_premiums = [0.60, 0.55, 0.50, 0.45]
else:
    default_allocations = [15.0, 30.0, 55.0]
    default_strikes = [90.0, 70.0, 50.0]
    default_premiums = [0.60, 0.50, 0.40]

with st.form("manual_put_ladder_form"):
    top_col1, top_col2 = st.columns(2)

    ladder_total_capital = top_col1.number_input(
        "Total Capital / Collateral",
        min_value=0.0,
        value=122000.0,
        step=1000.0,
        placeholder="Example: 100000"
    )

    ladder_dte = top_col2.number_input(
        "DTE",
        min_value=1,
        value=30,
        step=1,
        placeholder="Example: 30"
    )

    st.markdown("### Enter Ladder Legs")

    header_cols = st.columns([1, 1, 1, 1])
    header_cols[0].markdown("**Leg**")
    header_cols[1].markdown("**Allocation %**")
    header_cols[2].markdown("**Strike**")
    header_cols[3].markdown("**Premium**")

    ladder_rows = []

    for i, allocation in enumerate(default_allocations):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

        c1.write(f"Leg {i + 1}")

        allocation_pct = c2.number_input(
            f"Allocation % {i + 1}",
            min_value=0.0,
            max_value=100.0,
            value=allocation,
            step=1.0,
            key=f"{ladder_structure}_allocation_pct_{i}"
        )

        strike = c3.number_input(
            f"Strike {i + 1}",
            min_value=0.0,
            value=default_strikes[i],
            step=0.5,
            key=f"{ladder_structure}_strike_{i}"
        )

        premium = c4.number_input(
            f"Premium {i + 1}",
            min_value=0.0,
            value=default_premiums[i],
            step=0.01,
            key=f"{ladder_structure}_premium_{i}"
        )

        ladder_rows.append({
            "Leg": i + 1,
            "Allocation %": allocation_pct,
            "Strike": strike,
            "Premium": premium
        })

    calculate_ladder = st.form_submit_button("Calculate Put Ladder")


if calculate_ladder:
    ladder_df = pd.DataFrame(ladder_rows)

    allocation_sum = ladder_df["Allocation %"].sum()

    if ladder_total_capital <= 0:
        st.error("Total Capital / Collateral must be greater than 0.")

    elif ladder_dte <= 0:
        st.error("DTE must be greater than 0.")

    elif abs(allocation_sum - 100) > 0.01:
        st.error(f"Allocation must sum to 100%. Current sum: {allocation_sum:.2f}%")

    elif (ladder_df["Strike"] <= 0).any():
        st.error("All strikes must be greater than 0.")

    else:
        # Allocation
        ladder_df["Allocation Weight"] = ladder_df["Allocation %"] / 100
        ladder_df["Target Capital"] = ladder_total_capital * ladder_df["Allocation Weight"]

        # Cash-secured put collateral
        ladder_df["Collateral Per Contract"] = ladder_df["Strike"] * 100

        # Contracts per strike
        ladder_df["Contracts"] = np.floor(
            ladder_df["Target Capital"] / ladder_df["Collateral Per Contract"]
        ).astype(int)

        # Actual capital used after contract rounding
        ladder_df["Actual Collateral"] = (
            ladder_df["Contracts"] * ladder_df["Collateral Per Contract"]
        )

        ladder_df["Actual Weight %"] = np.where(
            ladder_total_capital > 0,
            ladder_df["Actual Collateral"] / ladder_total_capital * 100,
            np.nan
        )

        ladder_df["Unused Capital"] = (
            ladder_df["Target Capital"] - ladder_df["Actual Collateral"]
        )

        # Premium received
        ladder_df["Premium Cash"] = (
            ladder_df["Contracts"] * ladder_df["Premium"] * 100
        )

        # Return per leg
        ladder_df["Period Return %"] = np.where(
            ladder_df["Actual Collateral"] > 0,
            ladder_df["Premium Cash"] / ladder_df["Actual Collateral"] * 100,
            np.nan
        )

        ladder_df["Simple Annualized %"] = (
            ladder_df["Period Return %"] * 365 / ladder_dte
        )

        ladder_df["CAGR %"] = np.where(
            ladder_df["Period Return %"].notna(),
            ((1 + ladder_df["Period Return %"] / 100) ** (365 / ladder_dte) - 1) * 100,
            np.nan
        )

        # Assignment calculations
        ladder_df["Net Assignment Price"] = (
            ladder_df["Strike"] - ladder_df["Premium"]
        )

        ladder_df["Shares If Assigned"] = ladder_df["Contracts"] * 100

        # Basket calculations
        total_actual_collateral = ladder_df["Actual Collateral"].sum()
        total_premium_cash = ladder_df["Premium Cash"].sum()
        total_unused_capital = ladder_total_capital - total_actual_collateral
        total_contracts = ladder_df["Contracts"].sum()
        total_shares_if_assigned = ladder_df["Shares If Assigned"].sum()

        if total_actual_collateral > 0:
            weighted_period_return = total_premium_cash / total_actual_collateral
            weighted_simple_annualized = weighted_period_return * 365 / ladder_dte
            weighted_cagr = ((1 + weighted_period_return) ** (365 / ladder_dte) - 1)
        else:
            weighted_period_return = np.nan
            weighted_simple_annualized = np.nan
            weighted_cagr = np.nan

        if total_shares_if_assigned > 0:
            avg_assignment_price = (
                ladder_df["Shares If Assigned"] * ladder_df["Net Assignment Price"]
            ).sum() / total_shares_if_assigned
        else:
            avg_assignment_price = np.nan

        st.markdown("### Ladder Summary")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Contracts", f"{total_contracts:,}")
        s2.metric("Total Premium", f"${total_premium_cash:,.2f}")
        s3.metric("Used Collateral", f"${total_actual_collateral:,.2f}")
        s4.metric("Unused Capital", f"${total_unused_capital:,.2f}")

        s5, s6, s7, s8 = st.columns(4)
        s5.metric("Weighted Period Return", f"{weighted_period_return * 100:.2f}%")
        s6.metric("Weighted Simple Annualized", f"{weighted_simple_annualized * 100:.2f}%")
        s7.metric("Weighted CAGR", f"{weighted_cagr * 100:.2f}%")
        s8.metric("Avg Assignment Price", f"${avg_assignment_price:,.2f}")

        st.markdown("### Ladder Details")

        display_cols = [
            "Leg",
            "Allocation %",
            "Strike",
            "Premium",
            "Target Capital",
            "Contracts",
            "Actual Collateral",
            "Actual Weight %",
            "Unused Capital",
            "Premium Cash",
            "Period Return %",
            "Simple Annualized %",
            "CAGR %",
            "Net Assignment Price",
            "Shares If Assigned"
        ]

        ladder_display = ladder_df[display_cols].copy()

        for col in ladder_display.columns:
            if pd.api.types.is_numeric_dtype(ladder_display[col]):
                ladder_display[col] = ladder_display[col].round(2)

        st.dataframe(
            ladder_display,
            use_container_width=True,
            height=350
        )

        ladder_csv = ladder_display.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Ladder CSV",
            data=ladder_csv,
            file_name="put_ladder_calculator.csv",
            mime="text/csv"
        )

        st.markdown(
            """
            ### Formulas

            **Contracts**

            `Contracts = floor(Target Capital / (Strike * 100))`

            **Premium Cash**

            `Premium Cash = Contracts * Premium * 100`

            **Period Return per leg**

            `Period Return % = Premium Cash / Actual Collateral * 100`

            **Weighted Period Return**

            `Weighted Period Return = Total Premium Cash / Total Actual Collateral`

            **CAGR**

            `CAGR = ((1 + Period Return) ** (365 / DTE) - 1) * 100`

            **Net Assignment Price**

            `Net Assignment Price = Strike - Premium`

            **Average Assignment Price**

            `Avg Assignment Price = sum(Shares If Assigned * Net Assignment Price) / sum(Shares If Assigned)`
            """
        )
