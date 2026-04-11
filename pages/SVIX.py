import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("VIX Analysis + SVIX Rolling Z-Score Signals")

# ---------------------------------------------------------------------
# USER INPUT FOR VIX ENTRY LEVEL
# ---------------------------------------------------------------------
st.header("VIX Profit Timing Model")

ENTRY_LEVEL = st.number_input(
    "Enter VIX ENTRY LEVEL:",
    min_value=10.0,
    max_value=80.0,
    value=21.0,
    step=0.5
)

# ---------------------------------------------------------------------
# Helper to flatten yfinance columns
# ---------------------------------------------------------------------
def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) and len(col) > 0 else col
            for col in df.columns
        ]
    return df

# ---------------------------------------------------------------------
# Download VIX Data
# ---------------------------------------------------------------------
vix = yf.download("^VIX", start="2000-01-01", auto_adjust=False, progress=False)
vix = flatten_yf_columns(vix)

vix = vix[["High", "Low"]].dropna().copy()
vix["High"] = pd.to_numeric(vix["High"], errors="coerce")
vix["Low"] = pd.to_numeric(vix["Low"], errors="coerce")
vix = vix.dropna()

holding_periods = []

high_series = vix["High"]
low_series = vix["Low"]

for i in range(len(vix)):
    high = high_series.iloc[i]
    if high >= ENTRY_LEVEL:
        entry_price = high
        entry_date = vix.index[i]

        for j in range(i + 1, len(vix)):
            if low_series.iloc[j] < entry_price:
                exit_date = vix.index[j]
                days_to_profit = (exit_date - entry_date).days
                holding_periods.append((entry_date, exit_date, days_to_profit))
                break
        else:
            exit_date = vix.index[-1]
            days_to_profit = (exit_date - entry_date).days
            holding_periods.append((entry_date, exit_date, days_to_profit))

df_holding = pd.DataFrame(
    holding_periods,
    columns=["Entry_Date", "Exit_Date", "Days_to_Profit"]
)

# ---------------------------------------------------------------------
# Show Summary Stats Only
# ---------------------------------------------------------------------
st.write("### VIX Results")

if df_holding.empty:
    st.warning("No trades found for the selected ENTRY_LEVEL.")
else:
    st.write(f"**Number of trades:** {len(df_holding)}")
    st.write(f"**Average days to profit:** {df_holding['Days_to_Profit'].mean():.2f}")
    st.write(f"**SD days to profit:** {df_holding['Days_to_Profit'].std():.2f}")
    st.write(f"**Median days to profit:** {df_holding['Days_to_Profit'].median():.2f}")
    st.write(f"**Max days to profit:** {df_holding['Days_to_Profit'].max()}")
    st.write(f"**Min days to profit:** {df_holding['Days_to_Profit'].min()}")

# ---------------------------------------------------------------------
# SVIX Z-Score Model
# ---------------------------------------------------------------------
st.header("SVIX Rolling 20-Day Z-Score Strategy")

svix = yf.download("SVIX", start="2020-03-01", auto_adjust=False, progress=False)
svix = flatten_yf_columns(svix)

svix = svix[["Close"]].dropna().copy()
svix["Close"] = pd.to_numeric(svix["Close"], errors="coerce")
svix = svix.dropna()

svix["Rolling_Mean"] = svix["Close"].rolling(22).mean()
svix["Rolling_Std"] = svix["Close"].rolling(22).std()
svix["Z"] = (svix["Close"] - svix["Rolling_Mean"]) / svix["Rolling_Std"]

# Signals
svix["Signal_-2.5"] = svix["Z"] <= -2.5
svix["Signal_-3"] = svix["Z"] <= -3

signal_df = svix.loc[
    svix["Signal_-2.5"] | svix["Signal_-3"],
    ["Close", "Z", "Signal_-2.5", "Signal_-3"]
]

st.write("### Latest SVIX Signals")
st.write(signal_df.tail(15))

if svix["Z"].dropna().empty:
    st.warning("Not enough SVIX data yet to calculate the rolling Z-score.")
else:
    latest_z = svix["Z"].dropna().iloc[-1]
    latest_close = svix["Close"].iloc[-1]

    st.write(f"**Latest SVIX Price:** {latest_close:.2f}")
    st.write(f"**Latest 22-Day Z-Score:** {latest_z:.2f}")

    if latest_z <= -3:
        st.error("🔥 SVIX Z-Score ≤ -3 → EXTREME Buy Signal")
    elif latest_z <= -2.5:
        st.warning("⚠️ SVIX Z-Score ≤ -2.5 → Strong Buy Signal")
    else:
        st.info("No current Z-score signal.")

# Chart
st.write("### SVIX Z-Score Chart")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(svix.index, svix["Z"])
ax.axhline(-2.5, linestyle="--")
ax.axhline(-3, linestyle="--")
ax.set_ylabel("Z-Score")
ax.set_title("SVIX 22-Day Rolling Z-Score")
st.pyplot(fig)
