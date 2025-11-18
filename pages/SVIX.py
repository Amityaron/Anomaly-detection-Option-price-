import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

st.title("VIX Analysis + SVIX Rolling Z-Score Signals")

# ---------------------------------------------------------------------
# Download VIX Data
# ---------------------------------------------------------------------
st.header("VIX Profit Timing Model")

vix = yf.download("^VIX", start="2000-01-01")[['High', 'Low']].dropna()

ENTRY_LEVEL = 21
holding_periods = []

for i in range(len(vix)):
    high = float(vix['High'].iloc[i])
    if high >= ENTRY_LEVEL:
        entry_price = high
        entry_date = vix.index[i]

        # Look forward for profit (Low < entry price)
        for j in range(i + 1, len(vix)):
            if float(vix['Low'].iloc[j]) < entry_price:
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
    columns=['Entry_Date', 'Exit_Date', 'Days_to_Profit']
)

st.write("### VIX Results")
st.write(f"Number of trades: **{len(df_holding)}**")
st.write(f"Average days to profit: **{df_holding['Days_to_Profit'].mean():.2f}**")
st.write(f"Median days to profit: **{df_holding['Days_to_Profit'].median()}**")
st.write(f"Max days to profit: **{df_holding['Days_to_Profit'].max()}**")
st.write(f"Min days to profit: **{df_holding['Days_to_Profit'].min()}**")

st.dataframe(df_holding)


# ---------------------------------------------------------------------
# SVIX Rolling 20-Day Z-Score Logic
# ---------------------------------------------------------------------
st.header("SVIX Rolling 20-Day Z-Score Strategy")

svix = yf.download("SVIX", start="2020-01-01")[["Close"]].dropna()
svix["Rolling_Mean"] = svix["Close"].rolling(20).mean()
svix["Rolling_Std"] = svix["Close"].rolling(20).std()

# Z-score
svix["Z"] = (svix["Close"] - svix["Rolling_Mean"]) / svix["Rolling_Std"]

# Signals
svix["Signal_-2.5"] = svix["Z"] <= -2.5
svix["Signal_-3"] = svix["Z"] <= -3

# Table of signals
signal_df = svix[(svix["Signal_-2.5"] | svix["Signal_-3"])].copy()
signal_df = signal_df[["Close", "Z", "Signal_-2.5", "Signal_-3"]]

st.write("### SVIX Z-Score Signals (20-Day Rolling)")
st.dataframe(signal_df.tail(50))  # last 50 signals

# Current Z-score
if not svix.empty:
    latest_z = float(svix["Z"].iloc[-1])
    latest_close = float(svix["Close"].iloc[-1])
    st.write(f"**Latest SVIX Price:** {latest_close:.2f}")
    st.write(f"**Latest 20-Day Z-Score:** {latest_z:.2f}")

    if latest_z <= -3:
        st.error("🔥 SVIX Z-Score ≤ -3 → EXTREME Buy Signal")
    elif latest_z <= -2.5:
        st.warning("⚠️ SVIX Z-Score ≤ -2.5 → Strong Buy Signal")
    else:
        st.info("No current Z-score signal.")

# Plot
st.write("### SVIX Price & Z-Score Chart")
plt.figure(figsize=(10,4))
plt.plot(svix.index, svix["Z"])
plt.axhline(-2.5, linestyle='--')
plt.axhline(-3, linestyle='--')
st.pyplot(plt)
