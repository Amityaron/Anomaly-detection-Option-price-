"""
Anomaly Detection Stock Market App
----------------------------------
Streamlit app for:
1. 22-day Diff Z-Score buy/sell signals.
2. Monthly return distribution and probability table.

Important deployment fixes:
- Uses Plotly only, not matplotlib/seaborn, to avoid Streamlit Cloud segmentation faults.
- Replaces deprecated use_container_width=True with width="stretch" where supported.
- Rounds only numeric columns before display.
- Uses cached yfinance downloads.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =============================================================================
# Helpers
# =============================================================================

st.set_page_config(
    page_title="Anomaly Detection Stock Market App",
    layout="wide",
)


def show_plotly(fig: go.Figure) -> None:
    """
    Show a Plotly chart using the new Streamlit width API.
    Falls back for older Streamlit versions.
    """
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_dataframe(df: pd.DataFrame, *, hide_index: bool = True) -> None:
    """
    Show a dataframe using the new Streamlit width API.
    Falls back for older Streamlit versions.
    """
    try:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    except TypeError:
        st.dataframe(df, use_container_width=True)


def round_numeric_columns(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    """
    Round numeric columns only.
    This avoids warnings when the dataframe also contains dates/times.
    """
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include="number").columns
    df_display[numeric_cols] = df_display[numeric_cols].round(digits)
    return df_display


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance may return MultiIndex columns.
    For one ticker, flatten to standard columns like Close, Open, High, Low.
    """
    if isinstance(df.columns, pd.MultiIndex):
        level_0 = df.columns.get_level_values(0)
        level_1 = df.columns.get_level_values(1)

        if "Close" in level_0:
            df.columns = level_0
        elif "Close" in level_1:
            df.columns = level_1
        else:
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns]

    return df


@st.cache_data(show_spinner=False, ttl=60 * 30)
def download_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Download price data from Yahoo Finance.

    Notes:
    - progress=False prevents noisy console output.
    - threads=False is often more stable on Streamlit Cloud.
    - auto_adjust=False preserves the regular Close column behavior.
    """
    data = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        threads=False,
        auto_adjust=False,
    )

    data = flatten_yfinance_columns(data)
    data = data.sort_index()
    return data


def add_one_day(d: date) -> date:
    """
    yfinance treats end date as exclusive.
    Adding one day makes the selected end date included.
    """
    return d + timedelta(days=1)


def validate_date_range(start: date, end: date) -> None:
    if start >= end:
        st.error("End Date must be after Start Date.")
        st.stop()



def calculate_monthly_returns(close: pd.Series) -> pd.Series:
    """
    Calculate month-end percentage returns.
    Uses "ME" for modern pandas and falls back to "M" for older pandas.
    """
    try:
        return close.resample("ME").last().pct_change() * 100
    except ValueError:
        return close.resample("M").last().pct_change() * 100


# =============================================================================
# App title
# =============================================================================

st.title("Anomaly Detection Stock Market App")
st.write("Welcome to my anomaly detection app.")


# =============================================================================
# Section 1: 22-day Diff Z-Score signals
# =============================================================================

st.header("Diff Z-Score 22 Trading Days")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    signal_symbol = st.text_input(
        "Enter a ticker based on Yahoo Finance:",
        value="SPY",
        key="signal_symbol",
    ).strip().upper()

with col2:
    start_date = st.date_input(
        "Start Date",
        value=pd.to_datetime("2025-01-01").date(),
        key="signal_start_date",
    )

with col3:
    end_date = st.date_input(
        "End Date",
        value=pd.Timestamp.today().date(),
        key="signal_end_date",
    )

validate_date_range(start_date, end_date)

if not signal_symbol:
    st.error("Please enter a ticker symbol.")
    st.stop()

with st.spinner(f"Downloading data for {signal_symbol}..."):
    signal_data = download_prices(
        signal_symbol,
        start_date.isoformat(),
        add_one_day(end_date).isoformat(),
    )

if signal_data.empty:
    st.error("No data was downloaded. Please check the ticker or date range.")
    st.stop()

if "Close" not in signal_data.columns:
    st.error("Close column was not found in the downloaded data.")
    st.write("Available columns:", list(signal_data.columns))
    st.stop()

# Work on a copy to avoid cached-data mutation issues
spy = signal_data.copy()

n = 22

if len(spy) < n * 2:
    st.warning(
        f"Not enough rows for a stable {n}-day Diff Z-Score calculation. "
        f"Downloaded rows: {len(spy)}. Please choose a longer date range."
    )
    st.stop()

# Calculate 22-day difference:
# Diff = Close today - Close 22 trading days ago
spy["Diff_22"] = spy["Close"] - spy["Close"].shift(n)

# Calculate rolling mean and rolling standard deviation of Diff_22
spy["Diff_22_Mean"] = spy["Diff_22"].rolling(n).mean()
spy["Diff_22_STD"] = spy["Diff_22"].rolling(n).std()

# Avoid division by zero
diff_std = spy["Diff_22_STD"].where(spy["Diff_22_STD"] != 0)
spy["Diff_Z_Score_22"] = (
    (spy["Diff_22"] - spy["Diff_22_Mean"]) / diff_std
)

# Remove rows with NaN values from the signal calculation
spy = spy.dropna(subset=["Close", "Diff_Z_Score_22"]).copy()

if spy.empty:
    st.warning(
        "After calculating Diff Z-Score there are no valid rows. "
        "Please choose a longer date range."
    )
    st.stop()

buy_threshold = st.number_input(
    "Buy signal threshold: Diff Z-Score below",
    value=-2.5,
    step=0.1,
)

sell_threshold = st.number_input(
    "Sell signal threshold: Diff Z-Score above",
    value=2.5,
    step=0.1,
)

# Generate buy signal:
# Signal = first day where Diff Z-Score crosses into lower threshold zone
spy["Buy_Zone"] = (spy["Diff_Z_Score_22"] < buy_threshold).astype(int)
spy["Signal"] = spy["Buy_Zone"].diff().fillna(0)
spy.loc[spy["Signal"] < 0, "Signal"] = 0

# Generate sell signal:
# Sell_Signal = first day where Diff Z-Score crosses into upper threshold zone
spy["Sell_Zone"] = (spy["Diff_Z_Score_22"] > sell_threshold).astype(int)
spy["Sell_Signal"] = spy["Sell_Zone"].diff().fillna(0)
spy.loc[spy["Sell_Signal"] < 0, "Sell_Signal"] = 0

# Calculate percentage change from each buy signal to current price
buy_rows = spy.loc[spy["Signal"] == 1].copy()
current_price = float(spy.iloc[-1]["Close"])
latest_z_score = float(spy.iloc[-1]["Diff_Z_Score_22"])

if not buy_rows.empty:
    signal_table = pd.DataFrame(
        {
            "Buy_Signal_Date": buy_rows.index.strftime("%Y-%m-%d"),
            "Buy Price": buy_rows["Close"].astype(float).values,
            "Gain_Pct": ((current_price / buy_rows["Close"].astype(float).values) - 1) * 100,
        }
    )
else:
    signal_table = pd.DataFrame(columns=["Buy_Signal_Date", "Buy Price", "Gain_Pct"])

# Optional Bollinger Bands
add_bollinger = st.checkbox("Add Bollinger Band", value=False)

if add_bollinger:
    bb_col1, bb_col2 = st.columns(2)

    with bb_col1:
        bb_n = st.number_input(
            "Bollinger Band n",
            min_value=1,
            value=20,
            step=1,
        )

    with bb_col2:
        bb_sd = st.number_input(
            "Bollinger Band sd",
            min_value=0.1,
            value=2.0,
            step=0.1,
        )

    spy["BB_Middle"] = spy["Close"].rolling(int(bb_n)).mean()
    spy["BB_STD"] = spy["Close"].rolling(int(bb_n)).std()
    spy["BB_Upper"] = spy["BB_Middle"] + (float(bb_sd) * spy["BB_STD"])
    spy["BB_Lower"] = spy["BB_Middle"] - (float(bb_sd) * spy["BB_STD"])

# Plot Close and signals
title = f"{signal_symbol} ({start_date} to {end_date})"

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=spy.index,
        y=spy["Close"],
        mode="lines",
        name="Close",
    )
)

fig.add_trace(
    go.Scatter(
        x=spy.loc[spy["Signal"] == 1].index,
        y=spy.loc[spy["Signal"] == 1, "Close"],
        mode="markers",
        name="Buy Signal",
        marker=dict(size=10, symbol="circle"),
    )
)

fig.add_trace(
    go.Scatter(
        x=spy.loc[spy["Sell_Signal"] == 1].index,
        y=spy.loc[spy["Sell_Signal"] == 1, "Close"],
        mode="markers",
        name="Sell Signal",
        marker=dict(size=10, symbol="x"),
    )
)

fig.add_hline(
    y=current_price,
    line_dash="dash",
    annotation_text="Current Price",
    annotation_position="top left",
)

if add_bollinger:
    for col_name, trace_name in [
        ("BB_Upper", "BB Upper"),
        ("BB_Middle", "BB Middle"),
        ("BB_Lower", "BB Lower"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=spy.index,
                y=spy[col_name],
                mode="lines",
                name=trace_name,
                line=dict(dash="dot"),
            )
        )

fig.update_layout(
    title=title,
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified",
    height=600,
)

show_plotly(fig)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Current price", f"{current_price:,.2f}")
metric_col2.metric("Latest Diff Z-Score 22 Days", f"{latest_z_score:,.2f}")
metric_col3.metric("Number of buy signals", f"{len(signal_table)}")

st.subheader("Buy Signal Dates, Buy Prices, and Gain Percentages")
show_dataframe(round_numeric_columns(signal_table), hide_index=True)


# =============================================================================
# Section 2: Monthly Percentage Changes
# =============================================================================

st.header("Monthly Percentage Changes")

monthly_col1, monthly_col2 = st.columns([1, 2])

with monthly_col1:
    monthly_symbol = st.text_input(
        "Enter a ticker symbol for monthly analysis:",
        value=signal_symbol or "SPY",
        key="monthly_symbol",
    ).strip().upper()

with monthly_col2:
    monthly_start_date = st.date_input(
        "Monthly analysis start date",
        value=pd.to_datetime("1990-01-01").date(),
        key="monthly_start_date",
    )

monthly_end_date = pd.Timestamp.today().date()
validate_date_range(monthly_start_date, monthly_end_date)

with st.spinner(f"Downloading monthly data for {monthly_symbol}..."):
    sp500_data = download_prices(
        monthly_symbol,
        monthly_start_date.isoformat(),
        add_one_day(monthly_end_date).isoformat(),
    )

if sp500_data.empty:
    st.error("No monthly data was downloaded. Please check the ticker symbol.")
    st.stop()

if "Close" not in sp500_data.columns:
    st.error("Close column was not found in the downloaded monthly data.")
    st.write("Available columns:", list(sp500_data.columns))
    st.stop()

sp500_data = sp500_data.sort_index().copy()

# Monthly returns using month-end closes
monthly_returns = calculate_monthly_returns(sp500_data["Close"]).dropna()

if monthly_returns.empty:
    st.warning("No monthly returns data available to display.")
    st.stop()

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

monthly_returns_df = monthly_returns.to_frame(name="Monthly Return")
monthly_returns_df["Year"] = monthly_returns_df.index.year
monthly_returns_df["Month"] = monthly_returns_df.index.month
monthly_returns_df["Month Name"] = monthly_returns_df.index.strftime("%B")

# Latest available year in the downloaded data
latest_year = int(monthly_returns_df["Year"].max())
last_year_df = monthly_returns_df[monthly_returns_df["Year"] == latest_year].copy()

start_date_actual = sp500_data.index[0]
end_date_actual = sp500_data.index[-1]
total_years = max(1, int((end_date_actual - start_date_actual).days / 365.25))

# Plot monthly return distribution using Plotly only
fig_monthly = go.Figure()

for month_name in month_order:
    month_values = monthly_returns_df.loc[
        monthly_returns_df["Month Name"] == month_name,
        "Monthly Return",
    ].dropna()

    if not month_values.empty:
        fig_monthly.add_trace(
            go.Box(
                y=month_values,
                name=month_name,
                boxpoints=False,
            )
        )

fig_monthly.add_trace(
    go.Scatter(
        x=last_year_df["Month Name"],
        y=last_year_df["Monthly Return"],
        mode="markers",
        name=f"{latest_year} Returns",
        marker=dict(size=10, symbol="circle"),
    )
)

fig_monthly.add_hline(
    y=0,
    line_dash="dash",
    annotation_text="0%",
    annotation_position="top left",
)

fig_monthly.update_layout(
    title=(
        f"Monthly Percentage Changes for {monthly_symbol} "
        f"from {start_date_actual.date()} to {end_date_actual.date()} "
        f"(Total years: {total_years})"
    ),
    xaxis_title="Month",
    yaxis_title="Monthly Return (%)",
    height=650,
    showlegend=True,
    xaxis=dict(
        categoryorder="array",
        categoryarray=month_order,
    ),
)

show_plotly(fig_monthly)

st.write(f"Monthly Percentage Changes for {monthly_symbol}")
st.write(f"Data from {start_date_actual.date()} until {end_date_actual.date()}")
st.write(f"Total number of years observed: {total_years}")

# Probability of positive monthly returns
probabilities_df = (
    monthly_returns_df
    .groupby("Month Name", observed=False)["Monthly Return"]
    .agg(
        Observations="count",
        Positive_Count=lambda s: int((s > 0).sum()),
    )
    .reindex(month_order)
    .reset_index()
)

probabilities_df["Probability (%)"] = (
    probabilities_df["Positive_Count"] / probabilities_df["Observations"] * 100
)

probabilities_df["Probability (%)"] = probabilities_df["Probability (%)"].round(1)
probabilities_df["Observations"] = probabilities_df["Observations"].fillna(0).astype(int)
probabilities_df["Positive_Count"] = probabilities_df["Positive_Count"].fillna(0).astype(int)

st.subheader("Probability of Positive Monthly Returns")
show_dataframe(probabilities_df, hide_index=True)
