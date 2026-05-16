"""
Streamlit app: Equal-Weight Stock Index / GDP valuation model

Formula:
    Index Value = Stock Index (Equal Weight) / GDP

Default data sources:
- Yahoo Finance via yfinance for the equal-weight stock index proxy
- FRED via pandas-datareader for U.S. GDP

Run:
    pip install -r requirements.txt
    streamlit run streamlit_index_gdp_app.py
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Equal-Weight Stock Index / GDP Model",
    page_icon="📈",
    layout="wide",
)


@dataclass
class ModelConfig:
    ticker: str
    gdp_series: str
    start: dt.date
    end: dt.date
    normalize: bool
    rolling_window: int


@st.cache_data(ttl=60 * 60)
def load_market_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end + dt.timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No market data returned for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]

    out = close.rename("stock_index").to_frame()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.dropna()


@st.cache_data(ttl=60 * 60 * 6)
def load_gdp_data(series: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    gdp = web.DataReader(series, "fred", start, end)
    if gdp.empty:
        raise ValueError(f"No GDP data returned from FRED for series: {series}")

    gdp = gdp.rename(columns={series: "gdp"})
    gdp.index = pd.to_datetime(gdp.index).tz_localize(None)
    return gdp.dropna()


def build_model(config: ModelConfig) -> pd.DataFrame:
    market = load_market_data(config.ticker, config.start, config.end)
    gdp = load_gdp_data(config.gdp_series, config.start, config.end)

    # GDP is quarterly; align it to market dates by forward-filling the latest known GDP value.
    df = market.join(gdp, how="left")
    df["gdp"] = df["gdp"].ffill()
    df = df.dropna(subset=["stock_index", "gdp"])

    if df.empty:
        raise ValueError("No overlapping market and GDP data after alignment.")

    if config.normalize:
        # Keeps the user's formula, but compares normalized index and GDP levels.
        df["stock_index_used"] = df["stock_index"] / df["stock_index"].iloc[0] * 100
        df["gdp_used"] = df["gdp"] / df["gdp"].iloc[0] * 100
    else:
        df["stock_index_used"] = df["stock_index"]
        df["gdp_used"] = df["gdp"]

    df["index_value"] = df["stock_index_used"] / df["gdp_used"]
    df["mean"] = df["index_value"].expanding(min_periods=20).mean()
    df["sd"] = df["index_value"].expanding(min_periods=20).std(ddof=1)
    df["z_score"] = (df["index_value"] - df["mean"]) / df["sd"]

    w = max(20, int(config.rolling_window))
    df["rolling_mean"] = df["index_value"].rolling(w, min_periods=max(10, w // 4)).mean()
    df["rolling_sd"] = df["index_value"].rolling(w, min_periods=max(10, w // 4)).std(ddof=1)
    df["rolling_z_score"] = (df["index_value"] - df["rolling_mean"]) / df["rolling_sd"]

    for k in [1, 2, -1, -2]:
        df[f"mean_{k:+d}sd"] = df["mean"] + k * df["sd"]
        df[f"rolling_mean_{k:+d}sd"] = df["rolling_mean"] + k * df["rolling_sd"]

    return df


def status_from_z(z: float) -> str:
    if np.isnan(z):
        return "Not enough data"
    if z >= 2:
        return "Very expensive vs history"
    if z >= 1:
        return "Expensive vs history"
    if z <= -2:
        return "Very cheap vs history"
    if z <= -1:
        return "Cheap vs history"
    return "Near historical average"


def valuation_chart(df: pd.DataFrame, band_mode: str) -> go.Figure:
    if band_mode == "Rolling SD bands":
        mean_col = "rolling_mean"
        upper1 = "rolling_mean_+1sd"
        upper2 = "rolling_mean_+2sd"
        lower1 = "rolling_mean_-1sd"
        lower2 = "rolling_mean_-2sd"
    else:
        mean_col = "mean"
        upper1 = "mean_+1sd"
        upper2 = "mean_+2sd"
        lower1 = "mean_-1sd"
        lower2 = "mean_-2sd"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["index_value"], name="Index Value", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df[mean_col], name="Mean", line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df[upper1], name="+1 SD", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df[upper2], name="+2 SD", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df[lower1], name="-1 SD", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df[lower2], name="-2 SD", line=dict(width=1, dash="dot")))
    fig.update_layout(
        title="Index Value = Equal-Weight Stock Index / GDP, with standard-deviation bands",
        xaxis_title="Date",
        yaxis_title="Index Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def component_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["stock_index_used"], name="Equal-weight stock index used"))
    fig.add_trace(go.Scatter(x=df.index, y=df["gdp_used"], name="GDP used"))
    fig.update_layout(
        title="Inputs used in the formula",
        xaxis_title="Date",
        yaxis_title="Level",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


st.title("📈 Equal-Weight Stock Index / GDP Valuation Model")
st.markdown("""
This app implements the model:

**Index Value = Stock Index (Equal Weight) / GDP**

Then it calculates the historical mean, standard deviation, and z-score so you can see whether the market is high or low versus its own history.
""")

with st.sidebar:
    st.header("Settings")

    ticker = st.text_input(
        "Equal-weight stock index ticker / proxy",
        value="RSP",
        help="RSP is an ETF proxy for the S&P 500 Equal Weight Index. You can try ^SPXEW if Yahoo returns data in your region.",
    )

    gdp_choice = st.selectbox(
        "GDP series from FRED",
        options={
            "Nominal GDP: GDP": "GDP",
            "Real GDP: GDPC1": "GDPC1",
        },
        index=0,
        help="GDP is quarterly. The app forward-fills each quarterly value to daily market dates.",
    )

    today = dt.date.today()
    start = st.date_input("Start date", value=dt.date(2003, 1, 1), min_value=dt.date(1947, 1, 1), max_value=today)
    end = st.date_input("End date", value=today, min_value=dt.date(1947, 1, 1), max_value=today)

    normalize = st.checkbox(
        "Normalize index and GDP to 100 at start",
        value=True,
        help="Recommended because stock index points and GDP dollars are different units. The formula remains index/GDP, but both inputs are rebased.",
    )

    rolling_window = st.slider("Rolling SD window, trading days", min_value=60, max_value=1260, value=756, step=21)
    band_mode = st.radio("SD band mode", ["Full-sample expanding SD bands", "Rolling SD bands"], index=1)

if start >= end:
    st.error("Start date must be before end date.")
    st.stop()

config = ModelConfig(
    ticker=ticker.strip(),
    gdp_series=gdp_choice,
    start=start,
    end=end,
    normalize=normalize,
    rolling_window=rolling_window,
)

try:
    df = build_model(config)
except Exception as e:
    st.error(str(e))
    st.stop()

latest = df.dropna(subset=["index_value"]).iloc[-1]
latest_z = latest["rolling_z_score"] if band_mode == "Rolling SD bands" else latest["z_score"]
latest_sd = latest["rolling_sd"] if band_mode == "Rolling SD bands" else latest["sd"]
latest_mean = latest["rolling_mean"] if band_mode == "Rolling SD bands" else latest["mean"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Index Value", f"{latest['index_value']:.4f}")
col2.metric("Mean", f"{latest_mean:.4f}" if pd.notna(latest_mean) else "N/A")
col3.metric("Standard Deviation", f"{latest_sd:.4f}" if pd.notna(latest_sd) else "N/A")
col4.metric("Z-score", f"{latest_z:.2f}" if pd.notna(latest_z) else "N/A")

st.subheader(status_from_z(float(latest_z)))
st.plotly_chart(valuation_chart(df, band_mode), use_container_width=True)
st.plotly_chart(component_chart(df), use_container_width=True)

st.subheader("Latest data")
st.dataframe(
    df[[
        "stock_index",
        "gdp",
        "stock_index_used",
        "gdp_used",
        "index_value",
        "mean",
        "sd",
        "z_score",
        "rolling_mean",
        "rolling_sd",
        "rolling_z_score",
    ]].tail(20).sort_index(ascending=False),
    use_container_width=True,
)

csv = df.to_csv(index=True).encode("utf-8")
st.download_button(
    "Download model data as CSV",
    data=csv,
    file_name=f"{ticker}_gdp_valuation_model.csv",
    mime="text/csv",
)

st.caption(
    "Data note: Yahoo Finance market data is pulled through yfinance. GDP data comes from FRED. "
    "This is a valuation indicator, not investment advice. GDP is quarterly and may be revised."
)
