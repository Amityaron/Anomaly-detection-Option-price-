"""
Streamlit app: S&P 500 Equal Weight / GDP valuation model

Formula:
    Index Value = S&P 500 Equal Weight / GDP

Data source:
    Yahoo Finance only, via yfinance

Default tickers:
    S&P 500 Equal Weight: ^SPXEW
    US Annual GDP: GDPCA.I

Run:
    pip install streamlit yfinance pandas numpy plotly
    streamlit run "Index Value.py"
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
    page_title="S&P 500 Equal Weight / GDP Model",
    page_icon="📈",
    layout="wide",
)


@dataclass
class ModelConfig:
    stock_ticker: str
    gdp_ticker: str
    start: dt.date
    end: dt.date
    normalize: bool
    rolling_window: int


@st.cache_data(ttl=60 * 60)
def load_yahoo_close(ticker: str, start: dt.date, end: dt.date, column_name: str) -> pd.DataFrame:
    """
    Download Close price/value from Yahoo Finance using yfinance.
    Works for stock/index tickers and Yahoo economic series tickers.
    """

    df = yf.download(
        ticker,
        start=start,
        end=end + dt.timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(
            f"No Yahoo Finance data returned for ticker: {ticker}. "
            "Try another ticker or a longer date range."
        )

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            raise ValueError(f"Yahoo data for {ticker} does not contain a Close column.")
        close = df["Close"].iloc[:, 0]
    else:
        if "Close" not in df.columns:
            raise ValueError(f"Yahoo data for {ticker} does not contain a Close column.")
        close = df["Close"]

    out = close.rename(column_name).to_frame()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.dropna()

    if out.empty:
        raise ValueError(f"Yahoo Finance returned only empty values for ticker: {ticker}")

    return out


@st.cache_data(ttl=60 * 60)
def load_stock_index_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    return load_yahoo_close(
        ticker=ticker,
        start=start,
        end=end,
        column_name="stock_index",
    )


@st.cache_data(ttl=60 * 60 * 6)
def load_gdp_data_from_yahoo(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Load GDP from Yahoo Finance.

    Default:
        GDPCA.I = US Annual GDP

    Note:
        GDP data from Yahoo is usually annual, not daily.
        The model forward-fills GDP values to stock-market dates.
    """

    gdp = load_yahoo_close(
        ticker=ticker,
        start=start,
        end=end,
        column_name="gdp",
    )

    return gdp


def build_model(config: ModelConfig) -> pd.DataFrame:
    stock = load_stock_index_data(config.stock_ticker, config.start, config.end)
    gdp = load_gdp_data_from_yahoo(config.gdp_ticker, config.start, config.end)

    # GDP is annual/low-frequency; align it to stock dates by forward filling.
    df = stock.join(gdp, how="left")
    df["gdp"] = df["gdp"].ffill()
    df = df.dropna(subset=["stock_index", "gdp"])

    if df.empty:
        raise ValueError(
            "No overlapping stock-index and GDP data after alignment. "
            "Try an earlier start date, for example 2003-01-01."
        )

    if config.normalize:
        # Recommended because stock-index points and GDP dollars are different units.
        # The formula remains stock_index / GDP, but both inputs are rebased to 100.
        df["stock_index_used"] = df["stock_index"] / df["stock_index"].iloc[0] * 100
        df["gdp_used"] = df["gdp"] / df["gdp"].iloc[0] * 100
    else:
        df["stock_index_used"] = df["stock_index"]
        df["gdp_used"] = df["gdp"]

    # Main formula
    df["index_value"] = df["stock_index_used"] / df["gdp_used"]

    # Expanding historical statistics
    df["mean"] = df["index_value"].expanding(min_periods=20).mean()
    df["sd"] = df["index_value"].expanding(min_periods=20).std(ddof=1)
    df["z_score"] = (df["index_value"] - df["mean"]) / df["sd"]

    # Rolling statistics
    w = max(20, int(config.rolling_window))
    min_periods = max(10, w // 4)

    df["rolling_mean"] = df["index_value"].rolling(w, min_periods=min_periods).mean()
    df["rolling_sd"] = df["index_value"].rolling(w, min_periods=min_periods).std(ddof=1)
    df["rolling_z_score"] = (df["index_value"] - df["rolling_mean"]) / df["rolling_sd"]

    # SD bands
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

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["index_value"],
            name="Index Value",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[mean_col],
            name="Mean",
            line=dict(width=2, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[upper1],
            name="+1 SD",
            line=dict(width=1, dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[upper2],
            name="+2 SD",
            line=dict(width=1, dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[lower1],
            name="-1 SD",
            line=dict(width=1, dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[lower2],
            name="-2 SD",
            line=dict(width=1, dash="dot"),
        )
    )

    fig.update_layout(
        title="Index Value = S&P 500 Equal Weight / GDP, with SD bands",
        xaxis_title="Date",
        yaxis_title="Index Value",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


def component_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["stock_index_used"],
            name="S&P 500 Equal Weight used",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["gdp_used"],
            name="GDP used",
        )
    )

    fig.update_layout(
        title="Inputs Used in the Formula",
        xaxis_title="Date",
        yaxis_title="Level",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


def raw_input_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["stock_index"],
            name="Raw S&P 500 Equal Weight",
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["gdp"],
            name="Raw GDP",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Raw Yahoo Finance Inputs",
        xaxis_title="Date",
        yaxis=dict(title="S&P 500 Equal Weight"),
        yaxis2=dict(
            title="GDP",
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


st.title("📈 S&P 500 Equal Weight / GDP Valuation Model")

st.markdown(
    """
This app uses **Yahoo Finance only**.

Formula:

**Index Value = S&P 500 Equal Weight / GDP**

Then the app calculates:

- Historical mean
- Standard deviation
- ±1 SD and ±2 SD bands
- Z-score
- Rolling z-score
"""
)

with st.sidebar:
    st.header("Settings")

    stock_ticker = st.text_input(
        "S&P 500 Equal Weight Yahoo ticker",
        value="^SPXEW",
        help="Yahoo Finance ticker for S&P 500 Equal Weighted. If it fails, try RSP as ETF proxy.",
    )

    gdp_ticker = st.text_input(
        "GDP Yahoo ticker",
        value="GDPCA.I",
        help="Yahoo Finance ticker for US Annual GDP.",
    )

    today = dt.date.today()

    start = st.date_input(
        "Start date",
        value=dt.date(2003, 1, 1),
        min_value=dt.date(1950, 1, 1),
        max_value=today,
    )

    end = st.date_input(
        "End date",
        value=today,
        min_value=dt.date(1950, 1, 1),
        max_value=today,
    )

    normalize = st.checkbox(
        "Normalize both inputs to 100 at start",
        value=True,
        help=(
            "Recommended. S&P index points and GDP dollars are different units. "
            "This keeps the formula but compares both series on the same base."
        ),
    )

    rolling_window = st.slider(
        "Rolling SD window, trading days",
        min_value=60,
        max_value=1260,
        value=756,
        step=21,
        help="756 trading days is about 3 years.",
    )

    band_mode = st.radio(
        "SD band mode",
        ["Rolling SD bands", "Full-sample expanding SD bands"],
        index=0,
    )

    st.divider()

    st.caption(
        "Default Yahoo tickers: ^SPXEW for S&P 500 Equal Weight and GDPCA.I for US Annual GDP."
    )


if start >= end:
    st.error("Start date must be before end date.")
    st.stop()


config = ModelConfig(
    stock_ticker=stock_ticker.strip(),
    gdp_ticker=gdp_ticker.strip(),
    start=start,
    end=end,
    normalize=normalize,
    rolling_window=rolling_window,
)


try:
    df = build_model(config)
except Exception as e:
    st.error(str(e))

    st.info(
        """
Troubleshooting:

1. Try changing the S&P 500 Equal Weight ticker from `^SPXEW` to `RSP`.
2. Keep GDP ticker as `GDPCA.I`.
3. Try an earlier start date, such as `2003-01-01`.
4. Yahoo Finance sometimes blocks or delays data requests. Refresh and try again.
"""
    )

    st.stop()


latest = df.dropna(subset=["index_value"]).iloc[-1]

if band_mode == "Rolling SD bands":
    latest_z = latest["rolling_z_score"]
    latest_sd = latest["rolling_sd"]
    latest_mean = latest["rolling_mean"]
else:
    latest_z = latest["z_score"]
    latest_sd = latest["sd"]
    latest_mean = latest["mean"]


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Index Value",
    f"{latest['index_value']:.4f}",
)

col2.metric(
    "Mean",
    f"{latest_mean:.4f}" if pd.notna(latest_mean) else "N/A",
)

col3.metric(
    "Standard Deviation",
    f"{latest_sd:.4f}" if pd.notna(latest_sd) else "N/A",
)

col4.metric(
    "Z-score",
    f"{latest_z:.2f}" if pd.notna(latest_z) else "N/A",
)


st.subheader(status_from_z(float(latest_z)) if pd.notna(latest_z) else "Not enough data")

st.plotly_chart(
    valuation_chart(df, band_mode),
    use_container_width=True,
)

st.plotly_chart(
    component_chart(df),
    use_container_width=True,
)

with st.expander("Show raw Yahoo Finance inputs"):
    st.plotly_chart(
        raw_input_chart(df),
        use_container_width=True,
    )


st.subheader("Latest data")

display_cols = [
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
]

st.dataframe(
    df[display_cols].tail(30).sort_index(ascending=False),
    use_container_width=True,
)


csv = df.to_csv(index=True).encode("utf-8")

st.download_button(
    "Download model data as CSV",
    data=csv,
    file_name=f"{stock_ticker.replace('^', '')}_{gdp_ticker}_index_value_model.csv",
    mime="text/csv",
)


st.caption(
    "Data source: Yahoo Finance via yfinance. "
    "GDPCA.I is annual GDP, so the app forward-fills the latest GDP value to stock-market dates. "
    "This is a valuation indicator, not investment advice."
)
