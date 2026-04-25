import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date


st.set_page_config(
    page_title="RSI + Z-Score Buy Signal App",
    layout="wide"
)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()

    zscore = (series - ma) / std

    return zscore


def add_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    z_window: int = 20,
    ma_short: int = 50,
    ma_long: int = 200,
    use_moving_averages: bool = False
) -> pd.DataFrame:
    df = df.copy()

    df["RSI"] = compute_rsi(df["Close"], period=rsi_period)
    df["ZScore"] = compute_zscore(df["Close"], window=z_window)

    if use_moving_averages:
        df["MA_Short"] = df["Close"].rolling(ma_short).mean()
        df["MA_Long"] = df["Close"].rolling(ma_long).mean()

    return df


def generate_buy_signals(
    df: pd.DataFrame,
    rsi_threshold: float = 30,
    z_threshold: float = -2.0,
    use_trend_filter: bool = False
) -> pd.DataFrame:
    df = df.copy()

    rsi_ok = df["RSI"] < rsi_threshold
    z_ok = df["ZScore"] < z_threshold

    if use_trend_filter and "MA_Short" in df.columns and "MA_Long" in df.columns:
        trend_ok = df["MA_Short"] > df["MA_Long"]
        df["BuySignal"] = trend_ok & rsi_ok & z_ok
    else:
        df["BuySignal"] = rsi_ok & z_ok

    df["BuySignal_First"] = (
        df["BuySignal"]
        & (~df["BuySignal"].shift(1).fillna(False))
    )

    return df


def create_price_chart(
    df: pd.DataFrame,
    ticker: str,
    show_moving_averages: bool = False
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close"
    ))

    if show_moving_averages:
        if "MA_Short" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["MA_Short"],
                mode="lines",
                name="MA Short"
            ))

        if "MA_Long" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["MA_Long"],
                mode="lines",
                name="MA Long"
            ))

    buy_points = df[df["BuySignal_First"]]

    fig.add_trace(go.Scatter(
        x=buy_points.index,
        y=buy_points["Close"],
        mode="markers",
        name="Buy Signal",
        marker=dict(
            symbol="triangle-up",
            size=12,
            color="green"
        )
    ))

    fig.update_layout(
        title=f"{ticker} Price with Buy Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=550,
        hovermode="x unified"
    )

    return fig


def create_rsi_chart(
    df: pd.DataFrame,
    ticker: str,
    rsi_threshold: float
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["RSI"],
        mode="lines",
        name="RSI"
    ))

    fig.add_hline(
        y=rsi_threshold,
        line_dash="dash",
        annotation_text=f"RSI {rsi_threshold}"
    )

    fig.update_layout(
        title=f"{ticker} RSI",
        xaxis_title="Date",
        yaxis_title="RSI",
        template="plotly_white",
        height=350,
        hovermode="x unified"
    )

    return fig


def create_zscore_chart(
    df: pd.DataFrame,
    ticker: str,
    z_threshold: float
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["ZScore"],
        mode="lines",
        name="Z-Score"
    ))

    fig.add_hline(
        y=z_threshold,
        line_dash="dash",
        annotation_text=f"Z = {z_threshold}"
    )

    fig.update_layout(
        title=f"{ticker} Z-Score",
        xaxis_title="Date",
        yaxis_title="Z-Score",
        template="plotly_white",
        height=350,
        hovermode="x unified"
    )

    return fig


@st.cache_data
def load_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


st.title("RSI + Z-Score Buy Signal Dashboard")

with st.sidebar:
    st.header("Settings")

    ticker = st.text_input("Ticker", value="SOXL").upper()

    start_date = st.date_input(
        "Start date",
        value=date(2016, 1, 1)
    )

    end_date = st.date_input(
        "End date",
        value=date(2019, 4, 16)
    )

    st.subheader("Indicators")

    rsi_period = st.number_input(
        "RSI period",
        min_value=2,
        max_value=100,
        value=14,
        step=1
    )

    z_window = st.number_input(
        "Z-Score window",
        min_value=2,
        max_value=200,
        value=20,
        step=1
    )

    rsi_threshold = st.number_input(
        "RSI buy threshold",
        min_value=1.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

    z_threshold = st.number_input(
        "Z-Score buy threshold",
        min_value=-10.0,
        max_value=10.0,
        value=-2.0,
        step=0.1
    )

    st.subheader("Moving Average Filter")

    use_moving_averages = st.checkbox(
        "Show moving averages",
        value=False
    )

    use_trend_filter = st.checkbox(
        "Use trend filter: MA Short > MA Long",
        value=False
    )

    ma_short = st.number_input(
        "Short MA",
        min_value=2,
        max_value=300,
        value=50,
        step=1
    )

    ma_long = st.number_input(
        "Long MA",
        min_value=2,
        max_value=500,
        value=100,
        step=1
    )

    run_button = st.button("Run Analysis")


if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()


if run_button:
    with st.spinner("Downloading data and calculating signals..."):
        df = load_data(ticker, start_date, end_date)

        if df.empty:
            st.error(f"No data downloaded for {ticker}. Check the ticker or date range.")
            st.stop()

        df = add_indicators(
            df,
            rsi_period=rsi_period,
            z_window=z_window,
            ma_short=ma_short,
            ma_long=ma_long,
            use_moving_averages=use_moving_averages or use_trend_filter
        )

        df = generate_buy_signals(
            df,
            rsi_threshold=rsi_threshold,
            z_threshold=z_threshold,
            use_trend_filter=use_trend_filter
        )

    buy_signals = df[df["BuySignal_First"]]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Ticker", ticker)
    col2.metric("Rows downloaded", f"{len(df):,}")
    col3.metric("Buy signals", len(buy_signals))

    latest_close = df["Close"].dropna().iloc[-1]
    col4.metric("Latest close", f"{latest_close:.2f}")

    st.plotly_chart(
        create_price_chart(
            df,
            ticker,
            show_moving_averages=use_moving_averages or use_trend_filter
        ),
        use_container_width=True
    )

    st.plotly_chart(
        create_rsi_chart(
            df,
            ticker,
            rsi_threshold=rsi_threshold
        ),
        use_container_width=True
    )

    st.plotly_chart(
        create_zscore_chart(
            df,
            ticker,
            z_threshold=z_threshold
        ),
        use_container_width=True
    )

    st.subheader("Recent Data")

    columns_to_show = [
        "Close",
        "RSI",
        "ZScore",
        "BuySignal",
        "BuySignal_First"
    ]

    if "MA_Short" in df.columns:
        columns_to_show.append("MA_Short")

    if "MA_Long" in df.columns:
        columns_to_show.append("MA_Long")

    st.dataframe(
        df[columns_to_show].tail(100),
        use_container_width=True
    )

    st.subheader("Buy Signal Dates")

    if buy_signals.empty:
        st.info("No buy signals found for the selected settings.")
    else:
        st.dataframe(
            buy_signals[columns_to_show],
            use_container_width=True
        )

        csv = buy_signals[columns_to_show].to_csv().encode("utf-8")

        st.download_button(
            label="Download buy signals as CSV",
            data=csv,
            file_name=f"{ticker}_buy_signals.csv",
            mime="text/csv"
        )

else:
    st.info("Choose your settings in the sidebar, then click Run Analysis.")
