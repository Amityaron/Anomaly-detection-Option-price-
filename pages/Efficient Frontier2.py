import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit setup
st.title("Efficient Frontier Portfolio Optimization")

# Sidebar input
st.sidebar.header("Portfolio Settings")

default_tickers = "IAU,XLK,XLF,IVV,IVW,SMH,SOXX,BTC-USD,IXN,URTH"

ticker_input = st.sidebar.text_area(
    "Enter tickers separated by commas:",
    default_tickers
)

tickers = [
    ticker.strip().upper()
    for ticker in ticker_input.split(",")
    if ticker.strip()
]

# Check if tickers are provided
if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

# Date inputs
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2000-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("2024-12-05"))

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Portfolio settings
num_portfolios = st.sidebar.number_input(
    "Number of simulated portfolios",
    min_value=1000,
    max_value=200000,
    value=100000,
    step=1000
)

risk_free_rate = st.sidebar.number_input(
    "Risk-free rate",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
    format="%.2f"
)

# Download historical data
st.write("Downloading historical price data...")

try:
    downloaded_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if downloaded_data.empty:
        st.error("No data was downloaded. Please check the tickers or date range.")
        st.stop()

    # Handle single ticker and multiple ticker cases
    if len(tickers) == 1:
        data = downloaded_data["Close"].to_frame(name=tickers[0])
    else:
        data = downloaded_data["Close"]

except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

# Drop columns with no data
data = data.dropna(axis=1, how="all")

if data.empty:
    st.error("No usable price data found after removing empty columns.")
    st.stop()

# Warn if some tickers were removed
valid_tickers = list(data.columns)

removed_tickers = [ticker for ticker in tickers if ticker not in valid_tickers]

if removed_tickers:
    st.warning(
        "Some tickers were removed because no price data was found: "
        + ", ".join(removed_tickers)
    )

tickers = valid_tickers

if len(tickers) < 2:
    st.error("At least two valid tickers are required for portfolio optimization.")
    st.stop()

# Calculate daily returns
returns = data.pct_change().dropna()

if returns.empty:
    st.error("Not enough price data to calculate returns.")
    st.stop()

# Calculate annualized mean returns and covariance matrix
annual_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# Portfolio simulation
results = np.zeros((3, int(num_portfolios)))
weights_record = []

for i in range(int(num_portfolios)):
    # Generate random weights
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)

    weights_record.append(weights)

    # Expected portfolio return
    portfolio_return = np.dot(weights, annual_returns)

    # Expected portfolio volatility
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )

    # Sharpe ratio
    if portfolio_volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Store results
    results[0, i] = portfolio_volatility
    results[1, i] = portfolio_return
    results[2, i] = sharpe_ratio

# Convert results to DataFrame
results_df = pd.DataFrame(
    results.T,
    columns=["Volatility", "Return", "Sharpe Ratio"]
)

weights_df = pd.DataFrame(weights_record, columns=tickers)

# Find the portfolio with the maximum Sharpe Ratio
max_sharpe_idx = results_df["Sharpe Ratio"].idxmax()
max_sharpe_portfolio = results_df.loc[max_sharpe_idx]
max_sharpe_weights = weights_df.loc[max_sharpe_idx]

# Sort weights by magnitude
max_sharpe_weights_sorted = max_sharpe_weights.sort_values(ascending=False)

# Find the portfolio with the minimum volatility
min_vol_idx = results_df["Volatility"].idxmin()
min_vol_portfolio = results_df.loc[min_vol_idx]
min_vol_weights = weights_df.loc[min_vol_idx]

# Sort weights by magnitude
min_vol_weights_sorted = min_vol_weights.sort_values(ascending=False)

# Plot Efficient Frontier
st.subheader("Efficient Frontier")

fig, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(
    results_df["Volatility"],
    results_df["Return"],
    c=results_df["Sharpe Ratio"],
    cmap="viridis",
    alpha=0.7
)

fig.colorbar(scatter, ax=ax, label="Sharpe Ratio")

ax.scatter(
    max_sharpe_portfolio["Volatility"],
    max_sharpe_portfolio["Return"],
    c="red",
    marker="*",
    s=250,
    label="Max Sharpe Ratio"
)

ax.scatter(
    min_vol_portfolio["Volatility"],
    min_vol_portfolio["Return"],
    c="blue",
    marker="*",
    s=250,
    label="Min Volatility"
)

ax.set_title("Efficient Frontier")
ax.set_xlabel("Volatility / Standard Deviation")
ax.set_ylabel("Expected Return")
ax.legend()

st.pyplot(fig)

# Display Max Sharpe portfolio
st.subheader("Portfolio with Maximum Sharpe Ratio")

st.write(
    f"Return: {max_sharpe_portfolio['Return'] * 100:.2f}% | "
    f"Volatility: {max_sharpe_portfolio['Volatility'] * 100:.2f}% | "
    f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe Ratio']:.2f}"
)

max_sharpe_display = pd.DataFrame({
    "Ticker": max_sharpe_weights_sorted.index,
    "Weight (%)": max_sharpe_weights_sorted.values * 100
})

st.dataframe(max_sharpe_display, use_container_width=True)

# Display Min Volatility portfolio
st.subheader("Portfolio with Minimum Volatility")

st.write(
    f"Return: {min_vol_portfolio['Return'] * 100:.2f}% | "
    f"Volatility: {min_vol_portfolio['Volatility'] * 100:.2f}% | "
    f"Sharpe Ratio: {min_vol_portfolio['Sharpe Ratio']:.2f}"
)

min_vol_display = pd.DataFrame({
    "Ticker": min_vol_weights_sorted.index,
    "Weight (%)": min_vol_weights_sorted.values * 100
})

st.dataframe(min_vol_display, use_container_width=True)

# Optional raw price data
with st.expander("Show historical price data"):
    st.dataframe(data, use_container_width=True)

# Optional returns data
with st.expander("Show daily returns"):
    st.dataframe(returns, use_container_width=True)
