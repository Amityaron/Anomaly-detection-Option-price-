import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit setup
st.title("Efficient Frontier Portfolio Optimization")

# Input for tickers
st.sidebar.header("Portfolio Settings")
default_tickers = "IAU, XLK, XLF, IVV, IVW, SMH, SOXX, XWD.TO, BTC-USD, IXN, XLD.TO"
ticker_input = st.sidebar.text_area("Enter tickers separated by commas:", default_tickers)
tickers = [ticker.strip() for ticker in ticker_input.split(",") if ticker.strip()]

# Check if tickers are provided
if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

# Download historical data
try:
    data = yf.download(tickers, start="2000-01-01", end="2024-12-05")["Adj Close"]
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate annualized mean returns and covariance matrix
annual_returns = returns.mean() * 252  # 252 trading days
cov_matrix = returns.cov() * 252

# Portfolio simulation
num_portfolios = 100000
results = np.zeros((3, num_portfolios))
weights_record = []

# Risk-free rate for Sharpe ratio calculation
risk_free_rate = 0.05

for i in range(num_portfolios):
    # Generate random weights
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    weights_record.append(weights)

    # Expected portfolio return
    portfolio_return = np.dot(weights, annual_returns)

    # Expected portfolio volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Store results
    results[0, i] = portfolio_volatility
    results[1, i] = portfolio_return
    results[2, i] = sharpe_ratio

# Convert results to DataFrame
results_df = pd.DataFrame(results.T, columns=["Volatility", "Return", "Sharpe Ratio"])
weights_df = pd.DataFrame(weights_record, columns=tickers)

# Find the portfolio with the maximum Sharpe Ratio
max_sharpe_idx = results_df["Sharpe Ratio"].idxmax()
max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]
max_sharpe_weights = weights_df.iloc[max_sharpe_idx]

# Find the portfolio with the minimum volatility
min_vol_idx = results_df["Volatility"].idxmin()
min_vol_portfolio = results_df.iloc[min_vol_idx]
min_vol_weights = weights_df.iloc[min_vol_idx]

# Plot the Efficient Frontier
st.subheader("Efficient Frontier")
plt.figure(figsize=(10, 7))
plt.scatter(results_df["Volatility"], results_df["Return"], c=results_df["Sharpe Ratio"], cmap="viridis", alpha=0.7)
plt.colorbar(label="Sharpe Ratio")
plt.scatter(max_sharpe_portfolio[0], max_sharpe_portfolio[1], c="red", marker="*", s=200, label="Max Sharpe Ratio")
plt.scatter(min_vol_portfolio[0], min_vol_portfolio[1], c="blue", marker="*", s=200, label="Min Volatility")
plt.title("Efficient Frontier")
plt.xlabel("Volatility (Standard Deviation)")
plt.ylabel("Expected Return")
plt.legend()
st.pyplot(plt)

# Display key portfolios
st.subheader("Portfolio with Maximum Sharpe Ratio")
st.write(f"Return: {max_sharpe_portfolio['Return']*100:.2f}%, Volatility: {max_sharpe_portfolio['Volatility']*100:.2f}%, Sharpe Ratio: {max_sharpe_portfolio['Sharpe Ratio']:.2f}")
st.write(max_sharpe_weights * 100)

st.subheader("Portfolio with Minimum Volatility")
st.write(f"Return: {min_vol_portfolio['Return']*100:.2f}%, Volatility: {min_vol_portfolio['Volatility']*100:.2f}%, Sharpe Ratio: {min_vol_portfolio['Sharpe Ratio']:.2f}")
st.write(min_vol_weights * 100)
