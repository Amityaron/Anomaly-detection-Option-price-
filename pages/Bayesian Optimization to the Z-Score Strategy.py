import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

# Streamlit configuration for user inputs
st.title("Investment Strategy Comparison")

# User input fields
spy = st.text_input("Enter the ticker for SPY (S&P 500 ETF):", "QQQ")
sector_etfs = st.text_input("Enter sector ETF tickers separated by commas:", "IXN,QQQ").split(',')
initial_cash = st.number_input("Enter initial cash investment:", min_value=1000, value=10000, step=500)
years = st.number_input("Enter the number of years for annual return calculation:", min_value=1, value=17, step=1)

# Backtest parameters
months = years * 12
monthly_investment = initial_cash / months

# Function to fetch data from Yahoo Finance
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d')['Adj Close']
    return data

# Fetch data
start_date = '2007-01-01'
end_date = '2024-01-01'
spy_data = fetch_data(spy, start_date, end_date)
sector_data = fetch_data(sector_etfs, start_date, end_date)

# --- Strategy 1: DCA on SPY ---
def dca_strategy(spy_data, monthly_investment):
    spy_shares = 0
    spy_data_monthly = spy_data.resample('M').last()
    for price in spy_data_monthly:
        shares_bought = monthly_investment / price
        spy_shares += shares_bought
    portfolio_value = spy_shares * spy_data_monthly[-1]
    return portfolio_value

# --- Strategy 2: Z-Score Sector Strategy with Bayesian Optimization ---
def z_score_sector_strategy(sector_data, monthly_investment, z_threshold, window):
    sector_shares = {sector: 0 for sector in sector_data.columns}
    window = int(window)

    i = 0
    while i <= len(sector_data) - window:
        rolling_data = sector_data.iloc[i:i + window]
        rolling_mean = rolling_data.mean()
        rolling_std = rolling_data.std()
        current_prices = sector_data.iloc[i + window - 1]
        z_scores = (current_prices - rolling_mean) / rolling_std

        if (z_scores <= z_threshold).any():
            selected_ticker = z_scores.idxmin()
            shares_bought = monthly_investment / sector_data[selected_ticker].iloc[i + window - 1]
            sector_shares[selected_ticker] += shares_bought
            i += (window - i % window) - 1
        else:
            if i % window == 0:
                selected_ticker = z_scores.idxmin()
                shares_bought = monthly_investment / sector_data[selected_ticker].iloc[i + window - 1]
                sector_shares[selected_ticker] += shares_bought

        i += 1

    portfolio_value = sum(sector_shares[sector] * sector_data[sector].iloc[-1] for sector in sector_shares)
    return portfolio_value

# Optimization function
def optimize_z_score_strategy(window, z_threshold):
    return z_score_sector_strategy(sector_data, monthly_investment, z_threshold, window)

# Define the parameter bounds for window and z_threshold
pbounds = {
    'window': (20, 60),
    'z_threshold': (-3.0, -1.0)
}

# Perform Bayesian optimization
optimizer = BayesianOptimization(f=optimize_z_score_strategy, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=10, n_iter=50)

# Best parameters
best_params = optimizer.max['params']
z_score_result = z_score_sector_strategy(sector_data, monthly_investment, best_params['z_threshold'], best_params['window'])

# Calculate the annual profit return (CAGR)
def calculate_annual_return(final_value, initial_investment, years):
    return (final_value / initial_investment) ** (1 / years) - 1

dca_result = dca_strategy(spy_data, monthly_investment)
dca_annual_return = calculate_annual_return(dca_result, initial_cash, years)
z_score_annual_return = calculate_annual_return(z_score_result, initial_cash, years)

# Display results
st.write(f"Annual Profit Return (DCA on {spy}): {dca_annual_return * 100:.2f}%")
st.write(f"Annual Profit Return (Z-Score Sector Strategy): {z_score_annual_return * 100:.2f}%")
st.write(f"Profit of the DCA on {spy}: ${dca_result:,.2f}")
st.write(f"Profit of the Z-Score Sector Strategy: ${z_score_result:,.2f}")

# --- Visualization ---
fig, ax = plt.subplots()
labels = [f'DCA on {spy}', 'Z-Score Sector Strategy']
values = [dca_result, z_score_result]
ax.bar(labels, values, color=['blue', 'green'])
ax.set_title(f'Portfolio Value Comparison (After {years} Years)')
ax.set_ylabel('Portfolio Value ($)')
st.pyplot(fig)
