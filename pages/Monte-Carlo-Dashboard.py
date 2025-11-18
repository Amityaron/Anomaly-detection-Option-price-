import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from scipy.stats import norm

# Set the title and favicon that appear in the browser's tab bar.
st.set_page_config(
    page_title='Monte Carlo Stock Price Simulation',
    page_icon=':chart_with_upwards_trend:',  # Stock chart emoji.
)

# -------------------------------------------------------------------
# Declare some useful functions.

def geo_paths(S, T, r, q, sigma, steps, N):
    """Generates paths for a geometric Brownian motion."""
    dt = T / steps
    # Initialize the paths with the initial log stock price
    ST = np.zeros((steps + 1, N))
    ST[0] = np.log(S)
    # Simulate the log stock prices
    for t in range(1, steps + 1):
        ST[t] = (ST[t - 1] +
                 (r - q - 0.5 * sigma**2) * dt +
                 sigma * np.sqrt(dt) * np.random.normal(size=N))
    return np.exp(ST)

def black_scholes(S, K, r, T, sigma):
    """Calculate European Call Option Price using Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

@st.cache_data
def get_stock_data(ticker, start, end):
    """Fetch stock data using yfinance."""
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        return stock_data['Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# -------------------------------------------------------------------
# Page content and user interaction

# Set the title that appears at the top of the page.
st.title(":chart_with_upwards_trend: Monte Carlo Stock Price Simulation")

st.write("""
Simulate future stock prices using the Monte Carlo method and compare with the Black-Scholes model. Select a stock, and adjust parameters like the risk-free rate, volatility, and time horizon.
""")

# Sidebar input section for parameters
st.sidebar.header("Monte Carlo Simulation Parameters")

stock_ticker = st.sidebar.text_input('Enter stock ticker (e.g., AAPL, TSLA, MSFT):', 'AAPL')

# Date range input for historical data fetching
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300)

# Fetch stock data
if stock_ticker:
    stock_data = get_stock_data(stock_ticker, start=start_date, end=end_date)
    if stock_data is not None and not stock_data.empty:
        st.write(f"Displaying closing prices for {stock_ticker}:")
        st.line_chart(stock_data)

        # Automatically set S0 (Initial Stock Price) to the most recent stock price
        S0 = stock_data.iloc[-1].item() if not stock_data.empty else None
        if S0 is not None:
            st.sidebar.write(f"Latest Stock Price (S_0): {S0:.2f}")

            # Sidebar sliders for other parameters
            K = st.sidebar.slider('Strike Price (K)', min_value=int(S0 * 0.5), max_value=int(S0 * 2), value=int(S0 * 1.1))
            r = st.sidebar.slider('Risk-Free Rate (r)', min_value=0.0, max_value=0.1, value=0.05, step=0.01)
            sigma = st.sidebar.slider('Volatility (σ)', min_value=0.05, max_value=1.0, value=0.2, step=0.01)
            T = st.sidebar.slider('Time to Maturity (T) Years', min_value=0.05, max_value=5.0, value=1.0, step=0.1)
            N = st.sidebar.slider('Number of Simulations (N)', min_value=10, max_value=1000, value=100)

            # Time steps fixed at 100 for now
            steps = 100

            # Perform Monte Carlo simulation
            paths = geo_paths(S0, T, r, 0, sigma, steps, N)

            # Plot the simulation paths
            st.subheader('Monte Carlo Simulation Results')
            fig, ax = plt.subplots()
            ax.plot(paths)
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Stock Price")
            ax.set_title(f"Simulated Stock Price Paths for {stock_ticker}")
            st.pyplot(fig)

            # Displaying some statistics
            st.write("Monte Carlo Simulation Generates paths for a geometric Brownian motion.")
            st.write(f"Simulated final stock price mean: {paths[-1].mean():.2f}")
            st.write(f"Simulated final stock price standard deviation: {paths[-1].std():.2f}")

            # -------------------------------------------------------------------
            # Volatility Heatmap for Call Prices
            st.subheader('Volatility Heatmap for Call Option Prices')
            K_values = np.linspace(int(S0 * 0.5), int(S0 * 2), 50)  # Strike prices
            sigma_values = np.linspace(0.1, 1.0, 50)  # Volatilities
            call_prices = np.zeros((len(K_values), len(sigma_values)))

            for i, K_val in enumerate(K_values):
                for j, sigma_val in enumerate(sigma_values):
                    call_prices[i, j] = black_scholes(S0, K_val, r, T, sigma_val)

            fig, ax = plt.subplots()
            c = ax.imshow(call_prices, aspect='auto', cmap='coolwarm', extent=[sigma_values.min(), sigma_values.max(), K_values.min(), K_values.max()])
            ax.set_xlabel('Volatility (σ)')
            ax.set_ylabel('Strike Price (K)')
            ax.set_title(f"Call Option Prices for {stock_ticker}")
            fig.colorbar(c, ax=ax)
            st.pyplot(fig)

            # -------------------------------------------------------------------
            # Convergence Plot for Monte Carlo Simulation
            cumulative_averages = []
            total = 0

            # Calculate cumulative averages over each simulation
            for i in range(1, N + 1):
                total += paths[-1, i - 1]  # Sum of the final stock prices
                cumulative_averages.append(total / i)  # Cumulative average

            # Plot the convergence chart
            st.subheader('Convergence of Monte Carlo Simulations')
            fig, ax = plt.subplots()
            ax.plot(range(1, N + 1), cumulative_averages, label='Convergence of Estimate')
            ax.set_xlabel("Number of Simulations")
            ax.set_ylabel("Cumulative Average of Final Stock Prices")
            ax.set_title(f"Convergence Chart for {stock_ticker}")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Unable to fetch the latest stock price.")
    else:
        st.warning("No data available for the entered stock ticker.")
else:
    st.warning("Please enter a valid stock ticker.")
