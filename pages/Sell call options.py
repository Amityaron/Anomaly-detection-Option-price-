# Import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

# Black-Scholes Option Pricing Model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Function to calculate probability of expiring OTM
def probability_otm(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return 1 - norm.cdf(d2)  # Probability of expiring OTM for calls
    elif option_type == "put":
        return norm.cdf(d2)  # Probability of expiring OTM for puts

# App title and description
st.title("Options Selling Model App with Probability of Expiring OTM")
st.write("This app displays sell put and call options with their Option Pricing Model (OPM) values and the probability of expiring OTM by expiration. Filter by the probability of expiring OTM.")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter a stock ticker (e.g., SPY)", "SPY")
otm_prob_filter = st.sidebar.slider("Filter by Probability of Expiring OTM", min_value=0.0, max_value=1.0, value=(0.0, 1.0))

# Function to fetch options data (expiration dates only, to be cacheable)
@st.cache_data
def fetch_expiration_dates(ticker):
    stock = yf.Ticker(ticker)
    return stock.options

# Get stock and expiration dates
try:
    expiration_dates = fetch_expiration_dates(ticker)
    if not expiration_dates:
        st.error("No options data available for this ticker.")
    else:
        # Select expiration date
        exp_date = st.sidebar.selectbox("Select expiration date", expiration_dates)

        # Fetch options chain for the selected expiration date
        stock = yf.Ticker(ticker)
        options_chain = stock.option_chain(exp_date)
        calls = options_chain.calls
        puts = options_chain.puts

        # Get stock price and set risk-free rate
        history = stock.history(period="1d")
        if history.empty:
            st.error("No historical data available for this ticker.")
        else:
            stock_price = history["Close"].iloc[-1]
            risk_free_rate = 0.04  # Example risk-free rate

            # Time to expiration
            T = (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.now()).days / 365

            # Calculate OPM and probability of expiring OTM for each option
            def calculate_opm_and_otm_prob(df, option_type):
                df = df.dropna(subset=['impliedVolatility'])
                df["OPM"] = df.apply(
                    lambda x: black_scholes(
                        S=stock_price,
                        K=x["strike"],
                        T=T,
                        r=risk_free_rate,
                        sigma=x["impliedVolatility"],
                        option_type=option_type
                    ),
                    axis=1
                )
                df["P(OTM)"] = df.apply(
                    lambda x: probability_otm(
                        S=stock_price,
                        K=x["strike"],
                        T=T,
                        r=risk_free_rate,
                        sigma=x["impliedVolatility"],
                        option_type=option_type
                    ),
                    axis=1
                )
                return df
            
            # find_best_put_credit_spread
            def find_best_put_credit_spread(puts_df, stock_price, spread_widths):
                best_spread = None
                max_ror = -float('inf')

                for width in spread_widths:
                    for idx, short_put in puts_df.iterrows():
                        short_strike = short_put['strike']
                        long_strike = short_strike - width

                        # Find matching long put
                        long_put = puts_df[puts_df['strike'] == long_strike]
                        if long_put.empty:
                            continue
                        long_put = long_put.iloc[0]

                        # Calculate metrics
                        net_credit = short_put['OPM'] - long_put['OPM']
                        max_loss = width - net_credit
                        if max_loss <= 0:
                            continue  # Avoid division by zero or negative loss
                        ror = net_credit / max_loss
                        breakeven = short_strike - net_credit

                        if ror > max_ror:
                            max_ror = ror
                            best_spread = {
                                'Short Strike': short_strike,
                                'Long Strike': long_strike,
                                'Net Credit': round(net_credit, 2),
                                'Max Loss': round(max_loss, 2),
                                'Return on Risk': round(ror * 100, 2),
                                'Break-even': round(breakeven, 2),
                                'Probability OTM': round(short_put['P(OTM)'] * 100, 2)
                            }

                return best_spread

            # Calculate OPM and P(OTM) for calls and puts
            calls = calculate_opm_and_otm_prob(calls, "call")
            puts = calculate_opm_and_otm_prob(puts, "put")

            # Concatenate calls and puts into a single DataFrame
            options_df = pd.concat([calls.assign(Type="Sell Call"), puts.assign(Type="Sell Put")])

            # Filter based on Probability of Expiring OTM range
            filtered_options = options_df[(options_df["P(OTM)"] >= otm_prob_filter[0]) & (options_df["P(OTM)"] <= otm_prob_filter[1])]

            # Display the filtered options
            st.write("### Filtered Options for Selling")
            st.write(filtered_options[["Type", "strike", "lastPrice", "impliedVolatility", "OPM", "P(OTM)"]])
             # Define spread widths to evaluate
            spread_widths = [5, 10, 15, 20,25,30]

            # Filter puts based on user-defined OTM probability
            filtered_puts = puts[(puts['P(OTM)'] >= otm_prob_filter[0]) & (puts['P(OTM)'] <= otm_prob_filter[1])]
            st.write(filtered_puts)
            # Find the best spread
            best_spread = find_best_put_credit_spread(filtered_puts, stock_price, spread_widths)
            
            # Display the result
            if best_spread:
                st.subheader("📊 Optimal Put Credit Spread")
                st.write(pd.DataFrame([best_spread]))
            else:
                st.warning("No suitable put credit spread found for the selected parameters.")
except Exception as e:
    st.error("Could not retrieve data for the provided ticker symbol. Please check the ticker and try again.")
    st.error(f"Error details: {e}")
st.title("📈 SPY Implied Volatility vs Strike Price")

# Assume 'ticker' is already defined in your session (e.g., from sidebar input)
ticker_obj = ticker
    
# Get nearest expiration date
st.write(ticker_obj.options[0]) 
expiration = ticker_obj.options[0]
st.write(expiration)    
# Fetch option chain
opt_chain = ticker_obj.option_chain(expiration)
calls = opt_chain.calls
puts = opt_chain.puts
    
# Drop rows with missing IV or price data
calls = calls.dropna(subset=["impliedVolatility", "lastPrice", "bid", "ask"])
puts = puts.dropna(subset=["impliedVolatility", "lastPrice", "bid", "ask"])
    
# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(calls['strike'], calls['impliedVolatility'], label='Calls', color='blue')
ax.plot(puts['strike'], puts['impliedVolatility'], label='Puts', color='red')
ax.set_xlabel('Strike Price')
ax.set_ylabel('Implied Volatility')
ax.set_title(f'SPY IV vs Strike Price (Expiration: {expiration})')
ax.legend()
ax.grid(True)
    
# Display in Streamlit
st.pyplot(fig)


