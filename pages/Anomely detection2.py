# Build Streamlit app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go

# Set the app title 
st.title('Anomaly Detection Stock Market App')
st.write('Welcome to my Anomaly detection app!')

# Create a text input 
widgetuser_input = st.text_input('Enter a ticker based on Yahoo Finance:', 'SPY')

# Create date inputs for start and end dates
start_date = st.date_input('Start Date', value=pd.to_datetime('2025-01-01'))
end_date = st.date_input('End Date', value=pd.Timestamp.now())

# Download data from Yahoo Finance
symbol = widgetuser_input
spy = yf.download(symbol, start=start_date, end=end_date)

# Defensive check
if spy.empty:
    st.error("No data was downloaded. Please check the ticker or date range.")
    st.stop()

# Verify and flatten multi-level indexes
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

# ================================
# Diff Z-Score 22 Days Section
# ================================

n = 22  # number of days for diff and z-score

if 'Close' in spy.columns:
    # Calculate 22-day difference:
    # Diff = Close today - Close 22 trading days ago
    spy['Diff_22'] = spy['Close'] - spy['Close'].shift(n)

    # Calculate rolling mean and rolling standard deviation of Diff_22
    spy['Diff_22_Mean'] = spy['Diff_22'].rolling(n).mean()
    spy['Diff_22_STD'] = spy['Diff_22'].rolling(n).std()

    # Calculate z-score of the 22-day difference
    spy['Diff_Z_Score_22'] = (
        (spy['Diff_22'] - spy['Diff_22_Mean']) / spy['Diff_22_STD']
    )

    # Remove rows with NaN values
    spy.dropna(subset=['Close', 'Diff_Z_Score_22'], inplace=True)

    # Generate buy signal
    # Buy when Diff Z-Score 22 Days < -3
    spy['Signal'] = 0
    spy.loc[spy['Diff_Z_Score_22'] < -2.5, 'Signal'] = 1
    spy['Signal'] = spy['Signal'].diff().fillna(0)
    spy.loc[spy['Signal'] < 0, 'Signal'] = 0

    # Generate sell signal
    # Sell when Diff Z-Score 22 Days > 2
    spy['Sell_Signal'] = 0
    spy.loc[spy['Diff_Z_Score_22'] > 2.5, 'Sell_Signal'] = 1
    spy['Sell_Signal'] = spy['Sell_Signal'].diff().fillna(0)
    spy.loc[spy['Sell_Signal'] < 0, 'Sell_Signal'] = 0

else:
    st.error("Close column was not found in the downloaded data.")
    st.stop()

# Calculate percentage change for buy signals
lower_dates, buy_prices = [], []

for index, row in spy.loc[spy['Signal'] == 1].iterrows():
    buy_prices.append(row['Close'])
    lower_dates.append(index)

pct_change = [((spy.iloc[-1]['Close'] / x) - 1) * 100 for x in buy_prices]

# ================================
# Plot only price and signals - Plotly
# ================================

title = f"{symbol} ({start_date} to {end_date})"

add_bollinger = st.checkbox("Add Bollinger Band", value=False)

if add_bollinger:
    bb_n = st.number_input(
        "Bollinger Band n",
        min_value=1,
        value=20,
        step=1
    )

    bb_sd = st.number_input(
        "Bollinger Band sd",
        min_value=0.1,
        value=2.0,
        step=0.1
    )

    spy['BB_Middle'] = spy['Close'].rolling(int(bb_n)).mean()
    spy['BB_STD'] = spy['Close'].rolling(int(bb_n)).std()
    spy['BB_Upper'] = spy['BB_Middle'] + (bb_sd * spy['BB_STD'])
    spy['BB_Lower'] = spy['BB_Middle'] - (bb_sd * spy['BB_STD'])

fig = go.Figure()

# Plot Close price only
fig.add_trace(
    go.Scatter(
        x=spy.index,
        y=spy['Close'],
        mode='lines',
        name='Close'
    )
)

# Plot buy signals
fig.add_trace(
    go.Scatter(
        x=spy.loc[spy['Signal'] == 1].index,
        y=spy.loc[spy['Signal'] == 1, 'Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(size=10, symbol='circle')
    )
)

# Plot sell signals
fig.add_trace(
    go.Scatter(
        x=spy.loc[spy['Sell_Signal'] == 1].index,
        y=spy.loc[spy['Sell_Signal'] == 1, 'Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(size=10, symbol='x')
    )
)

# Current price horizontal line
fig.add_hline(
    y=spy.iloc[-1]['Close'],
    line_dash='dash',
    line_color='red',
    annotation_text='Current Price',
    annotation_position='top left'
)

# Optional Bollinger Bands
if add_bollinger:
    fig.add_trace(
        go.Scatter(
            x=spy.index,
            y=spy['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(dash='dot')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=spy.index,
            y=spy['BB_Middle'],
            mode='lines',
            name='BB Middle',
            line=dict(dash='dot')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=spy.index,
            y=spy['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(dash='dot')
        )
    )

fig.update_layout(
    title=title,
    xaxis_title='Date',
    yaxis_title='Price',
    hovermode='x unified',
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Display Buy Signal Dates, Buy Prices, and Gain Percentages
df = pd.DataFrame({
    'Buy_Signal_Date': lower_dates,
    'Buy Price': buy_prices,
    'Gain_Pct': pct_change
})

st.write("Current price:", round(spy.iloc[-1]['Close'], 2))
st.write("Latest Diff Z-Score 22 Days:", round(spy.iloc[-1]['Diff_Z_Score_22'], 2))
st.table(df.round(2))


# ================================
# Monthly Percentage Changes
# ================================

st.subheader('Monthly Percentage Changes')

# Start and end dates for downloading historical data
start_date_sp500 = '1990-01-01'
end_date_sp500 = pd.Timestamp.now()

# Download the data for the specified ticker symbol
symbol = st.text_input("Enter a ticker symbol:", "SPY")
sp500_data = yf.download(symbol, start=start_date_sp500, end=end_date_sp500)

# Defensive check
if sp500_data.empty:
    st.error("No monthly data was downloaded. Please check the ticker symbol.")
    st.stop()

# Flatten multi-level columns if needed
if isinstance(sp500_data.columns, pd.MultiIndex):
    sp500_data.columns = sp500_data.columns.get_level_values(0)

# Ensure the data is sorted by date
sp500_data.sort_index(inplace=True)

# Calculate daily and monthly returns
sp500_data['Daily Return'] = sp500_data['Close'].pct_change()
monthly_returns = sp500_data['Close'].resample('ME').ffill().pct_change() * 100

# Verify and convert monthly_returns to a Series if needed
if isinstance(monthly_returns, pd.DataFrame):
    monthly_returns = monthly_returns.squeeze()

# Check if monthly_returns is empty or invalid
if monthly_returns.empty:
    st.write("Error: No monthly returns data available to display.")
else:
    # Creating DataFrame for monthly returns and other related columns
    monthly_returns_df = monthly_returns.to_frame(name='Monthly Return')
    monthly_returns_df['Year'] = monthly_returns_df.index.year
    monthly_returns_df['Month'] = monthly_returns_df.index.month
    monthly_returns_df['Month Name'] = monthly_returns_df.index.strftime('%B')

    # Filter data for the latest year
    last_year = monthly_returns_df[
        monthly_returns_df['Year'] == monthly_returns_df['Year'].max()
    ]
    last_year_returns = last_year.set_index('Month Name')['Monthly Return']

    # Get the start date for the data and calculate total years observed
    start_date_sp500_actual = sp500_data.index[0]
    total_years = int(end_date_sp500.year - start_date_sp500_actual.year)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))

    sns.boxplot(
        x='Month Name',
        y='Monthly Return',
        data=monthly_returns_df,
        order=[
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ],
        ax=ax
    )

    ax.scatter(
        last_year_returns.index,
        last_year_returns.values,
        color='red',
        zorder=5,
        label='Last Year Returns'
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax.set_title(
        f'Monthly Percentage Changes for {symbol} '
        f'from {start_date_sp500_actual.date()} to {end_date_sp500.date()} '
        f'(Total years: {total_years})'
    )

    ax.set_xlabel('Month')
    ax.set_ylabel('Percentage Change')
    ax.grid(True)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.legend()

    # Display the plot and additional information
    st.pyplot(fig)

    st.write(f'Monthly Percentage Changes for {symbol}')
    st.write(f'Data from {start_date_sp500_actual.date()} until {end_date_sp500.date()}')
    st.write(f'Total number of years observed: {total_years}')

    # Calculate Probabilities of Positive Monthly Returns
    monthly_positive_counts = {
        month: 0 for month in [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
    }

    monthly_total_counts = {
        month: 0 for month in [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
    }

    for date, return_value in monthly_returns.items():
        if not pd.isna(return_value):
            month_name = date.strftime('%B')
            monthly_total_counts[month_name] += 1

            if return_value > 0:
                monthly_positive_counts[month_name] += 1

    probabilities = {}

    for month in monthly_positive_counts:
        if monthly_total_counts[month] > 0:
            probability = (
                monthly_positive_counts[month] / monthly_total_counts[month]
            ) * 100
            probabilities[month] = f"{probability:.1f}%"
        else:
            probabilities[month] = "N/A"

    # Display the probabilities table
    probabilities_df = pd.DataFrame(
        list(probabilities.items()),
        columns=['Month', 'Probability (%)']
    )

    st.write("Probability of Positive Monthly Returns:")
    st.table(probabilities_df)
