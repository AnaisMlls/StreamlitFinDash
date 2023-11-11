#==============================================================================
# Initiating: Libraries and functions 
#==============================================================================

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")

    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]
        return cookie

    def _get_yahoo_crumb(self, cookie):
        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]
        ret = {}

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret


#==============================================================================
# Functions 
#==============================================================================

# Function to fetch S&P 500 tickers
@st.cache_data
def fetch_sp500_tickers():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(sp500_url, header=0)
    return sp500_table[0]['Symbol'].tolist()

# Function to fetch stock data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Function to fetch company information
@st.cache_data
def get_company_info(ticker):
    company = YFinance(ticker)  # Changed from yf.Ticker to YFinance
    return company.info

# Function to fetch financial data
@st.cache_data
def get_financial_data(ticker, statement_type, period):
    company = yf.Ticker(ticker)

    if statement_type == "Income Statement":
        if period == "Annual":
            return company.financials
        elif period == "Quarterly":
            return company.quarterly_financials
    elif statement_type == "Balance Sheet":
        if period == "Annual":
            return company.balance_sheet
        elif period == "Quarterly":
            return company.quarterly_balance_sheet
    else:  # Cash Flow
        if period == "Annual":
            return company.cashflow
        elif period == "Quarterly":
            return company.quarterly_cashflow

# Monte Carlo simulation function
@st.cache_data
def monte_carlo_simulation(data, t, n, seed=123):
    # Setting the seed for reproducibility
    np.random.seed(seed)

    log_returns = np.log(1 + data['Close'].pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()

    daily_returns = np.exp(drift + stdev * np.random.randn(t, n))

    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = data['Close'].iloc[-1]

    for t in range(1, t):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]

    return price_paths


#==============================================================================
# Header
#==============================================================================

st.title("Financial Dashboard")
st.markdown("---")

# Data source section with an update button
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write("**_Data source:_**")
with col2:
    st.image('yahoo_finance.png', width=100)
with col3:
    # Update button
    if st.button('Update Data'):
        # Logic to refresh data
        st.experimental_rerun()

# Use the cached function to fetch S&P 500 tickers
ticker_list = fetch_sp500_tickers()

# Dropdown for stock selection
ticker = st.selectbox("Select a stock from S&P 500", ticker_list)


#==============================================================================
# Tab 0: Summary
#==============================================================================
def render_company_profile(ticker):
    st.write("### Stock Price")

    # Duration options in order
    duration_options = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX']

    # Add a date picker for chart duration selection with '1Y' as the default
    chart_duration = st.selectbox(
        "Select Chart Duration", 
        options=duration_options,
        index=duration_options.index('1Y')  # Set default to '1Y'
    )

    # Calculate start date based on selected duration
    end_date = datetime.today().strftime('%Y-%m-%d')
    if chart_duration == 'MAX':
        start_date = '2000-01-01'  # or as far back as desired
    else:
        duration_map = {
            '1M': 30,
            '3M': 90,
            '6M': 180,
            'YTD': (datetime.today() - datetime(datetime.today().year, 1, 1)).days,
            '1Y': 365,
            '3Y': 3 * 365,
            '5Y': 5 * 365
        }
        start_date = (datetime.today() - timedelta(days=duration_map[chart_duration])).strftime('%Y-%m-%d')

    # Fetch stock data
    stock_data = get_stock_data(ticker, start_date, end_date)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))
    fig.update_layout(
        title=f'{ticker} Stock Price Over Time',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white'  # Optional: sets the background to white
    )

    st.plotly_chart(fig)
    # Fetch and display company information
    company_info = get_company_info(ticker)
    
    st.write("### Company Information")
    st.write(f"**Country:** {company_info.get('country', '')}")
    st.write(f"**Industry:** {company_info.get('industryKey', '')}")
    st.write(f"**Website:** [Link]({company_info.get('website', '')})")
    st.write(f"**Description:** {company_info.get('longBusinessSummary', '')}")

    # Fetch and display company information
    st.write("### Key Statistics")
    info_keys = {
        'open': 'Open',
        'bid': 'Bid',
        'ask': 'Ask',
        'dayRange': "Day's Range",
        'fiftyTwoWeekRange': '52 Week Range',
        'volume': 'Volume',
        'averageVolume': 'Avg. Volume',
        'beta': 'Beta (5Y Monthly)',
        'trailingPE': 'PE Ratio (TTM)',
        'trailingEps': 'EPS (TTM)',
        'forwardDividendYield': 'Forward Dividend & Yield',
        'exDividendDate': 'Ex-Dividend Date',
        'oneYearTargetEst': '1Y Target Est'
    }
    # Fetch company info
    company_info = get_company_info(ticker)

    # Create a DataFrame from the statistics dictionary
    detailed_stats = {description: [company_info[key]] for key, description in info_keys.items() if key in company_info}
    # Convert the dictionary to a DataFrame with 'Key' and 'Value' columns
    stats_df = pd.DataFrame(list(detailed_stats.items()), columns=['Description', 'Value'])
    # Adjust the DataFrame's index to start from 1
    stats_df.index = range(1, len(stats_df) + 1)
    st.dataframe(stats_df)

    # Fetch and display major holders information
    major_holders = yf.Ticker(ticker).get_major_holders()
    st.write("### Major Holders")

    if not major_holders.empty:
        # Rename columns
        major_holders.columns = ['Percentage', 'Description']
        # Adjust the DataFrame's index to start from 1 instead of 0
        major_holders.index = range(1, len(major_holders) + 1)
        st.dataframe(major_holders)
    else:
        st.write("No major holders information available.")


#==============================================================================
# Tab 1: Chart
#==============================================================================
def render_chart(ticker):
    st.write("### Time Duration Selection")

    # Duration options dictionary
    duration_options = {
        "Custom": "Custom",
        "1M": 1,
        "3M": 3,
        "6M": 6,
        "YTD": "YTD",
        "1Y": 12,
        "3Y": 36,
        "5Y": 60,
        "MAX": "MAX"
    }

    # Set the default duration to 1Y (12 months)
    duration = st.selectbox("Select Duration", list(duration_options.keys()), index=list(duration_options.keys()).index('1Y'), key="duration_select")

    # Custom date selection
    if duration == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())

        if start_date > end_date:
            st.error("Error: End date must fall after start date.")
    else:
        end_date = datetime.now()
        if duration == "YTD":
            start_date = datetime(end_date.year, 1, 1)
        elif duration == "MAX":
            start_date = None  # Fetch as much data as available
        else:
            # Use a duration map to convert periods to days
            duration_days_map = {
                "1M": 30,
                "3M": 90,
                "6M": 180,
                "1Y": 365,
                "3Y": 3 * 365,
                "5Y": 5 * 365
            }
            start_date = end_date - timedelta(days=duration_days_map[duration])

    # Fetch stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Add Moving Average to the data with a window size of 50 days
    window_size = 50
    stock_data['SMA'] = stock_data['Close'].rolling(window=window_size).mean()

    # Plot Type Selection
    plot_type = st.selectbox("Plot Type", ["Line Plot", "Candlestick Plot"], key="plot_type_select")

    # Create the figure object with a secondary y-axis for volume
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot Line Plot or Candlestick
    if plot_type == "Line Plot":
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'), secondary_y=False)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA'], mode='lines', name='50 Day SMA', line=dict(color='orange')), secondary_y=False)
    else:  # Candlestick Plot
        fig.add_trace(go.Candlestick(x=stock_data.index,
                                     open=stock_data['Open'],
                                     high=stock_data['High'],
                                     low=stock_data['Low'],
                                     close=stock_data['Close'],
                                     name='Candlestick'), secondary_y=False)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA'], mode='lines', name='50 Day SMA', line=dict(color='orange')), secondary_y=False)

    # Plot Trading Volume on the secondary y-axis
    fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume', marker_color='lightblue', width=0.4), secondary_y=True)

    # Add figure title and set layout for the dates to be displayed as required (Day, Month, Year)
    fig.update_layout(
        height=700,
        title_text=f"{ticker} Stock Analysis",
        xaxis=dict(
            tickmode='auto',
            nticks=20,
            tickformat='%d-%m-%Y',
            rangeslider=dict(
                visible=True
            ),
            type='date'
        )
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Price</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


#==============================================================================
# Tab 2: Financials
#==============================================================================
def render_financials(ticker):

    st.write("## Financial Statements")
    statement_type = st.radio(
        "Select the Financial Statement",
        ('Income Statement', 'Balance Sheet', 'Cash Flow')
    )

    # Adding an option to select between Annual and Quarterly
    period = st.radio(
        "Select Period",
        ('Annual', 'Quarterly')
    )

    # Fetch financial data based on selected statement type and period
    financial_data = get_financial_data(ticker, statement_type, period)

    st.write(f"### {statement_type}")
    st.dataframe(financial_data)


#==============================================================================
# Tab 3: Monte Carlo Simulation
#==============================================================================
def render_simulation(ticker):
    st.write("### Monte Carlo Simulation for Stock Price Forecast")

    n_simulations = st.selectbox("Number of Simulations", [200, 500, 1000])
    time_horizon = st.selectbox("Time Horizon (days from today)", [30, 60, 90])

    stock_data = get_stock_data(ticker, datetime.now() - timedelta(days=365), datetime.now())
    current_stock_price = stock_data['Close'].iloc[-1]

    simulated_prices = monte_carlo_simulation(stock_data, time_horizon, n_simulations)
    plt.figure(figsize=(10, 6))

    # Use a more perceptually uniform colormap (e.g., 'viridis')
    colors = plt.cm.viridis(np.linspace(0, 1, n_simulations))

    # Plot each simulation path with a lower opacity
    for i in range(n_simulations):
        plt.plot(simulated_prices[:, i], color=colors[i], linewidth=0.5, alpha=0.25)  # Reduced alpha for less visual clutter

    # Plotting the current stock price as a thicker, bright red dashed line
    plt.axhline(y=current_stock_price, color='red', linestyle='--', linewidth=2, label=f"Current Stock Price: ${current_stock_price:.2f}")

    # Title and labels with increased font size for better readability
    plt.title(f"Monte Carlo Simulation for {ticker} Over Next {time_horizon} Days", fontsize=14)
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Stock Price", fontsize=12)

    # Adding grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Enhance the legend
    plt.legend(loc="upper left", fontsize=10)

    # Show the plot
    st.pyplot(plt)

    # Estimating the Value at Risk (VaR) at 95% confidence interval
    var_95 = np.percentile(simulated_prices[-1], 5)
    st.write(f"#### Value at Risk (VaR): ${var_95:.2f}")

 
#==============================================================================
# Tab 4: Additional Analysis
#==============================================================================
def render_analysis(ticker):
    st.subheader("Stocks Comparison and Financial Metrics")

    # Fetch the tickers from Wikipedia's S&P 500 list only once
    sp500_tickers = fetch_sp500_tickers()

    # Allow users to select stocks for comparison and metrics
    tickers = st.multiselect("Select up to 3 Stocks", options=sp500_tickers, default=['AAPL', 'MSFT', 'AMZN'])

    # Limit to 3 selections
    if len(tickers) > 3:
        st.warning("Please select no more than 3 stocks.")
        tickers = tickers[:3]

    if tickers:
        # Fetch and display closing price comparison chart using cached data
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        comparison_data = pd.DataFrame()
        for selected_ticker in tickers:
            stock_data = get_stock_data(selected_ticker, start_date, end_date)
            comparison_data[selected_ticker] = stock_data['Close']
        st.line_chart(comparison_data)

        # Collect financial metrics for each selected ticker
        metrics_data = []
        for selected_ticker in tickers:
            stock_info = get_company_info(selected_ticker)
            metrics_data.append({
                "Ticker": selected_ticker,
                "Market Cap": stock_info.get('marketCap', "Data not available"),
                "P/E Ratio": stock_info.get('trailingPE', "Data not available"),
                "Dividend Yield": stock_info.get('dividendYield', "Data not available")
            })

        # Convert list of dictionaries to DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.index += 1  # Adjust index to start from 1

        # Display the DataFrame
        st.dataframe(metrics_df)

#==============================================================================
# Main App Function
#==============================================================================  
def main():
   
    # Tabs
    tabs = st.tabs(["Company Profile", "Chart", "Financials", "Stock Simulation", " Stocks Comparison "])
    with tabs[0]:
        render_company_profile(ticker)
    with tabs[1]:
        render_chart(ticker)
    with tabs[2]:
        render_financials(ticker)
    with tabs[3]:
        render_simulation(ticker)
    with tabs[4]:
        render_analysis(ticker)
if __name__ == "__main__":
    main()
#==============================================================================
# END
#============================================================================== 








