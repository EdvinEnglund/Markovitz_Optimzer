import yfinance as yf
import pandas as pd
import numpy as np
from config import YEARS

def fetch_price_data(tickers, start="2010-01-01", end=None):
    """
    Returns closing prices for portfolio or list of tickers from yahoo finance in a pandas dataframe
    Rows/Index: date YYYY-MM-DD
    Columns: ticker closing price
    """
    if isinstance(tickers, dict):
        tickers = list(tickers.keys())

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )
    price_data = data.xs("Close", axis=1, level=0)
    # Drop columns with all NaN (invalid tickers)
    price_data = price_data.dropna(axis=1, how="all")
    return price_data

def get_log_returns(price_data, date="2026-03-22", years = YEARS):
    """
    Takes: Pandas DataFrame price_data with closing prices.
    Crops by reference date and timescope in years
    Returns: Pandas DataFrame price_data with log. daily returns
    """
    end_date = pd.to_datetime(date)
    start_date = end_date - pd.DateOffset(years=years)
    price_data = price_data.loc[start_date:end_date]
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    return log_returns








