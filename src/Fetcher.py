import yfinance as yf
import pandas as pd
import numpy as np
import json
from pathlib import Path

def fetch_price_data(tickers, start="2010-01-01", end=None):
    """
    Returns closing prices for list of tickers from yahoo finance in a pandas dataframe
    Rows/Index: date YYYY-MM-DD
    Columns: ticker closing price
    """
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

def get_log_returns(price_data, date="2026-03-22", years = 1):
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

def get_expected_returns(returns):
    """
    Estimates expected return for each ticker based on mean-log * 255 trading days
    """
    mu = (returns.mean() * 252).values
    return mu

def get_covariance_matrix(returns):
    """
    Returns covariance matrix of daily returns (i.e. log returns)
    """
    sigma = (returns.cov() * 252).values
    return sigma

def set_tickers(default_list):
    """
    Prompts  user to add tickers for analysis or use default portfolio
    """
    tickers = []
    print("Add tickers to list one by one")
    print("Leave ticker list empty to use default list")
    while True:
        ticker = input("Add ticker to list or press Enter to finnish: ")
        if ticker != "":
            tickers.append(ticker)
        else:
            break
    if len(tickers) == 0:
        print("No tickers added, using default list")
        tickers = default_list
    return tickers

def get_default_portfolio(filepath):
    """
    Takes in filepath where portfolio is located
    returns default portfolio dict with keys = ticker, value = weight
    """
    root = Path(__file__).resolve().parents[1]
    file_path = root / filepath
    with open(file_path, "r") as f:
        data = json.load(f)
    values = np.array(list(data.values()), dtype=float)
    total = values.sum()
    for ticker, weights in data.items():
        if total != 0:
            data[ticker] = data[ticker] / total
        else:
            data[ticker] = 1 / len(data)
    return data

def set_target_return():
    """
    Input function for optimizer based on target return
    """
    try:
        p = float(input("Choose target return: "))
    except ValueError:
        print("Invalid input. Using default = 0.1")
        p = 0.1
    if p < 0:
        p = 0
    print(f"Setting expected return to {p}")
    return p

def set_target_volatility():
    """
    Input function for optimizer based on target volatility
    """
    try:
        p = float(input("Choose target volatility: "))
    except ValueError:
        print("Invalid input. Using default = 0.1")
        p = 0.1
    if p < 0:
        p = 0
    print(f"Setting expected volatility to {p}")
    return p





