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

def set_portfolio(default_portfolio_path):
    """
    Prompts user to add ticker-weight pair for analysis or use default portfolio
    """
    print("ADD TICKER-WEIGHT PAIRS (ONE PAIR AT A TIME)")
    print("LEAVE PORTFOLIO EMPTY TO USE DEFAULT PORTFOLIO")
    portfolio = {}
    while True:
        ticker = input("Enter ticker or press Enter to finish portfolio building: ").strip()
        if ticker == "":
            break
        weight = input("Enter weight or value of asset (number): ").strip()
        try:
            weight = float(weight)  # convert immediately
        except ValueError:
            print("Value must be a number. Ticker-weight pair NOT added.")
            continue
        portfolio[ticker] = weight
    if len(portfolio) != 0:
        save = input(f"Do you want to save portfolio? (y/n): ").strip().lower()
        if save == "y":
            save_portfolio(portfolio, default_portfolio_path)
    else:
        print(f"No portfolio added, using default portfolio at {default_portfolio_path}")
        portfolio = get_default_portfolio(default_portfolio_path)
    return portfolio

def get_default_portfolio(default_portfolio_path):
    root = Path(__file__).resolve().parents[1]
    file_path = root / default_portfolio_path
    with open(file_path, "r") as f:
        portfolio = json.load(f)
    return normalize_portfolio(portfolio)

def save_portfolio(portfolio, default_portfolio_path):
    name = input("Enter file name to save (without extension) or press Enter for default portfolio: ").strip()
    file_name = f"{name}.json" if name != "" else default_portfolio_path
    root = Path(__file__).resolve().parents[1]
    save_path = root / "data" / file_name
    try:
        with open(save_path, "w") as f:
            json.dump(normalize_portfolio(portfolio), f, indent=4)
        print(f"Saved portfolio to {save_path}")
    except Exception as e:
        print(f"Failed to save file: {e}")
    return

def normalize_portfolio(portfolio):
    # Helper to normalize portfolio weights
    values = np.array(list(portfolio.values()), dtype=float)
    total = values.sum()
    if len(portfolio) == 0:
        print("Warning: no portfolio to normalize")
        return {}
    for ticker, weights in portfolio.items():
        if total != 0:
            portfolio[ticker] = portfolio[ticker] / total
        else:
            portfolio[ticker] = 1 / len(portfolio)
    return portfolio

def validate_portfolio(portfolio, valid_tickers):
    valid_portfolio = {k: v for k, v in portfolio.items() if k.upper() in set(valid_tickers)}
    return valid_portfolio

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





