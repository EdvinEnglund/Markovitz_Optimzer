import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import json
from pathlib import Path

def fetch_price_data(tickers, start="2010-01-01", end=None):
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

def get_stats(price_data, date="2026-03-22", years = 1):
    """
    Input:
        - Pandas DataFrame price_data
        - date: lasr day of period to base mu and sigma on
        - years, no o years to base data on.
    Returns:
        - all valid tickers from input
        - expected returns
        - covariance matrix (numpy 2D array)
    """
    end_date = pd.to_datetime(date)
    start_date = end_date - pd.DateOffset(years=years)
    price_data = price_data.loc[start_date:end_date]

    log_returns = np.log(price_data / price_data.shift(1)).dropna()

    mu = (log_returns.mean() * 252).values
    sigma = (log_returns.cov() * 252).values

    valid_tickers = list(price_data.columns)
    return valid_tickers, mu, sigma

def set_tickers(default_list):
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


# Get tickers from default portfolio
def get_tickers_from_json(filepath):
    root = Path(__file__).resolve().parents[1]
    file_path = root / filepath
    with open(file_path, "r") as f:
        data = json.load(f)
    return list(data.keys())

# Get weights from default portfolio
def get_weights_from_json(filepath):
    root = Path(__file__).resolve().parents[1]
    file_path = root / filepath
    with open(file_path, "r") as f:
        data = json.load(f)
    values = np.array(list(data.values()), dtype=float)
    total = values.sum()
    if total == 0:
        raise ValueError("Cannot normalize: sum of values is 0")
    normalized = values / total
    return normalized


def set_target_return():
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
    try:
        p = float(input("Choose target volatility: "))
    except ValueError:
        print("Invalid input. Using default = 0.1")
        p = 0.1
    if p < 0:
        p = 0
    print(f"Setting expected volatility to {p}")
    return p

if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    years = 1
    end_date = pd.to_datetime(today_str)
    start_date = end_date - pd.DateOffset(years=years)
    tickers = set_tickers()
    price_df = fetch_price_data(tickers, start=start_date, end=end_date)

    stats = get_stats(price_df, "2026-03-22", years)
    print(stats)



