import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_price_data(tickers, start="2025-01-01", end=None):
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

def compute_log_returns(price_df):
    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns.dropna()

def get_stats(tickers, years = 1):
    """
    Main pipeline function.
    Input:
        - tickers: list of tickers
        - years: number of years to base mu and sigma on
    Returns:
        - all valid tickers from input
        - expected returns
        - covariance matrix (numpy 2D array)
    """
    # Compute start day
    today_mmdd = datetime.today().strftime('%m-%d')
    yyyy = datetime.today().strftime("%Y")
    start = f"{int(yyyy) - years}-{today_mmdd}"

    # Fetch price data from yfinance
    price_data = fetch_price_data(tickers, start, None)

    # Get mu and Sigma
    returns = compute_log_returns(price_data)
    mu = (returns.mean() * 252).values
    sigma = (returns.cov() * 252).values

    valid_tickers = list(price_data.columns)
    return valid_tickers, mu, sigma

def set_tickers():
    tickers = []
    while True:
        ticker = input("Enter the ticker to fetch or press Enter to finish: ")
        if ticker != "":
            tickers.append(ticker)
        else:
            break
    return tickers

if __name__ == "__main__":
    tickers = set_tickers()
    stats = get_stats(tickers, 5)
    print(stats)


