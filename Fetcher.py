import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
today_str = date.today().strftime("%Y-%m-%d")

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

def get_stats(tickers, date=today_str, years = 1):
    """
    Input:
        - tickers: list of tickers
        - date: lasr day of period to base mu and sigma on
        - years, no o years to base data on.
    Returns:
        - all valid tickers from input
        - expected returns
        - covariance matrix (numpy 2D array)
    """
    end_date = pd.to_datetime(date)
    start_date = end_date - pd.DateOffset(years=years)

    price_data = fetch_price_data(tickers, start_date, end_date)

    log_returns = np.log(price_data / price_data.shift(1)).dropna()

    mu = (log_returns.mean() * 252).values
    sigma = (log_returns.cov() * 252).values

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
    tickers = set_tickers()
    stats = get_stats(tickers, "2026-03-22")
    print(stats)


