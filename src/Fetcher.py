import yfinance as yf
import json
from src import Utilities
from pathlib import Path
from config import DEFAULT_PORTFOLIO_PATH

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

def create_portfolio(default_portfolio_path = DEFAULT_PORTFOLIO_PATH, save = False):
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

    if len(portfolio) == 0:
        print(f"No portfolio created, using default portfolio at {default_portfolio_path}")
        portfolio = get_portfolio(DEFAULT_PORTFOLIO_PATH)
        return portfolio

    if save:
        name = input("Enter the name of your saved portfolio to save (press Enter to skip save): ")
        if name == "":
            print("Save skipped")
            return portfolio
        save_path = f"data/{name}.json"
        save_portfolio(portfolio, save_path, default_path=DEFAULT_PORTFOLIO_PATH)

    return portfolio

"""
def get_portfolio(default_portfolio_path):
    root = Path(__file__).resolve().parents[1]
    file_path = root / default_portfolio_path
    with open(file_path, "r") as f:
        portfolio = json.load(f)
    return normalize_portfolio(portfolio)
"""


def get_portfolio(path = DEFAULT_PORTFOLIO_PATH):
    root = Path(__file__).resolve().parents[1]
    file_path = root / path
    with open(file_path, "r") as f:
        portfolio = json.load(f)
    return Utilities.normalize_portfolio(portfolio)

def save_portfolio(portfolio, path , default_path = DEFAULT_PORTFOLIO_PATH):
    if path == "":
        path = default_path
    root = Path(__file__).resolve().parents[1]
    save_path = root / path
    try:
        with open(save_path, "w") as f:
            json.dump(Utilities.normalize_portfolio(portfolio), f, indent=4)
        print(f"Saved portfolio to {save_path}")
    except Exception as e:
        print(f"Failed to save file: {e}")
    return



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





