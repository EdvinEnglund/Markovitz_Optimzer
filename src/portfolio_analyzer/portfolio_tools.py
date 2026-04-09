import numpy as np
import random
import json
from config import DEFAULT_PORTFOLIO_PATH
from pathlib import Path

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
        name = input("Enter the name of your saved portfolio to save in data/ (press Enter to skip save): ")
        if name == "":
            print("Save skipped")
            return portfolio
        save_path = f"data/{name}.json"
        save_portfolio(portfolio, save_path)
    return portfolio

def get_portfolio(path = DEFAULT_PORTFOLIO_PATH):
    root = Path(__file__).resolve().parents[2]
    file_path = root / path
    with open(file_path, "r") as f:
        portfolio = json.load(f)
    return normalize_portfolio(portfolio)

def save_portfolio(portfolio, path):
    if path == "":
        print("Save skipped")
        return
    root = Path(__file__).resolve().parents[2]
    save_path = root / path
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

def randomize_portfolio(portfolio):
    port = {}
    if isinstance(portfolio, dict):
        for ticker, weights in portfolio.items():
            port[ticker] = random.random() - 0.5
    else:
        for ticker in portfolio:
            port[ticker] = random.random() - 0.5
    return normalize_portfolio(port)

def validate_portfolio(portfolio, valid_tickers):
    valid_portfolio = {k.upper(): v for k, v in portfolio.items() if k.upper() in set(valid_tickers)}
    return normalize_portfolio(valid_portfolio)