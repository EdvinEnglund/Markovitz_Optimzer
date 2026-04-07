import numpy as np
import pandas as pd
from config import SHORT_ALLOWED, SHORT_COST, DEFAULT_L, YEARS

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

def get_expected_returns(returns):
    """
    Takes: Pandas DataFrame price_data with daily log returns.
    Returns: np.Array of expected yearly returns
    """
    mu = (returns.mean() * 252).values
    return mu

def get_covariance_matrix(returns):
    """
    Takes: Pandas DataFrame price_data with daily log returns.
    Returns: np.2DArray covariance matrix of daily returns (i.e. log returns)
    """
    sigma = (returns.cov() * 252).values
    return sigma

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

def get_tangent_portfolio(valid_tickers, weights):
    portfolio = {}
    for i, ticker in enumerate(valid_tickers):
        portfolio[ticker] = weights[i]
    return normalize_portfolio(portfolio)

def validate_portfolio(portfolio, valid_tickers):
    valid_portfolio = {k: v for k, v in portfolio.items() if k.upper() in set(valid_tickers)}
    return normalize_portfolio(valid_portfolio)

# Helper to assert valid L
def assert_L(L):
    # L < 1 makes no sense
    if L < 1:
        L = 1
    if L == 1.0:
        # L = 1.0 breaks optimizer logic, increment float
        # ugly fix but works best
        L += 0.0001
    return L

# Helper to cap max return
def max_return(mu, L = DEFAULT_L, short_allowed = SHORT_ALLOWED, short_cost = SHORT_COST):
    if short_allowed:
        return L * max(mu) + (L - 1) * min(mu) - short_cost  # approximate max return w. shorting
    else:
        return max(mu)

# Helper to cap volatility
def max_volatility(sigma, short_allowed = SHORT_ALLOWED, L = DEFAULT_L):
    vols = np.sqrt(np.diag(sigma))
    if not short_allowed:
        return np.max(vols)
    # pick two most "extreme" assets
    i = np.argmax(vols)
    j = np.argmin(vols)

    w = np.zeros(len(vols))
    w[i] = L
    w[j] = -(L - 1)
    return np.sqrt(w @ sigma @ w)

# Get constraints and bounds common for all optimizers
def get_constraints_and_bounds(n, L = DEFAULT_L, short_allowed = SHORT_ALLOWED):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                   {"type": "ineq", "fun": lambda w: L - np.sum(np.abs(w))}]

    if short_allowed:
        bounds = [(None, None)] * n
    else:
        bounds = [(0, None)] * n
    return constraints, bounds