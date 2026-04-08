import numpy as np
import pandas as pd
import math
from src import Optimizer
import random
from config import YEARS, RF, NO_O_RANDOM_POINTS, SHORT_COST


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
    """
    mu = (returns.mean() * 252).values
    return mu

def get_covariance_matrix(returns):
    """
    Takes: Pandas DataFrame price_data with daily log returns.
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

def get_tangent_portfolio(returns_df):
    mu = get_expected_returns(returns_df)
    sigma = get_covariance_matrix(returns_df)
    weights = Optimizer.get_optimal_weights(mu, sigma)
    portfolio = {}
    valid_tickers = list(returns_df.columns)
    for i, ticker in enumerate(valid_tickers):
        portfolio[ticker] = weights[i]
    return normalize_portfolio(portfolio)

def randomize_portfolio(portfolio):
    port = {}
    if isinstance(portfolio, dict):
        for ticker, weights in portfolio.items():
            port[ticker] = random.random() - 0.5
    else:
        for ticker in portfolio:
            port[ticker] = random.random() - 0.5
    return normalize_portfolio(port)

def get_randomized_portfolios(portfolio, returns_df, n = NO_O_RANDOM_POINTS):
    random_vol_ret_points = []
    for _ in range(n):
        portfolio = randomize_portfolio(portfolio)
        stats = get_portfolio_stats(portfolio, returns_df)
        ret = stats.iloc[0]["Expected Return"]
        vol = stats.iloc[0]["Volatility"]
        point = (vol, ret)
        random_vol_ret_points.append(point)
    return random_vol_ret_points

def get_mean_random_portfolio_stats(points, ndigits = 4):
    n = len(points)
    mean_vol = sum(vol for (vol, ret) in points) / n
    mean_ret = sum(ret for (vol, ret) in points) / n
    mean_sharpe = sum(ret / vol for (vol, ret) in points) / n

    # --- Portfolio row ---
    row = pd.DataFrame([{
        "Expected Return": round(mean_ret, ndigits),
        "Volatility": round(mean_vol, ndigits),
        "Sharpe Ratio": round(mean_sharpe, ndigits),
        "Weight": 1.0,
        "Gross Leverage": None
    }], index=["Portfolio"])
    return row


def validate_portfolio(portfolio, valid_tickers):
    valid_portfolio = {k.upper(): v for k, v in portfolio.items() if k.upper() in set(valid_tickers)}
    return normalize_portfolio(valid_portfolio)

# argument portfolio must match daily return data
def get_portfolio_stats(portfolio, returns_df, rf = RF, ndigits=4):
    mu = get_expected_returns(returns_df)
    sigma = get_covariance_matrix(returns_df)
    weights = np.array(list(portfolio.values()))
    expected_return = -Optimizer.neg_return(weights, mu)
    volatility = math.sqrt(weights.T @ sigma @ weights)
    sharpe = -Optimizer.neg_sharpe_ratio(weights, mu, sigma)
    gross_leverage = np.sum(np.abs(weights))

    # --- Portfolio row ---
    portfolio_row = pd.DataFrame([{
        "Expected Return": round(expected_return, ndigits),
        "Volatility": round(volatility, ndigits),
        "Sharpe Ratio": round(sharpe, ndigits),
        "Weight": 1.0,
        "Gross Leverage": round(gross_leverage, ndigits)
    }], index=["Portfolio"])

    # Put portfolio first
    return portfolio_row

def get_tickers_stats(portfolio, returns_df, rf = RF, ndigits=4):
    # --- Individual assets ---
    tickers = list(portfolio.keys())
    weights = list(portfolio.values())
    mu = get_expected_returns(returns_df)
    sigma = get_covariance_matrix(returns_df)

    data = []
    for i, ticker in enumerate(tickers):
        asset_return = mu[i]
        asset_vol = math.sqrt(sigma[i,i])
        asset_sharpe = (asset_return - rf) / asset_vol

        data.append({
            "Ticker": ticker,
            "Expected Return": round(asset_return, ndigits),
            "Volatility": round(asset_vol, ndigits),
            "Sharpe Ratio": round(asset_sharpe, ndigits),
            "Weight": round(weights[i], ndigits),
            "Gross Leverage": None,
        })
    df = pd.DataFrame(data).set_index("Ticker")
    return df

def get_stats(portfolio, returns_df, rf = RF, ndigits=4):
    portfolio_row = get_portfolio_stats(portfolio, returns_df, rf, ndigits)
    tickers_stats = get_tickers_stats(portfolio, returns_df, rf, ndigits)
    # Put portfolio first
    df = pd.concat([portfolio_row, tickers_stats])
    return df

