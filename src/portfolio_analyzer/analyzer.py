import math
import pandas as pd
import numpy as np

from src.portfolio_analyzer import optimizer
from src.portfolio_analyzer import portfolio_tools
from src.portfolio_analyzer import utilities
from config import RF, SHORT_ALLOWED, DEFAULT_L, SHORT_COST, NO_O_RANDOM_POINTS
from src.portfolio_analyzer.portfolio_tools import normalize_portfolio


def get_tangent_portfolio(returns_df, rf = RF, short_allowed = SHORT_ALLOWED, L = DEFAULT_L, short_cost = SHORT_COST):
    mu, sigma = utilities.get_greeks(returns_df)
    weights = optimizer.get_optimal_weights(
        mu,
        sigma,
        rf = rf,
        short_allowed=short_allowed,
        L = L,
        short_cost=short_cost
    )
    portfolio = {}
    valid_tickers = list(returns_df.columns)
    for i, ticker in enumerate(valid_tickers):
        portfolio[ticker] = weights[i]
    return portfolio_tools.normalize_portfolio(portfolio)

def get_portfolio_stats(portfolio, returns_df, rf=RF, short_cost=SHORT_COST, ndigits=4):
    mu, sigma = utilities.get_greeks(returns_df)

    weights = np.array(list(normalize_portfolio(portfolio).values()))

    portfolio_return = weights @ mu
    short_penalty = utilities.get_short_penalty(weights, short_cost = short_cost)
    net_return = portfolio_return - short_penalty

    volatility = math.sqrt(weights.T @ sigma @ weights)
    sharpe = (net_return - rf) / volatility
    gross_leverage = np.sum(np.abs(weights))

    return pd.DataFrame([{
        "Expected Return": round(net_return, ndigits),
        "Volatility": round(volatility, ndigits),
        "Sharpe Ratio": round(sharpe, ndigits),
        "Weight": 1.0,
        "Gross Leverage": round(gross_leverage, ndigits)
    }], index=["Portfolio"])

def get_tickers_stats(portfolio, returns_df, rf = RF, ndigits=4):
    # --- Individual assets ---
    tickers = list(portfolio.keys())
    weights = list(portfolio.values())
    mu, sigma = utilities.get_greeks(returns_df)

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

def get_stats(portfolio, returns_df, rf = RF, short_cost = SHORT_COST, ndigits=4):
    portfolio_row = get_portfolio_stats(portfolio, returns_df, rf=rf, short_cost=short_cost, ndigits=ndigits)
    tickers_stats = get_tickers_stats(portfolio, returns_df, rf = rf, ndigits=ndigits)
    # Put portfolio first
    df = pd.concat([portfolio_row, tickers_stats])
    return df

def get_randomized_portfolio_points(portfolio, returns_df, n = NO_O_RANDOM_POINTS):
    random_vol_ret_points = []
    for _ in range(n):
        portfolio = portfolio_tools.randomize_portfolio(portfolio)
        stats = get_portfolio_stats(portfolio, returns_df)
        ret = stats.iloc[0]["Expected Return"]
        vol = stats.iloc[0]["Volatility"]
        point = (vol, ret)
        random_vol_ret_points.append(point)
    return random_vol_ret_points