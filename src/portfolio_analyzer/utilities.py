import numpy as np
import pandas as pd
from config import RF, SHORT_COST

def get_greeks(returns):
    mu = (returns.mean() * 252).values
    sigma = (returns.cov() * 252).values
    return mu, sigma

def get_short_penalty(weights, short_cost=SHORT_COST):
    return short_cost * np.sum(np.maximum(-weights, 0))

def neg_return(w, mu, rf = RF, short_cost = SHORT_COST):
    """
    Objective function for portfolio_analyzer based on target volatility.
    Returns negative of expected return from weight vector w and expected return vector mu.
    Adjusts for interest rate and shorting costs.
    """
    portfolio_return = np.dot(w, mu)
    short_penalty = get_short_penalty(w, short_cost = short_cost)
    return - (portfolio_return - short_penalty - rf)

def variance(w, sigma):
    """
    Objective function for portfolio_analyzer based on target return
    Returns variance of expected return from weight vector w and covariance matrix sigma
    Adjusts portfolio_analyzer for shorting costs.
    """
    var = w.T @ sigma @ w
    return var

def neg_sharpe_ratio(w, mu, sigma, rf = RF, short_cost = SHORT_COST):
    """
    Objective function for portfolio_analyzer for tangent portfolio
    Returns sharepe ratio of expected return from weight vector w, returns vector mu and covariance matrix sigma
    """
    portfolio_return = np.dot(w, mu)
    portfolio_vol = np.sqrt(w @ sigma @ w)
    # shorting cost
    short_penalty = get_short_penalty(w, short_cost = short_cost)
    return - (portfolio_return - short_penalty - rf)/portfolio_vol

def get_mean_portfolio_stats(points, ndigits = 4):
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


