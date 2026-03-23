"""
This script calculates the weight distribution of the lowest variance
portfolio for a given expected return.

Inputs:
- sigma: covariance matrix of assets
- mu: expected return of assets
- p: chosen expected return (caps at highest returning asset times L)
- L: max gross leverage
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from Fetcher import set_tickers, get_stats, set_target_return, set_target_volatility, fetch_price_data
import math
from datetime import date

# Helper to cap expected return
def max_return(mu, L, short_allowed=True, short_cost = 0.0):
    if short_allowed:
        return L * max(mu) + (L - 1) * min(mu) - short_cost  # approximate max return w. shorting
    else:
        return max(mu)

def max_volatility(sigma, short_allowed=True, L=1.0):
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

# Objective function: variance + shorting cost
def neg_return(w, mu, rf, short_cost = 0.0):
    portfolio_return = np.dot(w, mu)
    short_penalty = short_cost * np.sum(np.maximum(-w, 0))
    return - (portfolio_return - short_penalty - rf)

def variance(w, sigma, rf, short_cost):
    variance = w.T @ sigma @ w
    shorting_penalty = short_cost * np.sum(np.maximum(-w, 0)**2)
    return variance + shorting_penalty

def neg_sharpe_ratio(w, mu, sigma, rf, short_cost):
    portfolio_return = np.dot(w, mu)
    portfolio_vol = np.sqrt(w @ sigma @ w)
    # shorting cost
    short_penalty = short_cost * np.sum(np.maximum(-w, 0))
    return - (portfolio_return - short_penalty - rf)/portfolio_vol

def get_constraints_and_bounds(n, L, short_allowed=True):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                   {"type": "ineq", "fun": lambda w: L - np.sum(np.abs(w))}]

    if short_allowed:
        bounds = [(None, None)] * n
    else:
        bounds = [(0, None)] * n
    return constraints, bounds

def get_weights_by_return(mu, sigma, target=0.10, short_allowed=True, L = 3.00, rf = 0.02, short_cost=0.0):
    #cap raturns
    target = min(target, max_return(mu, L, short_allowed=short_allowed, short_cost=short_cost))
    n = len(mu)
    constraints, bounds = get_constraints_and_bounds(n, L, short_allowed=short_allowed)
    # Add target constraint
    constraints += [{
        "type": "eq",
        "fun": lambda w: (np.dot(mu, w)
                - short_cost * np.sum(np.maximum(-w, 0))
                - target)
    }]
    #initial guess = equal weighting
    w0 = np.ones(n) / n
    res = minimize(
        variance,
        w0,
        args=(sigma, rf, short_cost),
        method = "SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w = res.x
    ret = res.fun
    return w

def get_weights_by_volatility(mu, sigma, target=0.10, short_allowed=True, L = 3.00, rf = 0.02, short_cost=0.0):
    # cap raturns
    target = min(target, max_volatility(sigma, short_allowed, L))
    n = len(mu)
    constraints, bounds = get_constraints_and_bounds(n, L, short_allowed=short_allowed)
    # Add target constraint
    constraints += [{
        "type": "eq",
        "fun": lambda w: (w.T @ sigma @ w - target**2)
    }]
    # initial guess = equal weighting
    w0 = np.ones(n) / n
    res = minimize(
        neg_return,
        w0,
        args=(mu, rf, short_cost),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w = res.x
    return w

def get_tangent_portfolio(mu, sigma, rf = 0.02, short_allowed=True, L = 3.00, short_cost=0.0):
    n = len(mu)
    constraints, bounds = get_constraints_and_bounds(n, L, short_allowed=short_allowed)
    # initial guess = equal weighting
    w0 = np.ones(n) / n
    res = minimize(
        neg_sharpe_ratio,
        w0,
        args=(mu, sigma, rf, short_cost),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w = res.x
    return w


if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    rf = 0.02
    short_allowed = True
    short_cost = 0.05
    L = 1.5

    #tickers = set_tickers()
    tickers = ["AAPL", "GOOG", "AMZN", "MSFT", "CCJ", "NVDA", "uuuu", "glo.to"]

    years = 1
    end_date = pd.to_datetime(today_str)
    start_date = end_date - pd.DateOffset(years=years)

    price_df = fetch_price_data(tickers, start=start_date, end=end_date)
    valid_tickers, mu, sigma = get_stats(price_df, "2026-03-22", years = 5)
    #target = set_target_volatility()
    w = get_tangent_portfolio(mu, sigma, rf=rf, short_allowed=True, L=L, short_cost=short_cost)
    ret = -neg_return(w, mu, rf, short_cost = short_cost)
    var =  w.T @ sigma @ w
    #print(f"Optimal weights for expected return {target}:")
    for i, ticker in enumerate(valid_tickers):
        print(f"{ticker}: {w[i]*100:.2f}%")
    print(f"Volatility: {math.sqrt(var)*100:.2f}%")
    print(f"Expected return: {ret*100:.2f}%")
