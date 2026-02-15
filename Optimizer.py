"""
This script calculates the weight distribution of the lowest variance
portfolio for a given expected return.

Inputs:
- sigma: covariance matrix of assets
- mu: expected return of assets
- p: chosen expected return (caps at leverage)
L is leverage
"""
import numpy as np
from scipy.optimize import minimize
from Fetcher import set_tickers, get_stats
import math

# Helper to cap expected return
def max_return(mu, L, short_allowed=True, short_cost = 0.0):
    if short_allowed:
        return L * max(mu) + (L - 1) * min(mu) - short_cost  # approximate max return w. shorting
    else:
        return max(mu)

# Objective function: variance + shorting cost
def objective(w, sigma, short_cost):
    variance = w.T @ sigma @ w
    short_positions = np.abs(np.minimum(w, 0))
    shorting_penalty = np.sum(short_positions) * short_cost
    return variance + shorting_penalty

def get_weigths_by_p(mu, sigma, p=0.10, short_allowed=True, L = 1.00, short_cost=0.0):

    #cap raturns
    p = min(p, max_return(mu, L, short_allowed=short_allowed, short_cost=short_cost))

    n = len(mu)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                   {"type": "eq", "fun": lambda w: np.dot(mu, w)
                                                   - (np.sum(np.abs(np.minimum(w, 0))) * short_cost)
                                                   - p},
                   {"type": "ineq", "fun": lambda w: L - np.sum(np.abs(w))}]

    if short_allowed:
        bounds = [(None, None)] * n
    else:
        bounds = [(0, None)] * n
    #initial guess = equal weighting
    w0 = np.ones(n) / n
    res = minimize(
        objective,
        w0,
        args=(sigma, short_cost),
        method = "SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w = res.x
    return w
def set_expected_return():
    try:
        p = float(input("Choose expected return: "))
    except ValueError:
        print("Invalid input. Using default alpha = 0.1")
        p = 0.1
    if p < 0:
        p = 0
    print(f"Setting expected return to {p}")
    return p

if __name__ == "__main__":
    #tickers = set_tickers()
    tickers = ["AAPL", "GOOG", "AMZN", "MSFT", "CCJ", "NVDA", "uuuu", "glo.to"]
    valid_tickers, mu, sigma = get_stats(tickers, 5)
    p = set_expected_return()
    w = get_weigths_by_p(mu, sigma, p, short_allowed=True, L=3, short_cost=0.02)
    var =  w.T @ sigma @ w
    print(f"Optimal weights for expected return {p}:")
    for i, ticker in enumerate(valid_tickers):
        print(f"{ticker}: {w[i]*100:.2f}%")
    print(f"Volatility: {math.sqrt(var)*100:.2f}%")
