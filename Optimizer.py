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
def max_return(mu, L, short_allowed=True):
    if short_allowed:
        return L * max(mu) + (L - 1) * min(mu)  # approximate max return w. shorting
    else:
        return max(mu)

# Helper function for variance
def objective(w, sigma):
    return w.T @ sigma @ w

def get_weigths_by_p(mu, sigma, p=0.10, short_allowed=True, L = 1.00):

    p = min(p, max_return(mu, L, short_allowed=short_allowed))

    n = len(mu)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                   {"type": "eq", "fun": lambda w: np.dot(mu, w) - p},
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
        args=(sigma),
        method = "SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w = res.x
    var = objective(w, sigma)
    return w, var
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
    w, var = get_weigths_by_p(mu, sigma, p, short_allowed=True, L=1.0)

    print(f"Optimal weights for expected return {p}:")
    for i, ticker in enumerate(valid_tickers):
        print(f"{ticker}: {w[i]*100:.2f}%")
    print(f"Volatility: {math.sqrt(var)*100:.2f}%")
