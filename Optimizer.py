"""
This script calculates the weight distribution
sigma is covariance matrix of assets
mu is expected return of assets
p is chosen expected return
"""
import numpy as np
from scipy.optimize import minimize
from Fetcher import set_tickers, get_stats
import math

def objective(w, sigma):
    return w.T @ sigma @ w

def get_weigths_by_p(mu, sigma, p=0.10, short_allowed=False):

    # Define objective function within scope of sigma
    def objective(w): return w.T @ sigma @ w

    n = len(mu)

    # --- MUST FIX CONSTRAINTS SO WEIGHTS SUM TO 1 WHEN SHORTING ALLOWED
    constraints = [{"type": "eq", "fun":lambda w: np.sum(w) - 1},
                   {"type": "eq", "fun": lambda w: np.dot(mu, w) - p}]

    if short_allowed:
        bounds = [(-1, 1)] * n
    else:
        bounds = [(0, 1)] * n
    #initial guess = equal weighting
    w0 = np.ones(n) / n
    res = minimize(
        objective,
        w0,
        method = "SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w = res.x
    var = objective(w)
    return w, var
def set_expected_return():
    try:
        p = float(input("choose expected return [0-1]: "))
    except ValueError:
        print("Invalid input. Using default alpha = 0.1")
        p = 0.1
    if p > 1:
        p = 1
    if p < 0:
        p = 0
    print(f"setting expected return to {p}")
    return p

if __name__ == "__main__":
    tickers = set_tickers()
    valid_tickers, mu, sigma = get_stats(tickers, 5)
    p = set_expected_return()
    w, var = get_weigths_by_p(mu, sigma, p, short_allowed=True)

    print(f"Optimal weights for expected return {p}:")
    for i, ticker in enumerate(valid_tickers):
        print(f"{ticker}: {w[i]*100:.2f}%")
    print(f"Volatility: {math.sqrt(var)*100:.2f}%")
