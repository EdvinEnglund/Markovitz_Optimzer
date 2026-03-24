"""
This script calculates efficient frontier for alpha in steps of 0.05
sigma is covariance matrix of assets
mu is expected return of assets

returns plot of efficient frontier both in cases of shorting allowed / not allowed
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from Fetcher import set_tickers, get_stats, fetch_price_data
import math
from datetime import date
import pandas as pd


# Objective function for alpha, adjust for shorting cost
def F_alpha(w, mu, sigma, alpha, short_cost):
    fa = alpha * (w.T @ sigma @ w) - (1.0 - alpha) * (mu @ w)
    short_positions = np.abs(np.minimum(w, 0))
    short_penalty = np.sum(short_positions) * short_cost
    return fa + short_penalty

def assert_L(L):
    # L < 1 makes no sense
    if L < 1:
        L = 1
    if L == 1.0:
        # L = 1.0 breaks optimizer logic, increment float
        # ugly fix but works best
        L += 0.0001
    return L

def get_weights_by_alpha(mu, sigma, alpha, short_allowed = True, L = 1.5, short_cost = 0.0):
    L = assert_L(L)

    n = len(mu)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    # Adjust gross leverage

    cons.append({"type": "ineq", "fun": lambda w: L - np.sum(np.abs(w))})

    w0 = np.ones(n) / n
    if short_allowed:
        bounds = [(None, None)] * n
    else:
        bounds = [(0, None)] * n

    res = minimize(
        F_alpha,
        w0,
        args = (mu, sigma, alpha, short_cost),
        method='SLSQP',
        constraints=cons,
        bounds=bounds
    )
    return res.x

def calculate_efficient_frontier(mu, sigma, short_allowed = True, L = 1.5, short_cost = 0.0):
    res = []
    alphas = np.arange(0.05, 1.01, 0.05)
    for alpha in alphas:
        #shorting allowed
        w = get_weights_by_alpha(
            mu,
            sigma,
            alpha,
            short_allowed = short_allowed,
            L = L,
            short_cost = short_cost
        )
        res.append((
            w.T @ sigma @ w,
            mu@w,
            w
        ))

    return res

def plot_frontier(res, L):
    v_s = [math.sqrt(v) for (v, _, _) in res]
    r_s = [ret for (_, ret, _) in res]
    w_s = [w for (_,_,w) in res]

    plt.plot(v_s, r_s, label=f"L = {L}")
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    rf = 0.02
    short_allowed = True
    short_cost = 0.00
    L = 1.00

    #tickers = set_tickers()
    tickers = ["AAPL", "GOOG", "AMZN", "MSFT", "CCJ", "NVDA", "uuuu", "glo.to"]

    years = 1
    end_date = pd.to_datetime(today_str)
    start_date = end_date - pd.DateOffset(years=years)

    price_df = fetch_price_data(tickers, start=start_date, end=end_date)
    valid_tickers, mu, sigma = get_stats(price_df, "2026-03-22", years = 5)

    s = calculate_efficient_frontier(mu, sigma, L = L, short_cost = short_cost)
    plot_frontier(s, L = L)



