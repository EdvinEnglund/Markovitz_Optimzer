"""
This script calculates efficient frontier for alpha in steps of 0.05
sigma is covariance matrix of assets
mu is expected return of assets

returns plot of efficient frontier both in cases of shorting allowed / not allowed
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from Fetcher import set_tickers, get_stats
import math


# Objective function for alpha, adjust for shorting cost
def F_alpha(w, r, C, alpha, short_cost):
    fa = alpha * (w.T @ C @ w) - (1.0 - alpha) * (r @ w)
    short_positions = np.abs(np.minimum(w, 0))
    shorting_penalty = np.sum(short_positions) * short_cost
    return fa + shorting_penalty

def get_weights_by_alpha(r, C, alpha, short_allowed = True, L = 1.0, short_cost = 0.0):
    if L < 1:
        L = 1

    n = len(r)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    # Adjust gross leverage
    if L > 1.0:
        cons.append({"type": "ineq", "fun": lambda w: L - np.sum(np.abs(w))})

    w0 = np.ones(n) / n
    if short_allowed:
        bounds = [(None, None)] * n
    else:
        bounds = [(0, None)] * n

    res = minimize(
        F_alpha,
        w0,
        args = (r, C, alpha, short_cost),
        method='SLSQP',
        constraints=cons,
        bounds=bounds
    )

    return res.x

def calculate_efficient_frontier(mu, sigma, L =1.0, short_cost = 0.0):
    res_short = []
    res_no_short = []
    alphas = np.arange(0.05, 1.01, 0.05)
    for alpha in alphas:
        #shorting allowed
        short_w = get_weights_by_alpha(
            mu,
            sigma,
            alpha,
            short_allowed = True,
            L = L,
            short_cost = short_cost
        )
        res_short.append((
            short_w.T @ sigma @ short_w,
            mu@short_w,
            short_w
        ))

        #shorting disallowed
        no_short_w = get_weights_by_alpha(
            mu,
            sigma,
            alpha,
            short_allowed = False,
            L = L,
            short_cost = short_cost)
        res_no_short.append((
            no_short_w.T @ sigma @ no_short_w,
            mu @ no_short_w,
            no_short_w
        ))
    return res_short, res_no_short

def plot_frontier(res_short, res_no_short):
    v_s = [math.sqrt(v) for (v, _, _) in res_short]
    r_s = [ret for (_, ret, _) in res_short]
    w_s = [w for (_,_,w) in res_short]

    v_ns = [math.sqrt(v) for (v, _, _) in res_no_short]
    r_ns = [ret for (_, ret, _) in res_no_short]
    w_ns = [w for (_,_,w) in res_no_short]

    plt.plot(v_s, r_s, label="shorting allowed")
    plt.plot(v_ns, r_ns, label="shorting disallowed")
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #tickers = set_tickers()
    tickers = ["AAPL", "GOOG", "AMZN", "MSFT", "CCJ", "NVDA", "uuuu", "glo.to"]
    _, mu, sigma = get_stats(tickers, 5)
    short, no_short = calculate_efficient_frontier(mu, sigma, L = 1.15, short_cost = 0.01)

    plot_frontier(short, no_short)



