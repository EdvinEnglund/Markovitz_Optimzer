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


# Objective function for alpha
def F_alpha(w, r, C, alpha):
    return alpha * (w.T @ C @ w) - (1.0 - alpha) * (r @ w)

def find_w(r, C, alpha, short_allowed = True):

    n = len(r)

    # --- MUST FIX CONSTRAINTS SO WEIGHTS SUM TO 1 WHEN SHORTING ALLOWED
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0 = np.ones(n) / n

    # The problem does not say anything about using leverage, so the bounds are capped at (-1,1) and (1,1)
    if short_allowed:
        bounds = [(-1, 1)] * n
    else:
        bounds = [(0, 1)] * n

    res = minimize(
        F_alpha,
        w0,
        args = (r, C, alpha),
        method='SLSQP',
        constraints=cons,
        bounds=bounds
    )

    return res.x

def calculate_efficient_frontier(mu, sigma):
    res_short = []
    res_no_short = []
    alphas = np.arange(0.05, 1.01, 0.05)
    for alpha in alphas:
        #shorting allowed
        short_w = find_w(mu, sigma, alpha, short_allowed = True)
        res_short.append((
            short_w.T @ sigma @ short_w,
            mu@short_w,
            short_w
        ))

        #shorting disallowed
        no_short_w = find_w(mu, sigma, alpha, short_allowed = False)
        res_no_short.append((
            no_short_w.T @ sigma @ no_short_w,
            mu @ no_short_w,
            no_short_w
        ))
    return res_short, res_no_short

def plot_frontier(res_short, res_no_short):
    v_s = [v for (v, _, _) in res_short]
    r_s = [ret for (_, ret, _) in res_short]
    w_s = [w for (_,_,w) in res_short]

    v_ns = [v for (v, _, _) in res_no_short]
    r_ns = [ret for (_, ret, _) in res_no_short]
    w_ns = [w for (_,_,w) in res_no_short]

    plt.plot(v_s, r_s, label="shorting allowed")
    plt.plot(v_ns, r_ns, label="shorting disallowed")
    plt.xlabel("Variance")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    tickers = set_tickers()
    _, mu, sigma = get_stats(tickers, 5)
    short, no_short = calculate_efficient_frontier(mu, sigma)
    plot_frontier(short, no_short)



