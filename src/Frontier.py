"""
This script calculates efficient frontier for alpha in steps of 0.05
sigma is covariance matrix of assets
mu is expected return of assets

returns plot of efficient frontier both in cases of shorting allowed / not allowed
"""
from src.Optimizer import *
from src.Plotters import *
import pandas as pd

def F_alpha(w, mu, sigma, alpha, short_cost):
    fa = alpha * (w.T @ sigma @ w) - (1.0 - alpha) * (mu @ w)
    short_positions = np.abs(np.minimum(w, 0))
    short_penalty = np.sum(short_positions) * short_cost
    return fa + short_penalty

# Objective function for alpha, adjust for shorting cost
def get_weights_by_alpha(mu, sigma, alpha, short_allowed = True, L = 1.5, short_cost = 0.0):
    L = assert_L(L)

    n = len(mu)

    constraints, bounds = get_constraints_and_bounds(n, L, short_allowed=short_allowed)

    w0 = np.ones(n) / n

    res = minimize(
        F_alpha,
        w0,
        args = (mu, sigma, alpha, short_cost),
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )
    return res.x

def get_efficient_frontier(mu, sigma, short_allowed = True, L = 1.5, short_cost = 0.0):
    res = []
    alphas = np.arange(0.05, 1.01, 0.05)
    for alpha in alphas:
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


if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    rf = 0.02
    short_allowed = True
    short_cost = 0.00
    L = 1.2

    #tickers = set_tickers()
    tickers = ["AAPL", "GOOG", "AMZN", "MSFT", "CCJ", "NVDA", "uuuu", "glo.to"]

    years = 1
    end_date = pd.to_datetime(today_str)
    start_date = end_date - pd.DateOffset(years=years)

    price_df = fetch_price_data(tickers, start=start_date, end=end_date)
    valid_tickers, mu, sigma = get_stats(price_df, "2026-03-22", years = 5)

    s = get_efficient_frontier(mu, sigma, L = L, short_cost = short_cost)
    plot_frontier(s, L = L)



