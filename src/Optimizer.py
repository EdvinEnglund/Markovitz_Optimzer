from scipy.optimize import minimize
from src.Utilities import *

def neg_return(w, mu, rf, short_cost = 0.0):
    """
    Objective function for optimizer based on target volatility.
    Returns negative of expected return from weight vector w and expected return vector mu.
    Adjusts for interest rate and shorting costs.
    """
    portfolio_return = np.dot(w, mu)
    short_penalty = short_cost * np.sum(np.maximum(-w, 0))
    return - (portfolio_return - short_penalty - rf)

def variance(w, sigma, short_cost):
    """
    Objective function for optimizer based on target return
    Returns variance of expected return from weight vector w and covariance matrix sigma
    Adjusts optimizer for shorting costs.
    """
    variance = w.T @ sigma @ w
    short_penalty = short_cost * np.sum(np.maximum(-w, 0)**2)
    return variance + short_penalty

def neg_sharpe_ratio(w, mu, sigma, rf, short_cost):
    """
    Objective function for optimizer for tangent portfolio
    Returns sharepe ratio of expected return from weight vector w, returns vector mu and covariance matrix sigma
    """
    portfolio_return = np.dot(w, mu)
    portfolio_vol = np.sqrt(w @ sigma @ w)
    # shorting cost
    short_penalty = short_cost * np.sum(np.maximum(-w, 0))
    return - (portfolio_return - short_penalty - rf)/portfolio_vol

def get_weights_by_target_return(mu, sigma, target=0.10, short_allowed=True, L = 3.00, rf = 0.02, short_cost=0.0):
    L = assert_L(L)
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
        args=(sigma, short_cost),
        method = "SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w = res.x
    return w

def get_weights_by_target_volatility(mu, sigma, target=0.10, short_allowed=True, L = 3.00, rf = 0.02, short_cost=0.0):
    L = assert_L(L)
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
    L = assert_L(L)
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
