from scipy.optimize import minimize
import numpy as np
import math
# LOAD CONFIG CONSTANTS
from config import RF, SHORT_ALLOWED, SHORT_COST, DEFAULT_L, RETURN_CAP

# --- HELPERS ---
# Helper to assert valid L
def assert_L(L):
    # L < 1 makes no sense
    if L < 1:
        L = 1
    if L == 1.0:
        # L = 1.0 breaks optimizer logic, increment float
        # ugly fix but works best
        L += 0.0001
    return L

# Helper to cap max return
def max_return(mu, L = DEFAULT_L, short_allowed = SHORT_ALLOWED, short_cost = SHORT_COST):
    if short_allowed:
        return L * max(mu) + (L - 1) * min(mu) - short_cost  # approximate max return w. shorting
    else:
        return max(mu)

# Helper to cap volatility
def max_volatility(sigma, short_allowed = SHORT_ALLOWED, L = DEFAULT_L):
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

# Get constraints and bounds common for all optimizers
def get_constraints_and_bounds(n, L = DEFAULT_L, short_allowed = SHORT_ALLOWED):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                   {"type": "ineq", "fun": lambda w: L - np.sum(np.abs(w))}]

    if short_allowed:
        bounds = [(None, None)] * n
    else:
        bounds = [(0, None)] * n
    return constraints, bounds

# --- OBJECTIVE FUNCTIONS FOR OPTIMIZERS ---
def neg_return(w, mu, rf = RF, short_cost = SHORT_COST):
    """
    Objective function for optimizer based on target volatility.
    Returns negative of expected return from weight vector w and expected return vector mu.
    Adjusts for interest rate and shorting costs.
    """
    portfolio_return = np.dot(w, mu)
    short_penalty = short_cost * np.sum(np.maximum(-w, 0))
    return - (portfolio_return - short_penalty - rf)

def variance(w, sigma, short_cost = SHORT_COST):
    """
    Objective function for optimizer based on target return
    Returns variance of expected return from weight vector w and covariance matrix sigma
    Adjusts optimizer for shorting costs.
    """
    var = w.T @ sigma @ w
    short_penalty = short_cost * np.sum(np.maximum(-w, 0)**2)
    return var + short_penalty

def neg_sharpe_ratio(w, mu, sigma, rf = RF, short_cost = SHORT_COST):
    """
    Objective function for optimizer for tangent portfolio
    Returns sharepe ratio of expected return from weight vector w, returns vector mu and covariance matrix sigma
    """
    portfolio_return = np.dot(w, mu)
    portfolio_vol = np.sqrt(w @ sigma @ w)
    # shorting cost
    short_penalty = short_cost * np.sum(np.maximum(-w, 0))
    return - (portfolio_return - short_penalty - rf)/portfolio_vol

# Objective function for alpha, adjust for shorting cost
def F_alpha(w, mu, sigma, alpha, short_cost, rf = RF):
    short_positions = np.abs(np.minimum(w, 0))
    short_penalty = np.sum(short_positions) * short_cost
    fa = alpha * (w.T @ sigma @ w) - (1.0 - alpha) * ((mu @ w) - short_penalty - rf) #
    return fa

# --- OPTIMIZERS ---
def get_weights_by_target_return(mu,
                                 sigma,
                                 target=0.10,
                                 short_allowed = SHORT_ALLOWED,
                                 L = DEFAULT_L,
                                 rf = RF,
                                 short_cost = SHORT_COST):
    safe_L = assert_L(L)

    #cap raturns
    target = min(target, max_return(mu, L = safe_L, short_allowed=short_allowed, short_cost=short_cost))
    n = len(mu)
    constraints, bounds = get_constraints_and_bounds(n, L = safe_L, short_allowed=short_allowed)
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

def get_optimal_weights(mu, sigma, rf = RF, short_allowed = SHORT_ALLOWED, L = DEFAULT_L, short_cost = SHORT_COST):
    safe_L = assert_L(L)
    n = len(mu)
    constraints, bounds = get_constraints_and_bounds(n, L = safe_L, short_allowed=short_allowed)
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

def get_weights_by_alpha(mu, sigma, alpha, short_allowed = True, L = DEFAULT_L, short_cost = SHORT_COST):
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

def get_efficient_frontier(
        mu,
        sigma,
        short_allowed = True,
        rf = RF,
        L = DEFAULT_L,
        short_cost = SHORT_COST,
        return_cap = RETURN_CAP
    ):
    res = []
    alphas = np.arange(0.01, 1.01, 0.01)[::-1]
    for alpha in alphas:
        w = get_weights_by_alpha(
            mu,
            sigma,
            alpha,
            short_allowed = short_allowed,
            L = L,
            short_cost = short_cost
        )
        vol = math.sqrt(w.T @ sigma @ w)
        short_positions = np.abs(np.minimum(w, 0))
        short_penalty = np.sum(short_positions) * short_cost
        ret = mu@w - short_penalty - rf
        res.append((
            vol,
            ret
        ))
        if ret > return_cap: break
    return res


