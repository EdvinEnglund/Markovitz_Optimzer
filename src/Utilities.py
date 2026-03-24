import numpy as np

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
def max_return(mu, L, short_allowed=True, short_cost = 0.0):
    if short_allowed:
        return L * max(mu) + (L - 1) * min(mu) - short_cost  # approximate max return w. shorting
    else:
        return max(mu)

# Helper to cap volatility
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

# Get constraints and bounds common for all optimizers
def get_constraints_and_bounds(n, L, short_allowed=True):
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                   {"type": "ineq", "fun": lambda w: L - np.sum(np.abs(w))}]

    if short_allowed:
        bounds = [(None, None)] * n
    else:
        bounds = [(0, None)] * n
    return constraints, bounds