import matplotlib.pyplot as plt
import math

import math
import numpy as np
import matplotlib.pyplot as plt

def plot_frontier(res, L, optimal_p):
    volatility = np.array([math.sqrt(v) for (v, _, _) in res])
    returns = np.array([ret for (_, ret, _) in res])

    # Create grid for background gradient
    x = np.linspace(0, volatility.max(), 200)
    y = np.linspace(0, returns.max(), 200)
    X, Y = np.meshgrid(x, y)

    # Avoid division by zero
    Z = np.divide(Y, X, out=np.zeros_like(Y), where=X != 0)

    # Cap sharpe ratio at 5
    Z = np.minimum(Z, 5)
    plt.figure()

    # Background heatmap (blue → yellow)
    contour = plt.contourf(X, Y, Z, levels=10, cmap="viridis")

    # Efficient frontier line on top
    plt.plot(volatility, returns, color="red", linewidth=1, label=f"Efficient Frontier", zorder=2)

    if optimal_p is not None:
        px, py = optimal_p
        plt.scatter(px, py, color="red", marker="x", s=100, linewidths=2, label="Tangent Portfolio", zorder=3)

    # Labels and legend
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.legend()

    # Colorbar (Sharpe ratio index)
    cbar = plt.colorbar(contour)
    cbar.set_label("Sharpe Ratio")

    plt.show()