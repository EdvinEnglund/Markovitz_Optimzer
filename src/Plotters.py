import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_frontier(frontier, optimal_p, default_p):
    volatility = np.array([math.sqrt(v) for (v, _, _) in frontier])
    returns = np.array([ret for (_, ret, _) in frontier])

    # Values to crop plot
    min_x = min(0, min(optimal_p[0], default_p[0]))
    min_y = min(0, min(optimal_p[1], default_p[1]))
    max_x = max(max(optimal_p[0], default_p[0]), volatility.max())
    max_y = max(max(optimal_p[1], default_p[1]), returns.max())

    if optimal_p[0] != 0 and default_p[0] !=0:
        min_sharpe = min((optimal_p[1]/optimal_p[0]), (default_p[1]/default_p[0]))
    else:
        min_sharpe = 0

    # Create grid for background gradient
    x = np.linspace(min_x, max_x, 200)
    y = np.linspace(min_y, max_y, 200)
    X, Y = np.meshgrid(x, y)

    # Avoid division by zero
    Z = np.divide(Y, X, out=np.zeros_like(Y), where=X != 0)

    # Cap sharpe ratio color map
    Z = np.minimum(Z, 5)
    Z = np.maximum(Z, min_sharpe)
    plt.figure()

    # Background heatmap (blue → yellow)
    contour = plt.contourf(X, Y, Z, levels=10, cmap="viridis")

    # Efficient frontier line in middle
    plt.plot(volatility, returns, color="white", linewidth=1, label=f"Efficient Frontier", zorder=2)

    # opt/user portfolio on top
    if optimal_p is not None:
        opx, opy = optimal_p
        plt.scatter(opx, opy, color="red", marker="x", s=100, linewidths=2, label="Tangent Portfolio", zorder=3)

    if default_p is not None:
        dpx, dpy = default_p
        plt.scatter(dpx, dpy, color="orange", marker="x", s=100, linewidths=2, label="Your Portfolio", zorder=3)

    # Labels and legend
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.legend()

    # Colorbar (Sharpe ratio index)
    cbar = plt.colorbar(contour)
    cbar.set_label("Sharpe Ratio")

    # Save to media/
    save = input("Do you want to save the frontier figure? (y/n): ")
    if save == "y":
        save_fig(plt, default_name = "efficient_frontier")
    plt.show()

def save_fig(fig, default_name):
    name = input(f"Enter file name to save (without extension) or press Enter for {default_name}.png: ").strip()
    file_name = name if name != "" else default_name
    root = Path(__file__).resolve().parents[1]
    save_path = root / "media" / f"{file_name}.png"
    fig.savefig(save_path)