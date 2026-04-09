import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

def plot_frontier(
        frontier,
        optimal_p = None,
        default_p = None,
        benchmark_p = None,
        random_points=None,
        show=True,
        save=False):
    volatility = np.array([v for (v, _) in frontier])
    returns = np.array([ret for (_, ret) in frontier])

    # Values to crop plot
    max_x = volatility.max()
    max_y = max(max(optimal_p[1], default_p[1]), returns.max())
    min_x = min(0, min(optimal_p[0], default_p[0]))
    min_y = min(0, min(optimal_p[1], default_p[1]))


    if optimal_p[0] != 0 and default_p[0] != 0:
        min_sharpe = min(0, min((optimal_p[1]/optimal_p[0]), (default_p[1]/default_p[0])))
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

    # Background heatmap
    contour = plt.contourf(X, Y, Z, levels=10, cmap="viridis")

    # Efficient frontier
    plt.plot(volatility, returns, color="white", linewidth=1, label="Efficient Frontier", zorder=2)

    # Optional random portfolios
    if random_points is not None and len(random_points) > 0:
        rand_vol = np.array([v for (v, _) in random_points])
        rand_ret = np.array([r for (_, r) in random_points])

        plt.scatter(
            rand_vol,
            rand_ret,
            color="white",
            s=15,
            alpha=0.7,
            label="Random Portfolios",
            zorder=1
        )

    # Optimal portfolio
    if optimal_p is not None:
        opx, opy = optimal_p
        plt.scatter(opx, opy, color="red", marker="x", s=100, linewidths=2,
                    label="Tangent Portfolio", zorder=3)

    # Default/user portfolio
    if default_p is not None:
        dpx, dpy = default_p
        dpx = min(dpx, max_x)
        plt.scatter(dpx, dpy, color="orange", marker="x", s=100, linewidths=2,
                    label="Your Portfolio", zorder=3)

    # Benchmark point
    if benchmark_p is not None:
        bpx, bpy = benchmark_p
        bpx = min(bpx, max_x)
        plt.scatter(bpx, bpy, color="blue", marker="x", s=100, linewidths=2,
                    label="Benchmark Portfolio", zorder=3)
    # Labels and legend
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.legend()

    # Colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label("Sharpe Ratio")

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    if save:
        save_fig(plt, file_name="efficient_frontier")

    if show:
        plt.show()
    else:
        plt.close()

def plot_heatmap(portfolio, sigma, show=True, save=False):
    if isinstance(portfolio, dict):
        portfolio = list(portfolio.keys())

    portfolio = list(portfolio)

    # Validate dimensions
    if sigma.shape[0] != len(portfolio):
        raise ValueError("sigma size must match portfolio length")

    cov_df = pd.DataFrame(sigma, index=portfolio, columns=portfolio)
    # Plot covariance heatmap
    fig, ax = plt.subplots()
    cax = ax.imshow(cov_df, interpolation='nearest')

    ax.set_xticks(range(len(portfolio)))
    ax.set_yticks(range(len(portfolio)))
    ax.set_xticklabels(portfolio, rotation=90)
    ax.set_yticklabels(portfolio)

    fig.colorbar(cax)
    ax.set_title("Covariance Heatmap", pad=15)

    if save:
        save_fig(fig, "covariance_heatmap")

    if show:
        plt.show()
    else:
        plt.close()

def plot_PCA(returns_df, save = False, show = True):
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(returns_df.T)  # Transpose: tickers as samples

    explained_var = pca.explained_variance_ratio_ * 100  # in percentage

    fig, ax = plt.subplots()

    # Scatter plot
    ax.scatter(pca_result[:, 0], pca_result[:, 1], color='blue')

    # Annotate points with ticker names
    for i, ticker in enumerate(returns_df.columns):
        ax.text(pca_result[i, 0] + 0.002, pca_result[i, 1] + 0.002, ticker, fontsize=9)

    # Labels including explained variance
    ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}% explained variance)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}% explained variance)")
    ax.set_title("PCA of daily log returns")

    ax.grid(True)
    plt.tight_layout()

    if save:
        save_fig(fig, "pca")
    if show:
        plt.show()
    else:
        plt.close()

    return fig

def save_fig(fig, file_name):
    root = Path(__file__).resolve().parents[2]
    base_path = root / "media"
    save_path = base_path / f"{file_name}.png"
    counter = 1
    while save_path.exists():
        save_path = base_path / f"{file_name}_{counter}.png"
        counter += 1
    fig.savefig(save_path)
    return save_path