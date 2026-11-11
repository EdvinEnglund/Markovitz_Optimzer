# Markovitz Optimizer

A Python project for exploring Markowitz mean–variance portfolio optimization. It downloads historical market prices, estimates portfolio risk and return, and compares an existing allocation with a tangency portfolio that maximizes the estimated Sharpe ratio.

The project is an experimental analysis toolkit under active development. The main analysis workflow is implemented; historical backtesting and validation are still planned. Results currently describe the data used to fit the portfolio, rather than out-of-sample performance.

## Functionality

- **Portfolio management:** enter ticker–weight pairs interactively or load and save JSON portfolios. Weights are normalized to sum to one, and tickers with no downloaded price data are removed.
- **Market data:** fetch adjusted closing prices from Yahoo Finance through `yfinance` and calculate daily log returns over a configurable lookback period.
- **Portfolio statistics:** report annualized expected return, volatility, Sharpe ratio, asset weights, and gross leverage, both for the supplied portfolio and the optimized allocation.
- **Optimization:** use SciPy's SLSQP solver to maximize Sharpe ratio, minimize variance at a target return, maximize return at a target volatility, or vary the risk–return tradeoff. The target-return and target-volatility routines are available through the Python API.
- **Constraints:** support long-only or long/short optimization, a gross leverage cap, and a uniform cost on short positions.
- **Comparisons:** optionally include a benchmark portfolio and generate random portfolio points for visual comparison. The bundled benchmark holds SPY; the default portfolio equally weights 11 sector ETFs.
- **Visualization:** display and optionally save an efficient-frontier approximation, a covariance heatmap, and a two-component PCA plot of asset return histories.

## How the analysis works

1. Load or create a portfolio and download prices for its tickers.
2. Calculate daily log returns, dropping dates with missing return values.
3. Estimate annualized mean returns and covariance by multiplying their daily estimates by 252.
4. Optimize weights subject to `sum(weights) = 1` and `sum(abs(weights)) <= L`. Long-only optimization also requires nonnegative weights.
5. Print statistics and plot the supplied, optimized, random, and optional benchmark portfolios.

The short-position penalty is `short_cost * sum(max(-weight, 0))`, deducted from estimated return. The main portfolio Sharpe calculation subtracts the risk-free rate before dividing by volatility. The frontier is sampled by solving a series of weighted variance-versus-return objectives.

## Setup

Use **Python 3.12 or newer**: the current analysis script uses f-string syntax introduced in Python 3.12. Run the following commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install scikit-learn
```

`scikit-learn` is required by the PCA plotting module but is not yet listed in `requirements.txt`. Price downloads require an internet connection. Interactive plots require a graphical Matplotlib backend.

## Usage

Use `scripts.analyze_portfolio` as the main entry point, running it as a module so the repository imports resolve correctly.

Analyze the bundled portfolio and compare it with the bundled benchmark:

```bash
python -m scripts.analyze_portfolio --portfolio data/default_portfolio.json --benchmark
```

Create a portfolio interactively:

```bash
python -m scripts.analyze_portfolio --save-portfolio
```

Enter each ticker and its weight or holding value, then submit an empty ticker to finish. An empty portfolio selects the bundled default. Saving prompts for a filename under `data/`.

Change the lookback and optimization settings, and save the figures:

```bash
python -m scripts.analyze_portfolio --portfolio data/default_portfolio.json --years 3 --rf 0.02 --L 2 --short_cost 0.01 --save-plots
```

Other useful options:

| Option | Behavior |
| --- | --- |
| `--save-optimal` | Prompt for a filename and save the optimized portfolio in `data/`. |
| `--no-plots` | Skip displaying and saving figures. |
| `--no-stats` | Suppress supplied, optimized, and benchmark statistics; the random-portfolio summary is still printed. |
| `--benchmark` | Enable benchmark comparison; off by default in the CLI. |
| `--help` | List command-line arguments. |

Figures are saved under `media/` as `efficient_frontier.png`, `covariance_heatmap.png`, and `pca.png`. Numbered filenames prevent existing figures from being overwritten. Saving figures currently requires plotting to remain enabled.

### Portfolio format

A portfolio is a JSON object mapping ticker symbols to numeric weights or holding values:

```json
{
  "XLK": 0.5,
  "XLF": 0.3,
  "XLV": 0.2
}
```

Positive weights represent long positions; negative weights represent shorts. Use a nonzero net total for meaningful normalization. The current implementation replaces zero-sum inputs with equal weights.

### Configuration

Defaults live in `config.py`; rates and returns use decimal units (`0.02` means 2%).

| Setting | Default | Meaning |
| --- | --- | --- |
| `YEARS` | `1` | Historical estimation window in years. |
| `RF` | `0.02` | Annual risk-free rate used in Sharpe calculations. |
| `SHORT_ALLOWED` | `True` | Allow negative optimized weights. |
| `DEFAULT_L` | `5` | Gross leverage cap: total absolute portfolio weight. |
| `SHORT_COST` | `0.0` | Uniform annual cost per unit of short exposure. |
| `RETURN_CAP` | `2` | Stop frontier sampling after estimated return exceeds this value. |
| `NO_O_RANDOM_POINTS` | `300` | Number of random portfolio points. |
| `DEFAULT_PORTFOLIO_PATH` | `data/default_portfolio.json` | Fallback portfolio. |
| `BENCHMARK_PORTFOLIO_PATH` | `data/benchmark_portfolio.json` | Default benchmark. |

For long-only analysis, set `SHORT_ALLOWED = False` in `config.py`, or call `scripts.analyze_portfolio.main(..., short_allowed=False)` from Python. The CLI currently has only an enabling `--short_allowed` flag, so it cannot turn off a `True` configuration default.

## Repository structure

```text
config.py                       Shared defaults
requirements.txt                Pinned dependencies (PCA dependency still missing)
scripts/analyze_portfolio.py    Main CLI and callable analysis workflow
src/portfolio_analyzer/
    fetcher.py                  Price downloads and log returns
    portfolio_tools.py          Portfolio input, JSON persistence, normalization
    utilities.py                Annualized estimates and objective helpers
    optimizer.py                Constrained optimizers and frontier sampling
    analyzer.py                 Portfolio statistics and comparisons
    plotters.py                 Frontier, covariance, and PCA figures
data/                           Saved portfolios
media/                          Saved plots
notebooks/empty.ipynb            Placeholder for future notebook analysis
TODO.txt                        Development backlog
```

## Progress and remaining work

The core data-fetching, optimization, statistics, JSON persistence, and visualization modules are implemented and connected through `scripts/analyze_portfolio.py`. Example portfolios and saved figures are included. Other scripts (`analyzer_script.py`, `create_and_analyze_portfolio.py`, and `portfolio_builder.py`) contain outdated or unresolved references and need refactoring before use.

Current limitations to address:

- Optimizers return solver weights without checking convergence or reporting failed constraints.
- Statistics rely on portfolio weights having the same ticker order as the return-data columns; explicit alignment is still needed.
- Random portfolios do not enforce the optimizer's shorting and leverage constraints. Their summary and the frontier background use return divided by volatility without subtracting the risk-free rate.
- The CLI parses `--benchmark_path` but does not forward it to the workflow. Change the configuration or pass `benchmark_path` directly to `main()` for a custom benchmark.
- Custom risk-free rates and short costs are not propagated consistently to every comparison statistic.
- No automated test suite or completed backtesting notebook is included yet.

The backlog in [TODO.txt](TODO.txt) calls for:

- Cleaning up modules, scripts, comments, and the data/output directory layout.
- Saving and loading price data as CSV for repeatable analyses.
- Consolidating a parameterized pipeline that accepts a ticker list, and adding an analysis notebook.
- Backtesting tangency portfolios against an index and equal-weight portfolios across leverage limits, shorting settings, estimation windows, and evaluation horizons.
- Reviewing short-cost treatment and return/covariance estimation, including dividend handling beyond the current adjusted-price downloads.
- Optionally supporting per-asset shortability and borrowing fees.
