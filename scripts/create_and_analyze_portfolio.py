from src import Fetcher
from src import Optimizer
from src import Utilities
from src import Plotters
from src import Frontier
from datetime import date
import pandas as pd
import numpy as np
import math
from config import YEARS, RF
from src.Utilities import validate_portfolio

my_portfolio = Fetcher.create_portfolio()
tickers = my_portfolio.keys()

today_str = date.today().strftime("%Y-%m-%d")
end_date = pd.to_datetime(today_str)
start_date = end_date - pd.DateOffset(years=YEARS)

# GET PRICE AND RETURN DATA
price_df = Fetcher.fetch_price_data(my_portfolio, start_date, end_date)

log_returns = Utilities.get_log_returns(price_df, date = today_str)
mu = Utilities.get_expected_returns(log_returns)
sigma = Utilities.get_covariance_matrix(log_returns)

valid_tickers = list(price_df.columns)

### CREATE FUNCTION GET PORTFOLIO STATS FOR BLOCKS BELOW THAT RETURNS/DISPLAYS PD DF
valid_portfolio = validate_portfolio(my_portfolio, valid_tickers)
portfolio_weights = np.array(list(valid_portfolio.values()))
portfolio_expected_return = -Optimizer.neg_return(portfolio_weights, mu) + RF #adds rf because neg_return function subtracts it
portfolio_volatility = math.sqrt(portfolio_weights.T @ sigma @ portfolio_weights)
portfolio_sharpe = -Optimizer.neg_sharpe_ratio(portfolio_weights, mu, sigma)
portfolio_gross_leverage = np.sum(np.abs(portfolio_weights))

optimal_weights = Optimizer.get_optimal_weights(mu, sigma)
optimal_portfolio = Utilities.get_tangent_portfolio(valid_tickers, optimal_weights)
optimal_expected_return = -Optimizer.neg_return(optimal_weights, mu) + RF
optimal_volatility = math.sqrt(optimal_weights.T @ sigma @ optimal_weights)
optimal_sharpe = -Optimizer.neg_sharpe_ratio(optimal_weights, mu, sigma)
opt_gross_leverage = np.sum(np.abs(optimal_weights))

# DEAFAULT/INPUT PORTFOLIO
print("### Your portfolio: ###")
for i, ticker in enumerate(valid_tickers):
    print(f"{ticker}: {portfolio_weights[i] * 100:.2f}%")
print(f"Volatility: {portfolio_volatility * 100:.2f}%")
print(f"Expected return: {portfolio_expected_return * 100:.2f}%")
print(f"Sharpe ratio: {portfolio_sharpe:.3f}")
print(f"Gross leverage: {100*portfolio_gross_leverage:.2f}%")

# PRINT RESULTS OF ANALYSIS
# TANGENT PORTFOLIO:
print("### Tangent portfolio: ###")
for i, ticker in enumerate(valid_tickers):
    print(f"{ticker}: {optimal_weights[i] * 100:.2f}%")
print(f"Volatility: {optimal_volatility * 100:.2f}%")
print(f"Expected return: {optimal_expected_return * 100:.2f}%")
print(f"Sharpe ratio: {optimal_sharpe:.3f}")
print(f"Gross leverage: {100*opt_gross_leverage:.2f}%")

# CREATE AND DISPLAY EFFICIENT FRONTIER
frontier = Frontier.get_efficient_frontier(mu, sigma)
default_p = (portfolio_volatility, portfolio_expected_return)
optimal_p = (optimal_volatility, optimal_expected_return)
Plotters.plot_frontier(frontier, optimal_p, default_p)




