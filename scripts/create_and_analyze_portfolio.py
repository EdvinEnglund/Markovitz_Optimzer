from src import Fetcher
from src import Optimizer
from src import Utilities
from src import Plotters
from datetime import date
import pandas as pd
from config import YEARS, DEFAULT_PORTFOLIO_PATH

my_portfolio = Fetcher.create_portfolio()
if len(my_portfolio) != 0:
    save = input(f"Do you want to save portfolio? (y/n): ").strip().lower()
    if save == "y":
        save_path = f"data/{input("Enter the name of your saved portfolio to save: ")}.json"
        Fetcher.save_portfolio(my_portfolio, save_path, default_path = DEFAULT_PORTFOLIO_PATH)

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

valid_portfolio = Utilities.validate_portfolio(my_portfolio, valid_tickers)
my_portfolio_stats = Utilities.get_stats(valid_portfolio, log_returns)

optimal_portfolio = Utilities.get_tangent_portfolio(log_returns)
optimal_portfolio_stats = Utilities.get_stats(optimal_portfolio, log_returns)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', None)

print("Your portfolio stats:")
print(my_portfolio_stats)
print("Optimal portfolio stats:")
print(optimal_portfolio_stats)

# CREATE AND DISPLAY EFFICIENT FRONTIER
frontier = Optimizer.get_efficient_frontier(mu, sigma)
default_p = (
    my_portfolio_stats.at[my_portfolio_stats.index[0], "Volatility"],
    my_portfolio_stats.at[my_portfolio_stats.index[0], "Expected Return"]
)
optimal_p = (
    optimal_portfolio_stats.at[optimal_portfolio_stats.index[0], "Volatility"],
    optimal_portfolio_stats.at[optimal_portfolio_stats.index[0], "Expected Return"]
)
Plotters.plot_frontier(frontier, optimal_p, default_p)




