from src.Fetcher import *
from src.Frontier import *
from src.Plotters import *
from config import RF, SHORT_ALLOWED, SHORT_COST, DEFAULT_L, YEARS, DEFAULT_PORTFOLIO_PATH

# LOAD CONFIG CONSTANTS
rf = RF
short_allowed = SHORT_ALLOWED
short_cost = SHORT_COST
L = DEFAULT_L
years = YEARS
default_portfolio_path = DEFAULT_PORTFOLIO_PATH

# LOAD DEFAULT PORTFOLIO
default_portfolio = get_default_portfolio(default_portfolio_path)
default_list = list(default_portfolio.keys())
default_weights = list(default_portfolio.values())

# PROMPT USER TO ADD TICKERS OR USE DEFAULT PORTFOLIO
#tickers = set_tickers(default_list)

portfolio = set_portfolio(default_portfolio_path)
tickers = list(portfolio.keys())
weights = list(portfolio.values())

# USE TODAYS DATA FOR PRICE DATA
today_str = date.today().strftime("%Y-%m-%d")
end_date = pd.to_datetime(today_str)
start_date = end_date - pd.DateOffset(years=years)

# ETL: YAHOO FINANCE DATA FETCHING -> CLEAN NUMPY DATA FOR OPTIMIZERS
price_df = fetch_price_data(tickers, start=start_date, end=end_date)
log_returns = get_log_returns(price_df, today_str, years = years)
mu = get_expected_returns(log_returns)
sigma = get_covariance_matrix(log_returns)

# ANALYSIS: GET TANGENT PORTFOLIO AND STATS
w = get_tangent_portfolio(mu, sigma, rf=rf, short_allowed=short_allowed, L=L, short_cost=short_cost)
ret = -neg_return(w, mu, rf=rf, short_cost = short_cost) + rf #adds rf because the objective function subtracts it
vol =  math.sqrt(w.T @ sigma @ w)
sharpe = - neg_sharpe_ratio(w, mu, sigma, rf = rf, short_cost = short_cost)
net_leverage = np.sum(np.abs(w))

# CREATE AND DISPLAY EFFICIENT FRONTIER
frontier = get_efficient_frontier(mu, sigma, short_allowed=short_allowed, L = L, short_cost=short_cost)
optimal_p = (vol, ret)
plot_frontier(frontier, L, optimal_p)

# PRINT RESULTS OF ANALYSIS
valid_tickers = list(price_df.columns)
print("### Tangent portfolio: ###")
for i, ticker in enumerate(valid_tickers):
    print(f"{ticker}: {w[i] * 100:.2f}%")
print(f"Volatility: {math.sqrt(vol) * 100:.2f}%")
print(f"Expected return: {ret * 100:.2f}%")
print(f"Sharpe ratio: {sharpe:.3f}")
print(f"Gross leverage: {100*net_leverage:.2f}%")

