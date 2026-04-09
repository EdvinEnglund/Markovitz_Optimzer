from config import RF, SHORT_ALLOWED, SHORT_COST, DEFAULT_L, YEARS, DEFAULT_PORTFOLIO_PATH

# LOAD CONFIG CONSTANTS
rf = RF
short_allowed = SHORT_ALLOWED
short_cost = SHORT_COST
L = DEFAULT_L
years = YEARS
default_portfolio_path = DEFAULT_PORTFOLIO_PATH

# PROMPT USER TO ADD TICKERS OR USE DEFAULT PORTFOLIO
portfolio = set_portfolio(default_portfolio_path)
tickers = list(portfolio.keys())

# USE TODAY'S DATA FOR PRICE DATA
today_str = date.today().strftime("%Y-%m-%d")
end_date = pd.to_datetime(today_str)
start_date = end_date - pd.DateOffset(years=years)

# ETL: YAHOO FINANCE DATA FETCHING -> NUMPY DATA FOR OPTIMIZERS
price_df = fetch_price_data(tickers, start=start_date, end=end_date)
log_returns = get_log_returns(price_df, today_str, years = years)
mu = get_expected_returns(log_returns)
sigma = get_covariance_matrix(log_returns)

# REMOVE INVALID TICKERS FROM DEFAULT/INPUT PORTFOLIO
valid_tickers = list(price_df.columns)
portfolio = validate_portfolio(portfolio, valid_tickers)
default_weights = list(portfolio.values())

# ANALYSIS:
# GET TANGENT PORTFOLIO AND STATS
opt_w = get_tangent_portfolio(mu, sigma, rf=rf, short_allowed=short_allowed, L=L, short_cost=short_cost)
opt_ret = -neg_return(opt_w, mu, rf=rf, short_cost = short_cost) + rf #adds rf because neg_return function subtracts it
opt_vol =  math.sqrt(opt_w.T @ sigma @ opt_w)
opt_sharpe = -neg_sharpe_ratio(opt_w, mu, sigma, rf = rf, short_cost = short_cost)
opt_gross_leverage = np.sum(np.abs(opt_w))

# DEFAULT/INPUT PORTFOLIO
def_w = np.array(default_weights)
def_ret = -neg_return(def_w, mu, rf=rf, short_cost = short_cost) + rf #adds rf because neg_return function subtracts it
def_vol =  math.sqrt(def_w.T @ sigma @ def_w)
def_sharpe = -neg_sharpe_ratio(def_w, mu, sigma, rf = rf, short_cost = short_cost)
def_gross_leverage = np.sum(np.abs(def_w))

# CREATE AND DISPLAY EFFICIENT FRONTIER
frontier = get_efficient_frontier(mu, sigma, short_allowed=short_allowed, L = L, short_cost=short_cost)
optimal_p = (opt_vol, opt_ret)
default_p = (def_vol, def_ret)
plot_frontier(frontier, optimal_p, default_p)

# PRINT RESULTS OF ANALYSIS
# TANGENT PORTFOLIO:
print("### Tangent portfolio: ###")
for i, ticker in enumerate(valid_tickers):
    print(f"{ticker}: {opt_w[i] * 100:.2f}%")
print(f"Volatility: {math.sqrt(opt_vol) * 100:.2f}%")
print(f"Expected return: {opt_ret * 100:.2f}%")
print(f"Sharpe ratio: {opt_sharpe:.3f}")
print(f"Gross leverage: {100*opt_gross_leverage:.2f}%")

# DEAFAULT/INPUT PORTFOLIO
print("### Your portfolio: ###")
for i, ticker in enumerate(valid_tickers):
    print(f"{ticker}: {def_w[i] * 100:.2f}%")
print(f"Volatility: {math.sqrt(def_vol) * 100:.2f}%")
print(f"Expected return: {def_ret * 100:.2f}%")
print(f"Sharpe ratio: {def_sharpe:.3f}")
print(f"Gross leverage: {100*def_gross_leverage:.2f}%")

