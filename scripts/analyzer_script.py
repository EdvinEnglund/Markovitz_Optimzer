from src.Frontier import *
from src.Frontier import *
from src.Plotters import *
from config import RF, SHORT_ALLOWED, SHORT_COST, DEFAULT_L, YEARS, DEFAULT_PORTFOLIO_PATH

#CONSTANTS
rf = RF
short_allowed = SHORT_ALLOWED
short_cost = SHORT_COST
L = DEFAULT_L
years = YEARS
default_portfolio_path = DEFAULT_PORTFOLIO_PATH

default_list = get_tickers_from_json(default_portfolio_path)

print(default_portfolio_path)
print(default_list)
tickers = set_tickers(default_list)

today_str = date.today().strftime("%Y-%m-%d")
end_date = pd.to_datetime(today_str)
start_date = end_date - pd.DateOffset(years=years)

price_df = fetch_price_data(tickers, start=start_date, end=end_date)
valid_tickers, mu, sigma = get_stats(price_df, today_str, years = years)

w = get_tangent_portfolio(mu, sigma, rf=rf, short_allowed=short_allowed, L=L, short_cost=short_cost)
ret = -neg_return(w, mu, rf=rf, short_cost = short_cost)
var =  w.T @ sigma @ w
sharpe = - neg_sharpe_ratio(w, mu, sigma, rf = rf, short_cost = short_cost)

frontier = get_efficient_frontier(mu, sigma, short_allowed=short_allowed)

plot_frontier(frontier, L)

print("### Tangent portfolio: ###")
for i, ticker in enumerate(valid_tickers):
    print(f"{ticker}: {w[i] * 100:.2f}%")
print(f"Volatility: {math.sqrt(var) * 100:.2f}%")
print(f"Expected return: {ret * 100:.2f}%")
print(f"Sharpe ratio: {sharpe:.3f}")

