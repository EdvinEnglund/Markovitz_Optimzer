from src.Frontier import *
from src.Frontier import *
from src.Plotters import *
from config import rf, short_allowed, short_cost, l, years, default_portfolio_path

#CONSTANTS
RF = rf
SHORT_ALLOWED = short_allowed
SHORT_COST = short_cost
L = l
YEARS = years
PORTFOLIO_PATH = default_portfolio_path

default_list = get_tickers_from_json(PORTFOLIO_PATH)

tickers = set_tickers(default_list)

today_str = date.today().strftime("%Y-%m-%d")
end_date = pd.to_datetime(today_str)
start_date = end_date - pd.DateOffset(years=YEARS)

price_df = fetch_price_data(tickers, start=start_date, end=end_date)
valid_tickers, mu, sigma = get_stats(price_df, today_str, years = YEARS)

w = get_tangent_portfolio(mu, sigma, rf=RF, short_allowed=SHORT_ALLOWED, L=L, short_cost=SHORT_COST)
ret = -neg_return(w, mu, rf=RF, short_cost = SHORT_COST)
var =  w.T @ sigma @ w
sharpe = - neg_sharpe_ratio(w, mu, sigma, rf = RF, short_cost = SHORT_COST)

frontier = get_efficient_frontier(mu, sigma, short_allowed=short_allowed)

plot_frontier(frontier, L)

print("### Tangent portfolio: ###")
for i, ticker in enumerate(valid_tickers):
    print(f"{ticker}: {w[i] * 100:.2f}%")
print(f"Volatility: {math.sqrt(var) * 100:.2f}%")
print(f"Expected return: {ret * 100:.2f}%")
print(f"Sharpe ratio: {sharpe:.3f}")

