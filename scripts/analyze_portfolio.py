import argparse
from src.portfolio_analyzer import optimizer, utilities, fetcher, plotters, portfolio_tools, analyzer
from datetime import date
import pandas as pd
from config import YEARS, RF, SHORT_ALLOWED, SHORT_COST, DEFAULT_L, BENCHMARK_PORTFOLIO_PATH
def main(my_portfolio = "CREATE",
         years = YEARS,
         rf = RF,
         short_allowed = SHORT_ALLOWED,
         L = DEFAULT_L,
         short_cost = SHORT_COST,
         benchmark_path = BENCHMARK_PORTFOLIO_PATH,
         benchmark = True,
         save_portfolio = False,
         save_optimal = False,
         show_plots = True,
         save_plots = False,
         show_stats = True,
         ):
    if my_portfolio == "CREATE":
        my_portfolio = portfolio_tools.create_portfolio(save=save_portfolio)

    if isinstance(my_portfolio, str):
        # if file_path
        my_portfolio = portfolio_tools.get_portfolio(my_portfolio)

    tickers = my_portfolio.keys()
    today_str = date.today().strftime("%Y-%m-%d")
    end_date = pd.to_datetime(today_str)
    start_date = end_date - pd.DateOffset(years=years)
    # GET PRICE AND RETURN DATA
    price_df = fetcher.fetch_price_data(my_portfolio, start_date, end_date)

    log_returns = fetcher.get_log_returns(price_df, date=today_str, years=years)

    #mu = utilities.get_expected_returns(log_returns)
    #sigma = utilities.get_covariance_matrix(log_returns)
    mu, sigma = utilities.get_greeks(log_returns)
    valid_tickers = list(price_df.columns)

    valid_portfolio = portfolio_tools.validate_portfolio(my_portfolio, valid_tickers)
    my_portfolio_stats = analyzer.get_stats(valid_portfolio, log_returns, rf = rf, short_cost=short_cost)
    optimal_portfolio = analyzer.get_tangent_portfolio(log_returns,
                                                        rf = rf,
                                                        short_allowed = short_allowed,
                                                        L = L,
                                                        short_cost = short_cost)
    if save_optimal:
        path = f"data/{input("Enter the name (w/o. extension) of the tangent portfolio to save at /data: ")}.json"
        portfolio_tools.save_portfolio(optimal_portfolio, path)
    optimal_portfolio_stats = analyzer.get_stats(optimal_portfolio, log_returns, rf = rf)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_columns', None)
    if benchmark:
        benchmark_portfolio = portfolio_tools.get_portfolio(benchmark_path)
        benchmark_price_df = fetcher.fetch_price_data(benchmark_portfolio, start_date, end_date)
        valid_benchmark_tickers = list(benchmark_price_df.columns)
        valid_benchmark_portfolio = portfolio_tools.validate_portfolio(benchmark_portfolio, valid_benchmark_tickers)
        benchmark_log_returns = fetcher.get_log_returns(benchmark_price_df, date=today_str, years=years)
        benchmark_portfolio_stats = analyzer.get_stats(valid_benchmark_portfolio, benchmark_log_returns)

    if show_stats:
        print("Your portfolio stats:")
        print(my_portfolio_stats)
        print("Optimal portfolio stats:")
        print(optimal_portfolio_stats)
        if benchmark:
            print("Benchmark portfolio stats:")
            print(benchmark_portfolio_stats)

    random_points = analyzer.get_randomized_portfolio_points(valid_portfolio, log_returns)
    random_portfolio_stats = utilities.get_mean_portfolio_stats(random_points)
    print("Mean of random portfolio stats:")
    print(random_portfolio_stats)

    # CREATE AND DISPLAY EFFICIENT FRONTIER
    frontier = optimizer.get_efficient_frontier(mu,
                                                sigma,
                                                short_allowed = short_allowed,
                                                rf = rf,
                                                L = L,
                                                short_cost = short_cost)
    default_p = (
        my_portfolio_stats.at[my_portfolio_stats.index[0], "Volatility"],
        my_portfolio_stats.at[my_portfolio_stats.index[0], "Expected Return"]
    )
    optimal_p = (
        optimal_portfolio_stats.at[optimal_portfolio_stats.index[0], "Volatility"],
        optimal_portfolio_stats.at[optimal_portfolio_stats.index[0], "Expected Return"]
    )
    if benchmark:
        benchmark_p = (
            benchmark_portfolio_stats.at[benchmark_portfolio_stats.index[0], "Volatility"],
            benchmark_portfolio_stats.at[benchmark_portfolio_stats.index[0], "Expected Return"]
        )
    else: benchmark_p = None

    if show_plots:
        plotters.plot_frontier(frontier,
                               optimal_p,
                               default_p,
                               save=save_plots,
                               show=show_plots,
                               random_points=random_points,
                               benchmark_p = benchmark_p)

        plotters.plot_heatmap(valid_tickers, sigma, save=save_plots, show=show_plots)
        plotters.plot_PCA(log_returns, save=save_plots, show=show_plots)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", default="CREATE")
    parser.add_argument("--years", type=int, default=YEARS)
    parser.add_argument("--rf", type=float, default=RF)
    parser.add_argument("--short_allowed", action="store_true", default=SHORT_ALLOWED)
    parser.add_argument("--L", type=float, default=DEFAULT_L)
    parser.add_argument("--short_cost", type=float, default=SHORT_COST)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark_path", default=BENCHMARK_PORTFOLIO_PATH)
    parser.add_argument("--save-portfolio", action="store_true", help="Save created portfolio to file")
    parser.add_argument("--save-optimal", action="store_true", help="Save optimal portfolio to file")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--no-stats", action="store_true", help="Disable stats printing")

    args = parser.parse_args()
    main(
        my_portfolio=args.portfolio,
        years=args.years,
        rf=args.rf,
        short_allowed=args.short_allowed,
        L=args.L,
        short_cost=args.short_cost,
        benchmark=args.benchmark,
        save_portfolio=args.save_portfolio,
        save_optimal=args.save_optimal,
        show_plots=not args.no_plots,
        save_plots=args.save_plots,
        show_stats=not args.no_stats,
    )
