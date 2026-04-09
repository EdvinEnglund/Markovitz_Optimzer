import yfinance as yf

def fetch_price_data(tickers, start="2010-01-01", end=None):
    """
    Returns closing prices for portfolio or list of tickers from yahoo finance in a pandas dataframe
    Rows/Index: date YYYY-MM-DD
    Columns: ticker closing price
    """
    if isinstance(tickers, dict):
        tickers = list(tickers.keys())

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )
    price_data = data.xs("Close", axis=1, level=0)
    # Drop columns with all NaN (invalid tickers)
    price_data = price_data.dropna(axis=1, how="all")
    return price_data








