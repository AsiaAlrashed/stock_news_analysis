import yfinance as yf
from datetime import datetime, timedelta


class StockFetcher:
    """
    Class to get stock prices from Yahoo Finance.
    """

    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def get_prices(self, date):
        """
        Gets the stock price before and after a specific date.
        param date: The date the news was published in %Y%m%dT%H%M%S format
        return: (price before, price after)
        """
        formatted_date = datetime.strptime(date, "%Y%m%dT%H%M%S")

        # Specify a date one day before and one day after the news is published.
        start_date = (formatted_date - timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = (formatted_date + timedelta(days=1)).strftime("%Y-%m-%d")

        # load data
        stock_data = self.stock.history(start=start_date, end=end_date)

        if stock_data.empty:
            return None, None

        # Extract price before and after
        price_before = stock_data.iloc[0]["Close"]
        price_after = stock_data.iloc[-1]["Close"]

        return price_before, price_after
