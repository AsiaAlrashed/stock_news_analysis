from news_fetcher import NewsFetcher
from stock_fetcher import StockFetcher
from feature_extraction import DataProcessor
from utils import DataUtils
import pandas as pd


class DataLoad:
    """
    Class to load news and stock price and save them to CSV.
    """

    def __init__(self, tickers, news_limit=10):
        self.tickers = tickers
        self.news_limit = news_limit
        self.news_data = []

    def fetch_news(self):
        """
        Fetch news for all selected stocks.
        """
        for ticker in self.tickers:
            news_fetcher = NewsFetcher(ticker, self.news_limit)
            self.news_data.extend(news_fetcher.get_news())

    def fetch_stock_prices(self):
        """
        Get stock prices before and after news releases.
        """
        for news in self.news_data:
            stock_fetcher = StockFetcher(news["name_stock"])
            price_before, price_after = stock_fetcher.get_prices(news["published_at"])

            news["price_before"] = price_before
            news["price_after"] = price_after

    def save_data(
        self, file_path="F:\\final_project\\stock_news_analysis\\data\\raw_news.csv"
    ):
        """
        Save news with stock price in CSV.
        """
        DataUtils.save_to_csv(self.news_data, file_path)

    def process_data(
        self,
        raw_data_path="F:\\final_project\\stock_news_analysis\\data\\raw_news.csv",
        processed_data_path="F:\\final_project\\stock_news_analysis\\data\\processed_news.csv",
    ):
        """
        Processing raw data using DataProcessor.
        """
        processor = DataProcessor()
        return processor.process_data(raw_data_path, processed_data_path)

    def run(self):
        """
        Run the full download steps.
        """
        print(" Fetching news ...")
        self.fetch_news()
        print("News brought")

        print("Fetching stock price ...")
        self.fetch_stock_prices()
        print("Prices updated")

        print("Saving data ...")
        self.save_data()
        print("Data saved successfully")

        print("Data is being processed ...")
        self.process_data()
        print("Final data is ready ")


# تشغيل تحميل البيانات
if __name__ == "__main__":
    tickers = ["TSLA", "AAPL", "MSFT", "AMZN"]
    data_loader = DataLoad(tickers, news_limit=1000)
    data_loader.run()

