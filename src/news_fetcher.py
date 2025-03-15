import requests
from src.config import ALPHA_VANTAGE_API_KEY


class NewsFetcher:
    """
    Class to fetch news from Alpha Vantage API.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, ticker, limit=10):
        self.ticker = ticker
        self.limit = limit

    def get_news(self):
        """
        Retrieve news for a specific stock.
        return: News list
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": self.ticker,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "limit": self.limit,
        }

        response = requests.get(self.BASE_URL, params=params)
        news_data = response.json()

        if "feed" not in news_data:
            print(f"Error fetching news: {news_data.get('Note', 'Unknown issue')}")
            return []

        articles = news_data["feed"]
        return [
            {
                "headline": article["title"],
                "published_at": article["time_published"],
                "source": article["source"],
                "sentiment": article["overall_sentiment_label"],
                "name_stock": self.ticker,
            }
            for article in articles
        ]
