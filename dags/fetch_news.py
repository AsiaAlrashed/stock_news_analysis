import mage_ai.data_preparation.decorators as mage
import pandas as pd
from src.news_fetcher import NewsFetcher
from datetime import datetime


@mage.data_loader
def load_data(*args, **kwargs):
    """
    Fetch stock news using NewsFetcher class and save to CSV.
    """
    tickers = ["TSLA", "AAPL", "MSFT", "AMZN"]
    all_news = []

    for ticker in tickers:
        fetcher = NewsFetcher(ticker, limit=10)
        news_articles = fetcher.get_news()

        for article in news_articles:
            all_news.append(article)

    df = pd.DataFrame(all_news)
    df["published_at"] = pd.to_datetime(df["published_at"], format="%Y%m%dT%H%M%S")

    df = df.sort_values(by="published_at", ascending=False)

    csv_path = "data/news_new.csv"
    df.to_csv(csv_path, index=False)
    return df
