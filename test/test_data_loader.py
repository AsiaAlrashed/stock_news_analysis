import pytest
import pandas as pd
import os
from src.data_loader import DataLoader

TEST_RAW_FILE = "tests/test_raw_news.csv"


class TestDataLoader:
    @pytest.fixture(scope="class")
    def data_loader(self):
        """
        Create a DataLoader object for testing.
        """
        return DataLoader(["TSLA"], news_limit=2)  # Select a few news items to test.

    def test_fetch_news(self, data_loader):
        """
        Test the news fetch and make sure it is not empty.
        """
        data_loader.fetch_news()
        assert len(data_loader.news_data) > 0, "No news brought back!"

    def test_fetch_stock_prices(self, data_loader):
        """
        Test the stock price fetching and make sure it is correct.
        """
        data_loader.fetch_news()
        data_loader.fetch_stock_prices()

        for news in data_loader.news_data:
            assert news["price_before"] is not None, "Price before news is missing!"
            assert news["price_after"] is not None, " Price after news is missing"

    def test_save_data(self, data_loader):
        """
        Test saving data to a CSV file.
        """
        data_loader.fetch_news()
        data_loader.fetch_stock_prices()
        data_loader.save_data(TEST_RAW_FILE)

        assert os.path.exists(TEST_RAW_FILE), "News file not saved!"
        df = pd.read_csv(TEST_RAW_FILE)
        assert not df.empty, "The file is empty after saving.!"

    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        """
        Delete files after tests are completed.
        """
        yield
        if os.path.exists(TEST_RAW_FILE):
            os.remove(TEST_RAW_FILE)
