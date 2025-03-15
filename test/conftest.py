import pytest
from src.data_loader import DataLoader


@pytest.fixture(scope="session")
def test_data_loader():
    """
    Create one DataLoader object for each test session.
    """
    loader = DataLoader(["TSLA"], news_limit=2)
    loader.fetch_news()
    loader.fetch_stock_prices()
    return loader
