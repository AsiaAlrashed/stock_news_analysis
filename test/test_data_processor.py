import pytest
import pandas as pd
import os
from src.data_processor import DataProcessor

TEST_RAW_FILE = "tests/test_raw_news.csv"
TEST_PROCESSED_FILE = "tests/test_processed_news.csv"


class TestDataProcessor:
    def test_process_data(self):
        """
        Test data processing and verify impact calculation and news classification.
        """
        processor = DataProcessor()
        df = processor.process_data(TEST_RAW_FILE, TEST_PROCESSED_FILE)

        assert os.path.exists(
            TEST_PROCESSED_FILE
        ), "The processed data file was not saved!"
        assert "price_change (%)" in df.columns, "The impact is not calculated!"
        assert "final_label" in df.columns, "News not classficated!"

    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        """
        Delete files after the test is complete.
        """
        yield
        if os.path.exists(TEST_PROCESSED_FILE):
            os.remove(TEST_PROCESSED_FILE)
