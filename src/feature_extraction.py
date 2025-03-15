import pandas as pd
from datetime import datetime


class DataProcessor:
    """
    Class for processing extracted data, such as impact calculation, time transformation, and news classification.
    """

    @staticmethod
    def calculate_impact(price_before, price_after):
        """
        Calculates the actual impact of the news based on the change in the stock price.
        param price_before: The stock price before the news
        param price_after: The stock price after the news
        return: The percentage change in price
        """
        if price_before and price_after:
            return ((price_after - price_before) / price_before) * 100
        return None

    @staticmethod
    def classify_news(impact):
        """
        Classify news based on the actual impact on the stock price.
        """
        if impact is None:
            return "Neutral"
        elif impact > 1.0:
            return "Positive"
        elif impact < -1.0:
            return "Negative"
        else:
            return "Neutral"

    @staticmethod
    def convert_datetime(timestamp):
        """
        Convert date from Alpha Vantage format to `datetime` in Pandas.
        """
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%S")

    def process_data(self, raw_data_path, processed_data_path):
        """
        Process and save the extracted data.
        param raw_data_path: Path to the raw news file (CSV)
        param processed_data_path: Path to save the post-processed data (CSV)
        """
        print("Data is being processed ...")

        df = pd.read_csv(raw_data_path)

        df["published_at"] = df["published_at"].apply(self.convert_datetime)

        df["price_change (%)"] = df.apply(
            lambda row: self.calculate_impact(row["price_before"], row["price_after"]),
            axis=1,
        )

        df["final_label"] = df["price_change (%)"].apply(self.classify_news)

        # Save data after processing
        df.to_csv(processed_data_path, index=False)
        print(f"The data after processing is saved in{processed_data_path}")

        return df
