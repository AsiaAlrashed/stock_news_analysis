import mage_ai.data_preparation.decorators as mage
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set numerical values ​​for news sentiment
SENTIMENT_MAPPING = {"Positive": 1, "Neutral": 0, "Negative": -1}

# Determine the half-life of new news (the shorter it is, the greater the impact of new news)

HALF_LIFE_HOURS = 24  # #Determine the half-life of new news (the shorter it is, the greater the impact of new news)


def time_decay_weight(published_at):
    """
    Calculating the time weight of the news, so that more recent news has more impact.
    """
    current_time = datetime.utcnow()
    published_time = datetime.strptime(str(published_at), "%Y-%m-%d %H:%M:%S")

    time_difference = (
        current_time - published_time
    ).total_seconds() / 3600  # Difference in hours
    decay_factor = 0.5 ** (
        time_difference / HALF_LIFE_HOURS
    )  # Exponential Decrease Calculation

    return decay_factor


@mage.transformer
def rank_stock(df: pd.DataFrame):
    """
    Calculates stock ranking based on news and its time impact.
    """
    # Convert text values ​​in `predicted_sentiment` to numeric values
    df["sentiment_score"] = df["predicted_sentiment"].map(SENTIMENT_MAPPING)

    # Calculate the time weight for each news item
    df["time_weight"] = df["published_at"].apply(time_decay_weight)

    # Modify news effect by time
    df["weighted_sentiment"] = df["sentiment_score"] * df["time_weight"]

    # Calculate the average valuation for each stock
    stock_ranking = df.groupby("name_stock")["weighted_sentiment"].mean().reset_index()

    # Stocks ranked in descending order by news impact
    stock_ranking = stock_ranking.sort_values(by="weighted_sentiment", ascending=False)

    output_path = "data/ranked_stocks.csv"
    stock_ranking.to_csv(output_path, index=False)
    print(f"Stock ranking saved in: {output_path}")

    return stock_ranking
