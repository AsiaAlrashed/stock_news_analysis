Stock Market Sentiment Analysis & Ranking System

Overview

This project leverages AI-driven sentiment analysis to evaluate financial news and assess their impact on stock prices. It ranks stocks based on sentiment trends to provide investment recommendations.

Features

Sentiment Analysis: Uses DistilBERT to analyze financial news headlines.

Stock Ranking System: Assigns a weighted sentiment score to each stock to suggest investment opportunities.

End-to-End ML Pipeline: Implemented using Mage for workflow automation.

API Integration: A FastAPI backend serves real-time analysis results.

Technologies Used

Python (transformer, Pandas, NumPy)

Mage (for ML pipeline orchestration)

FastAPI (for API deployment)

DistilBERT (for NLP-based sentiment analysis)

csv file (for data storage)

How It Works

Data Processing: News headlines and stock data are extracted and preprocessed.

Sentiment Classification: Headlines are classified as Positive, Neutral, or Negative.

Stock Ranking: Stocks are ranked based on their weighted sentiment score.

API Serving: The results are served via a fast api for integration with investment platforms.

Installation & Setup
1. Install dependencies
   
   pip install -r requirements.txt
   
2. Train model
   
   python models.train_model.py
   
3. Run the FastAPI server
   
   uvicorn src.api:app --reload

Stock Ranking Example Output

![لقطة شاشة_15-3-2025_2109_chatgpt com](https://github.com/user-attachments/assets/15195dd9-e033-4a79-a7a4-a1449221b3a8)


Future Improvements
 ~Develop a frontend dashboard for better visualization.
