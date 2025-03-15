from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

app = FastAPI()

MAGE_PIPELINE_URL = "http://localhost:6789/api/pipelines/stock_news_pipeline/triggers"


@app.post("/fetch-news/")
def fetch_news(ticker: str, limit: int):
    response = requests.post(MAGE_PIPELINE_URL, json={"ticker": ticker, "limit": limit})

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="خطأ في تشغيل Pipeline")

    df = pd.read_csv("data/ranked_stocks.csv")  # تحميل البيانات الناتجة

    # 🎨 رسم المخطط البياني
    plt.figure(figsize=(10, 6))
    plt.barh(df["name_stock"], df["weighted_sentiment"], color="blue")
    plt.xlabel("التقييم")
    plt.ylabel("السهم")
    plt.title("ترتيب الأسهم")
    plt.gca().invert_yaxis()

    # حفظ الصورة وإرجاعها
    img_io = BytesIO()
    plt.savefig(img_io, format="png")
    img_io.seek(0)
    return Response(content=img_io.getvalue(), media_type="image/png")
