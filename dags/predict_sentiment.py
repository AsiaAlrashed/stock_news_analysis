import mage_ai.data_preparation.decorators as mage
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from models.sentiment_model import NewsClassifier

# Stocks ranked in descending order by news impact
LABEL_MAPPING = {0: "Negative", 1: "Neutral", 2: "Positive"}

# 🖥️ تحديد الجهاز المناسب (GPU أو CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔄 تحميل النموذج المدرب
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
model = NewsClassifier(bert_model).to(device)
model.load_state_dict(torch.load("models/news_classifier.pth", map_location=device))


def predict_news_sentiment(news_text, source_encoded):
    """
    يأخذ نص الخبر ومصدره، ويعيد التصنيف المتوقع.
    """
    inputs = tokenizer(
        news_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    source_tensor = torch.tensor([source_encoded], dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            source=source_tensor,
        )

    predicted_class = torch.argmax(outputs, dim=1).item()
    return LABEL_MAPPING[predicted_class]


@mage.transformer
def transform(df):
    """
    معالجة البيانات وإضافة توقع المشاعر إلى DataFrame.
    """
    # تحميل البيانات المشفرة للمصادر

    df_source = pd.read_csv("data/sources_stock.csv")
    df = df.merge(df_source, on="source", how="left")

    # التنبؤ بالمشاعر لكل خبر
    df["predicted_sentiment"] = df.apply(
        lambda row: predict_news_sentiment(row["headline"], row["source_encoded"]),
        axis=1,
    )

    # حفظ النتائج
    output_path = "data/predicted_news.csv"
    df.to_csv(output_path, index=False)

    return df
