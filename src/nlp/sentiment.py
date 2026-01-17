from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

def get_sentiment(text):
    result = sentiment_model(text)[0]
    return result["label"], result["score"]
