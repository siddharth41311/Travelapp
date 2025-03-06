# Load the saved pipeline
import joblib
from sqlalchemy import create_engine
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from review_pipeline import preprocess_reviews


pipeline = joblib.load('sentiment_analysis_pipeline.pkl')
load_dotenv()
DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL)

with engine.connect() as conn:
    inference_hotel = pd.read_sql('SELECT * FROM hotel_review', conn)

def predict_sentiment(review_text):
    review_df = pd.DataFrame({"review": [review_text]})  # Convert to DataFrame
    prediction = pipeline.predict(review_df["review"])[0]  # Predict
    # sentiment_labels = {1: "Positive", 0: "Negative", 2: "Neutral"}
    return prediction


# Predict sentiment for each review individually
inference_hotel["Predicted_Sentiment"] = inference_hotel["review"].apply(lambda x: predict_sentiment(x))


print(inference_hotel)