import joblib
from sqlalchemy import create_engine
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

preprocessor = joblib.load('preprocessor.pkl')
selector = joblib.load('feature_selector.pkl')
model = joblib.load('hotel_final_price_model.pkl')

load_dotenv()
NEW_URL = os.getenv("NEW_URL")
engine = create_engine(NEW_URL)


with engine.connect() as conn:
    inference_df = pd.read_sql('SELECT * FROM HotelBooking', conn)

drop_columns = ['travelCode', 'User_ID', 'Check_in_Date', 'Check_Out_Date', 'Amenities', 'Total_Cost_per_Night']
inference_df.drop(columns=[col for col in drop_columns if col in inference_df.columns], inplace=True)

# Apply preprocessing and feature selection
inference_df['Stars'] = inference_df['Stars'].astype('category')
inference_df.drop('Total_Cost',axis=1,inplace=True)
X_new = preprocessor.transform(inference_df)
X_new = selector.transform(X_new)

# Make predictions
predictions = model.predict(X_new)
print("Predicted Hotel Costs:", predictions)
