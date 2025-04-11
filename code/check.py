import joblib
import numpy as np
import pandas as pd

# ✅ Load Saved Model, Preprocessor & Feature Selection Indices
final_model = joblib.load('hotel_final_price_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
important_indices = np.load('important_indices.npy')

# ✅ Define Feature Names (Ensure Correct Order)
feature_names = ['Hotel_Name', 'Hotel_Star_Rating', 'City', 'Nights_Stayed', 'Guests', 'Room_Type']

# ✅ Function to Predict Hotel Cost
def predict_hotel_price(input_data):
    """
    Predicts the total cost for a given hotel booking input.
    :param input_data: List of input values in the same order as feature_names.
    :return: Predicted hotel cost.
    """
    # Convert Input Data to DataFrame
    df_input = pd.DataFrame([input_data], columns=feature_names)

    # ✅ Apply Preprocessing (Encoding)
    processed_input = preprocessor.transform(df_input)

    # ✅ Apply Feature Selection (Use Only Important Features)
    processed_input = processed_input[:, important_indices]

    # ✅ Make Prediction
    predicted_price = final_model.predict(processed_input)

    return predicted_price[0]

# ✅ Example Input (Change This for Different Predictions)
example_input = ["Fairfield by Marriott Visakhapatnam", 4, "Visakhapatnam", 3, 2, "Deluxe"]

# ✅ Get Prediction
predicted_price = predict_hotel_price(example_input)
print(f"Predicted Total Cost: ${predicted_price:.2f}")


