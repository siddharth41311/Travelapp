import joblib
import numpy as np
import pandas as pd

# Load models
preprocessor = joblib.load('preprocessor.pkl')
selector = joblib.load('feature_selector.pkl')
model = joblib.load('hotel_final_price_model.pkl')

# Taking user input
custom_data = {
    'travelCode': [input("Enter Travel Code: ")],  # Will be dropped later
    'User_ID': [input("Enter User ID: ")],  # Will be dropped later
    'Departure': [input("Enter Departure Location: ")],
    'Arrival': [input("Enter Arrival Location: ")],
    'Hotel': [input("Enter Hotel Name: ")],
    'Stars': [int(input("Enter Hotel Stars (1-5): "))],  # Will be converted to category
    'Check_in_Date': [input("Enter Check-in Date (YYYY-MM-DD): ")],  # Will be dropped later
    'Bedroom_Type': [input("Enter Bedroom Type: ")],
    'Room_Price_per_Night': [float(input("Enter Room Price per Night: "))],
    'Number_of_Adults': [int(input("Enter Number of Adults: "))],
    'Number_of_Children': [int(input("Enter Number of Children: "))],
    'Number_of_Bedrooms': [int(input("Enter Number of Bedrooms: "))],
    'Total_Price_per_Night': [float(input("Enter Total Price per Night: "))],
    'Amenities': [input("Enter Amenities (comma-separated): ")],  # Will be dropped later
    'Days_of_Stay': [int(input("Enter Days of Stay: "))],
    'Total_Cost': [float(input("Enter Total Cost: "))],  # Will be dropped before prediction
    'Check_Out_Date': [input("Enter Check-out Date (YYYY-MM-DD): ")]  # Will be dropped later
}

# Convert input to DataFrame
custom_df = pd.DataFrame(custom_data)

# Convert 'Stars' column to categorical
custom_df['Stars'] = custom_df['Stars'].astype('category')

# Drop unnecessary columns before preprocessing
drop_columns = ['travelCode', 'User_ID', 'Check_in_Date', 'Check_Out_Date', 'Amenities', 'Total_Cost']
custom_df.drop(columns=[col for col in drop_columns if col in custom_df.columns], inplace=True)

# Apply preprocessing and feature selection
X_custom = preprocessor.transform(custom_df)
X_custom = selector.transform(X_custom)

# Make predictions
predictions = model.predict(X_custom)
print("Predicted Hotel Costs:", predictions)

