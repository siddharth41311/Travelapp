import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

# Connect to MySQL database
load_dotenv()
NEW_URL = os.getenv("NEW_URL")
engine = create_engine(NEW_URL)

# Connect to MySQL database
with engine.connect() as conn:
    df = pd.read_sql('SELECT * FROM HotelBookings', conn)

# Drop non-essential columns
drop_columns = ['travelCode', 'User_ID', 'Check_in_Date', 'Check_Out_Date', 'Amenities', 'Total_Cost_per_Night']
df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

# Convert categorical features
df['Bedroom_Type'] = df['Bedroom_Type'].astype('category').cat.codes
df['Stars'] = df['Stars'].astype(int)

# Define features and target
y = df['Total_Cost']
X = df.drop(columns=['Total_Cost'])

# Standardize numerical features
numerical_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = ['Bedroom_Type']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Feature Selection using RFE
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(base_model, n_features_to_select=5)
X_train = rfe.fit_transform(X_train, y_train)
X_test = rfe.transform(X_test)

# Hyperparameter Optimization
def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['RandomForest', 'XGBoost'])
    
    if model_type == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model = RandomForestRegressor(**params, random_state=42)
    
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1)
        }
        model = XGBRegressor(**params, random_state=42)
    
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("Best Model:", best_params['model_type'])
print("Best Hyperparameters:", best_params)

# Train final model
if best_params['model_type'] == 'RandomForest':
    final_model = RandomForestRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'}, random_state=42)
else:
    final_model = XGBRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'}, random_state=42)

final_model.fit(X_train, y_train)

# Evaluate model
y_pred = final_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")

y_pred = final_model.predict(X_train)
mae = mean_absolute_error(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred)

print(f"Model Performance:\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")

# Save model
# Save model
joblib.dump(final_model, 'hotel_price_model.pkl')
print("Model trained and saved successfully!")
